"""Quadruped Joytick environment adapted for PPO training."""

from dataclasses import dataclass, field
from typing import Any

# Supporting
from etils.epath import Path

# Math
import jax
import jax.numpy as jnp

# Brax
from brax import math
from brax.base import Motion, System, Transform
from brax.base import State as PipelineState
from brax.envs.base import State


from quadruped_mjx_rl.configs import EnvironmentConfig
from quadruped_mjx_rl.configs import RobotConfig
from quadruped_mjx_rl.configs.config_classes import environment_config_classes
from quadruped_mjx_rl.environments.ppo_enhanced import QuadrupedJoystickEnhancedEnv


_ENVIRONMENT_CLASS = "TeacherStudent"


@dataclass
class TeacherStudentEnvironmentConfig(EnvironmentConfig["QuadrupedJoystickTeacherStudentEnv"]):
    environment_class: str = _ENVIRONMENT_CLASS


class QuadrupedJoystickTeacherStudentEnv(QuadrupedJoystickEnhancedEnv):

    def __init__(
        self,
        environment_config: TeacherStudentEnvironmentConfig,
        robot_config: RobotConfig,
        init_scene_path: Path,
    ):
        super().__init__(environment_config, robot_config, init_scene_path)

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)

        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = super().step(state, action)

        return state

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
        obs: jax.Array | dict[str, jax.Array],
    ) -> jax.Array | dict[str, jax.Array]:
        assert isinstance(obs, dict)
        #TODO from here on
        obs_history = obs

        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        obs = jnp.concatenate(
            [
                jnp.array([local_rpyrate[2]]) * 0.25,  # yaw rate
                math.rotate(jnp.array([0, 0, -1]), inv_torso_rot),  # projected gravity
                state_info["command"] * jnp.array([2.0, 2.0, 0.25]),  # command
                pipeline_state.q[7:] - self._default_pose,  # motor angles
                state_info["last_act"],  # last action
            ]
        )

        # clip, noise
        obs_noise = (
            self._obs_noise_config.general_noise
            * jax.random.uniform(state_info["rng"], obs.shape, minval=-1, maxval=1)
        )
        obs = jnp.clip(obs, -100.0, 100.0) + obs_noise

        # stack observations through time
        obs = jnp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return obs

    def _get_rewards(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        action: jax.Array,
        done: jax.Array,
    ):
        rewards = super()._get_rewards(pipeline_state, state_info, action)
        x, xd = pipeline_state.x, pipeline_state.xd
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state_info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state_info["last_contact"]
        first_contact = (state_info["feet_air_time"] > 0) * contact_filt_mm
        state_info["feet_air_time"] += self.dt

        rewards["lin_vel_z"] = self._reward_lin_vel_z(xd)
        rewards["ang_vel_xy"] = self._reward_ang_vel_xy(xd)
        rewards["orientation"] = self._reward_orientation(x)
        rewards["torques"] = self._reward_torques(pipeline_state.qfrc_actuator)
        rewards["action_rate"] = self._reward_action_rate(action, state_info["last_act"])
        rewards["stand_still"] = self._reward_stand_still(
            state_info["command"], joint_angles
        )
        rewards["feet_air_time"] = self._reward_feet_air_time(
            state_info["feet_air_time"], first_contact, state_info["command"]
        )
        rewards["foot_slip"] = self._reward_foot_slip(pipeline_state, contact_filt_cm)
        rewards["termination"] = self._reward_termination(done, state_info["step"])

        state_info["last_vel"] = joint_vel
        state_info["last_contact"] = contact
        state_info["feet_air_time"] *= ~contact_filt_mm

        return rewards

    # ----------- utility computations ------------
    def _compute_kick(self, step_count: jnp.int32, kick_noise: jax.Array) -> jax.Array:
        kick_interval = self._kick_interval
        kick_theta = jax.random.uniform(kick_noise, maxval=2 * jnp.pi)
        kick = jnp.array([jnp.cos(kick_theta), jnp.sin(kick_theta)])
        kick *= jnp.mod(step_count, kick_interval) == 0
        return kick

    def _kick_robot(self, state: State, kick: jax.Array) -> State:
        qvel = state.pipeline_state.qvel
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        return state.tree_replace({"pipeline_state.qvel": qvel})

    # ------------ reward functions ---------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jnp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jnp.sum(jnp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jnp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jnp.sum(jnp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        # Penalize changes in actions
        return jnp.sum(jnp.square(act - last_act))

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        rew_air_time = jnp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= math.normalize(commands[:2])[1] > 0.05  # no reward for zero command
        return rew_air_time

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jnp.sum(jnp.abs(joint_angles - self._default_pose)) * (
            math.normalize(commands[:2])[1] < 0.1
        )

    def _reward_foot_slip(
        self, pipeline_state: PipelineState, contact_filt: jax.Array
    ) -> jax.Array:
        # get velocities at feet which are offset from lower legs
        pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
        feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
        offset = Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jnp.sum(jnp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < self._resampling_time)


environment_config_classes[_ENVIRONMENT_CLASS] = EnhancedEnvironmentConfig
