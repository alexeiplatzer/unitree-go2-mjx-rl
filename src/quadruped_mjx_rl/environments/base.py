"""Base environment for training quadruped joystick policies in MJX."""

# Supporting
from dataclasses import asdict
from typing import List, Sequence

# Math
import jax
import jax.numpy as jnp

# Sim
import mujoco
import numpy as np

# Brax
from brax import base
from brax import math
from brax.base import Motion, Transform
from brax.base import State as PipelineState
from brax.base import System
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils.epath import PathLike

from quadruped_mjx_rl.configs import EnvironmentConfig
from quadruped_mjx_rl.configs import RobotConfig

_ENVIRONMENT_CLASS = "Base"


class QuadrupedJoystickBaseEnv(PipelineEnv):
    """base environment for training the go2 quadruped joystick policy in MJX."""

    def __init__(
        self,
        environment_config: EnvironmentConfig,
        robot_config: RobotConfig,
        init_scene_path: PathLike,
    ):
        self._dt = environment_config.sim.ctrl_dt
        n_frames = int(environment_config.sim.ctrl_dt / environment_config.sim.sim_dt)
        sys = self.make_system(init_scene_path, environment_config)
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self.rewards = environment_config.rewards
        self.reward_scales = asdict(environment_config.rewards.scales)

        self._obs_noise_config = environment_config.observation_noise

        self._action_scale = environment_config.control.action_scale
        self._resampling_time = environment_config.command.resampling_time
        self._command_ranges = environment_config.command.ranges
        self._termination_body_height = environment_config.rewards.termination_body_height

        initial_keyframe_name = robot_config.initial_keyframe
        initial_keyframe = sys.mj_model.keyframe(initial_keyframe_name)
        self._init_q = jnp.array(initial_keyframe.qpos)
        self._default_pose = initial_keyframe.qpos[7:]

        # joint ranges
        self.joints_lower_limits = jnp.array(robot_config.joints_lower_limits * 4)
        self.joints_upper_limits = jnp.array(robot_config.joints_upper_limits * 4)

        # find body definition
        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, name=robot_config.torso_name
        )

        # find lower leg definition
        lower_leg_body = [
            robot_config.lower_leg_bodies.front_left,
            robot_config.lower_leg_bodies.rear_left,
            robot_config.lower_leg_bodies.front_right,
            robot_config.lower_leg_bodies.rear_right,
        ]
        lower_leg_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body
        ]
        if any(id_ == -1 for id_ in lower_leg_body_id):
            raise Exception("Body not found.")
        self._lower_leg_body_id = np.array(lower_leg_body_id)

        # find feet definition
        feet_site = [
            robot_config.feet_sites.front_left,
            robot_config.feet_sites.rear_left,
            robot_config.feet_sites.front_right,
            robot_config.feet_sites.rear_right,
        ]
        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        if any(id_ == -1 for id_ in feet_site_id):
            raise Exception("Site not found.")
        self._feet_site_id = np.array(feet_site_id)

        self._foot_radius = robot_config.foot_radius

        # numbers of DOFs for velocity and position
        self._nv = sys.nv
        self._nq = sys.nq

    def make_system(
        self, init_scene_path: PathLike, environment_config: EnvironmentConfig
    ) -> System:
        sys = mjcf.load(init_scene_path)
        sys = sys.tree_replace({"opt.timestep": environment_config.sim.sim_dt})

        return self._override_menagerie_params(sys, environment_config)

    def _override_menagerie_params(
        self, sys: System, environment_config: EnvironmentConfig
    ) -> System:
        """
        Here any changes to the parameters predefined in the robot's definition can be made.
        """
        return sys

    def sample_command(self, rng: jax.Array) -> jax.Array:
        key1, key2, key3 = jax.random.split(rng, 3)
        lin_vel_x = jax.random.uniform(
            key=key1,
            shape=(1,),
            minval=self._command_ranges.lin_vel_x[0],
            maxval=self._command_ranges.lin_vel_x[1],
        )
        lin_vel_y = jax.random.uniform(
            key=key2,
            shape=(1,),
            minval=self._command_ranges.lin_vel_y[0],
            maxval=self._command_ranges.lin_vel_y[1],
        )
        ang_vel_yaw = jax.random.uniform(
            key=key3,
            shape=(1,),
            minval=self._command_ranges.ang_vel_yaw[0],
            maxval=self._command_ranges.ang_vel_yaw[1],
        )
        new_command = jnp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_command

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "step": 0,
            "rewards": {k: jnp.zeros(()) for k in self.reward_scales.keys()},
            "last_act": jnp.zeros(12),
            "command": self.sample_command(key),
        }

        obs_history = jnp.zeros(15 * 31)  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)

        reward, done = jnp.zeros(2)

        metrics = {"total_dist": jnp.zeros(())}
        for k in self.reward_scales.keys():
            metrics[f"reward/{k}"] = jnp.zeros(())

        return State(
            pipeline_state, obs, reward, done, metrics, state_info
        )

    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        pipeline_state = self._physics_step(state, action)

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)

        done = self._check_termination(pipeline_state)

        # reward
        rewards = self._get_rewards(pipeline_state, state.info, action, done)
        rewards = {k: v * self.reward_scales[k] for k, v in rewards.items()}
        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10_000.0)

        # state management
        state.info["rng"] = rng
        state.info["step"] += 1
        state.info["rewards"] = rewards
        state.info["last_act"] = action

        # sample new command if more than max timesteps achieved
        state.info["command"] = jnp.where(
            state.info["step"] > self._resampling_time,
            self.sample_command(cmd_rng),
            state.info["command"],
        )

        # reset the step counter when done
        state.info["step"] = jnp.where(
            done | (state.info["step"] > self._resampling_time), 0, state.info["step"]
        )

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(x.pos[self._torso_idx - 1])[1]

        state.metrics.update({f"reward/{k}": v for k, v in rewards.items()})

        done = jnp.float32(done)
        state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
        return state

    def _physics_step(self, state: State, action: jax.Array) -> PipelineState:
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = jnp.clip(
            motor_targets, self.joints_lower_limits, self.joints_upper_limits
        )
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        return pipeline_state

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        obs: jax.Array | dict[str, jax.Array],
    ) -> jax.Array | dict[str, jax.Array]:
        return obs

    def _check_termination(self, pipeline_state: PipelineState) -> jax.Array:
        # done if joint limits are reached or robot is falling
        up = jnp.array([0.0, 0.0, 1.0])
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]
        # flipped over
        done = jnp.dot(math.rotate(up, pipeline_state.x.rot[self._torso_idx - 1]), up) < 0
        # joint limits exceeded
        done |= jnp.any(joint_angles < self.joints_lower_limits)
        done |= jnp.any(joint_angles > self.joints_upper_limits)
        # dropped too low
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < self._termination_body_height
        return done

    def _get_rewards(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        action: jax.Array,
        done: jax.Array,
    ):
        x, xd = pipeline_state.x, pipeline_state.xd
        return {
            "tracking_lin_vel": (self._reward_tracking_lin_vel(state_info["command"], x, xd)),
            "tracking_ang_vel": (self._reward_tracking_ang_vel(state_info["command"], x, xd)),
        }

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jnp.sum(jnp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jnp.exp(-lin_vel_error / self.rewards.tracking_sigma)
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jnp.square(commands[2] - base_ang_vel[2])
        return jnp.exp(-ang_vel_error / self.rewards.tracking_sigma)

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)
