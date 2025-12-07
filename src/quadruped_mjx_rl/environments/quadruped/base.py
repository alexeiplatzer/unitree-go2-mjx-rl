from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from quadruped_mjx_rl import math
from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.environments.base import PipelineEnv
from quadruped_mjx_rl.environments.physics_pipeline import (
    Motion,
    Transform,
    EnvModel,
    EnvSpec,
    PipelineState,
    State,
)
from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.environments.vision import VisionEnvConfig


@dataclass
class EnvironmentConfig(Configuration):
    @dataclass
    class ObservationConfig:
        general_noise: float | None = 0.05
        clip: float | None = 100.0
        history_length: int | None = 15  # keep track of the last 15 steps

    observation_noise: ObservationConfig = field(default_factory=ObservationConfig)

    @dataclass
    class ControlConfig:
        action_scale: float | list[float] = 0.3

    control: ControlConfig = field(default_factory=ControlConfig)

    @dataclass
    class CommandConfig:
        pass

    command: CommandConfig = field(default_factory=CommandConfig)

    @dataclass
    class DomainRandConfig:
        apply_kicks: bool = True
        kick_vel: float = 0.05
        kick_interval: int = 10

    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)

    @dataclass
    class SimConfig:
        ctrl_dt: float = 0.02
        sim_dt: float = 0.004

        @dataclass
        class OverrideConfig:
            Kp: float = 35.0
            Kd: float = 0.5

        override: OverrideConfig = field(default_factory=OverrideConfig)

    sim: SimConfig = field(default_factory=SimConfig)

    @dataclass
    class RewardConfig:
        termination_body_height: float = 0.18
        reward_clip_min: float = -100.0
        reward_clip_max: float = 10_000.0

        # The coefficients for all reward terms used for training. All
        # physical quantities are in SI units, if not otherwise specified,
        # i.e., joint positions are in rad, positions are measured in meters,
        # torques in Nm, and time in seconds, and forces in Newtons.
        @dataclass
        class ScalesConfig:
            """
            The coefficients for all reward terms used for training. All physical quantities
            are in SI units, if not otherwise specified, i.e., joint positions are in rad,
            positions are measured in meters, torques in Nm, and time in seconds,
            and forces in Newtons.
            """

            # Regularization terms:
            lin_vel_z: float = -2.0  # Penalize base velocity in the z direction (L2 penalty).
            ang_vel_xy: float = -0.05  # Penalize base roll and pitch rate (L2 penalty).
            orientation: float = -5.0  # Penalize non-zero roll and pitch angles (L2 penalty).
            torques: float = -0.0002  # L2 regularization of joint torques, |tau|^2.
            action_rate: float = -0.01  # Penalize changes in actions; encourage smooth actions.
            feet_air_time: float = 0.2  # Encourage long swing steps (not high clearances).
            termination: float = -1.0  # Early termination penalty.
            foot_slip: float = -0.1  # Penalize foot slipping on the ground.

        scales: ScalesConfig = field(default_factory=ScalesConfig)

    rewards: RewardConfig = field(default_factory=RewardConfig)

    vision_env_config: VisionEnvConfig | None = field(default_factory=VisionEnvConfig)

    @classmethod
    def config_base_class_key(cls) -> str:
        return "environment"

    @classmethod
    def config_class_key(cls) -> str:
        return "QuadrupedBase"

    @classmethod
    def get_environment_class(cls) -> type["QuadrupedBaseEnv"]:
        return QuadrupedBaseEnv

    @classmethod
    def _get_config_class_dict(cls) -> dict[str, type["Configuration"]]:
        return _environment_config_classes


register_config_base_class(EnvironmentConfig)

_environment_config_classes = {}

register_environment_config_class = EnvironmentConfig.make_register_config_class()

register_environment_config_class(EnvironmentConfig)


class QuadrupedBaseEnv(PipelineEnv):
    """Base class for all quadruped locomotion environments. Defines common methods and
    attributes needed for all quadruped environments."""

    def __init__(
        self,
        environment_config: EnvironmentConfig,
        robot_config: RobotConfig,
        env_model: EnvSpec | EnvModel,
    ):
        super().__init__(
            env_spec=env_model,
            sim_dt=environment_config.sim.sim_dt,
            ctrl_dt=environment_config.sim.ctrl_dt,
        )

        self._rewards_config = environment_config.rewards
        self.reward_scales = asdict(environment_config.rewards.scales)

        self._termination_body_height = environment_config.rewards.termination_body_height

        self._obs_config = environment_config.observation_noise

        self._action_scale = environment_config.control.action_scale

        self._apply_kicks = environment_config.domain_rand.apply_kicks
        self._kick_interval = environment_config.domain_rand.kick_interval
        self._kick_vel = environment_config.domain_rand.kick_vel

        initial_keyframe_name = robot_config.initial_keyframe
        initial_keyframe = self._env_model.keyframe(initial_keyframe_name)
        self._init_q = jnp.array(initial_keyframe.qpos)
        self._default_pose = initial_keyframe.qpos[7:19]

        # joint ranges
        self.joints_lower_limits = jnp.array(robot_config.joints_lower_limits * 4)
        self.joints_upper_limits = jnp.array(robot_config.joints_upper_limits * 4)

        # find body definition
        self._torso_idx = mujoco.mj_name2id(
            self._env_model, mujoco.mjtObj.mjOBJ_BODY.value, name=robot_config.main_body_name
        )

        # find lower leg definition
        lower_leg_body = [
            robot_config.lower_leg_bodies.front_left,
            robot_config.lower_leg_bodies.rear_left,
            robot_config.lower_leg_bodies.front_right,
            robot_config.lower_leg_bodies.rear_right,
        ]
        lower_leg_body_id = [
            mujoco.mj_name2id(self._env_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
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
            mujoco.mj_name2id(self._env_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        if any(id_ == -1 for id_ in feet_site_id):
            raise Exception("Site not found.")
        self._feet_site_id = np.array(feet_site_id)

        self._foot_radius = robot_config.foot_radius

        # numbers of DOFs for velocity and position
        self._nv = self._pipeline_model.model.nv
        self._nq = self._pipeline_model.model.nq

    @staticmethod
    def customize_model(model: EnvModel, environment_config: EnvironmentConfig) -> EnvModel:
        model.actuator_gainprm[:, 0] = environment_config.sim.override.Kp
        model.actuator_biasprm[:, 1] = -environment_config.sim.override.Kp
        model.dof_damping[6:] = environment_config.sim.override.Kd
        return model

    def reset(self, rng: jax.Array) -> State:
        rng, init_qpos_rng = jax.random.split(rng, 2)
        init_qpos = self._set_init_qpos(init_qpos_rng)
        pipeline_state = self.pipeline_init(init_qpos, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "step": 0,
            "rewards": {k: jnp.zeros(()) for k in self.reward_scales.keys()},
            "last_act": jnp.zeros(self.action_size),
            "last_vel": jnp.zeros(12),
            "last_contact": jnp.zeros(shape=4, dtype=bool),
            "feet_air_time": jnp.zeros(4),
            "kick": jnp.array([0.0, 0.0]),
        }

        obs = self._init_obs(pipeline_state, state_info)

        reward, done = jnp.zeros(2)

        metrics = {f"reward/{k}": jnp.zeros(()) for k in self.reward_scales.keys()}
        metrics["total_dist"] = jnp.zeros(())

        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:

        if self._apply_kicks:
            # give the robot a random kick for robustness
            state.info["rng"], kick_noise = jax.random.split(state.info["rng"], 2)
            kick = self._compute_kick(step_count=state.info["step"], kick_noise=kick_noise)
            state = self._kick_robot(state, kick)
            state.info["kick"] = kick

        pipeline_state = self._physics_step(state, action)

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)

        done = self._check_termination(pipeline_state)

        # reward
        rewards = self._get_rewards(pipeline_state, state.info, action, done)
        rewards = {k: v * self.reward_scales[k] for k, v in rewards.items()}
        reward = jnp.clip(
            sum(rewards.values()) * self.dt,
            self._rewards_config.reward_clip_min,
            self._rewards_config.reward_clip_max,
        )

        # state management
        state.info["step"] += 1
        state.info["rewards"] = rewards
        state.info["last_act"] = action

        # reset the step counter when done
        state.info["step"] = jnp.where(done, 0, state.info["step"])

        state.metrics.update({f"reward/{k}": v for k, v in rewards.items()})

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(
            state.pipeline_state.x.pos[self._torso_idx - 1]
        )[1]

        done = jnp.float32(done)
        state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
        return state

    # ------------ utility computations ------------
    def _set_init_qpos(self, rng: jax.Array) -> jax.Array:
        """Sets the initial position and orientation of all the movable bodies in the
        environment in generalized coordinates."""
        return self._init_q

    def _physics_step(self, state: State, action: jax.Array) -> PipelineState:
        """Performs all the physical steps in the simulation that happen between the steps of
        the RL environment. Applies the action to the robot's actuators for the duration."""
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = jnp.clip(
            motor_targets, self.joints_lower_limits, self.joints_upper_limits
        )
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        return pipeline_state

    def _compute_kick(self, step_count: jnp.int32, kick_noise: jax.Array) -> jax.Array:
        """Computes the vector representing the kick."""
        kick_interval = self._kick_interval
        kick_theta = jax.random.uniform(kick_noise, maxval=2 * jnp.pi)
        kick = jnp.array([jnp.cos(kick_theta), jnp.sin(kick_theta)])
        kick *= jnp.mod(step_count, kick_interval) == 0
        return kick

    def _kick_robot(self, state: State, kick: jax.Array) -> State:
        """Applies a random external kick to the robot's base. Helpful to make the robot's
        balancing more robust."""
        qvel = state.pipeline_state.data.qvel
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        return state.tree_replace({"pipeline_state.data.qvel": qvel})

    # ------------ observations ------------
    def _init_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
    ) -> jax.Array | dict[str, jax.Array]:
        return self._init_proprioceptive_obs(pipeline_state, state_info)

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
        previous_obs: jax.Array | dict[str, jax.Array],
    ) -> jax.Array | dict[str, jax.Array]:
        return self._get_proprioceptive_obs(pipeline_state, state_info, previous_obs)

    def _init_proprioceptive_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
    ) -> jax.Array | dict[str, jax.Array]:
        obs = self._get_proprioceptive_obs_vector(pipeline_state, state_info)
        if self._obs_config.history_length is not None:
            obs_history = jnp.zeros(obs.size * self._obs_config.history_length)
            obs = self._update_obs_history(obs_history=obs_history, current_obs=obs)
        return obs

    def _get_proprioceptive_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
        previous_obs: jax.Array | dict[str, jax.Array],
    ) -> jax.Array | dict[str, jax.Array]:
        obs = self._get_proprioceptive_obs_vector(pipeline_state, state_info)
        if self._obs_config.history_length is not None:
            assert isinstance(previous_obs, jax.Array)
            obs = self._update_obs_history(obs_history=previous_obs, current_obs=obs)
        return obs

    def _update_obs_history(self, obs_history: jax.Array, current_obs: jax.Array) -> jax.Array:
        """Updates the observation history vector."""
        # stack observations through time
        return jnp.roll(obs_history, current_obs.size).at[: current_obs.size].set(current_obs)

    def _get_proprioceptive_obs_vector(
        self, pipeline_state: PipelineState, state_info: dict[str, Any]
    ) -> jax.Array:
        """Compounds all proprioceptive observations into a single vector. Clips and adds noise
        to values if requested."""
        obs_list = self._get_proprioceptive_obs_list(pipeline_state, state_info)
        obs = jnp.concatenate(obs_list)

        # clip, noise
        if self._obs_config.clip is not None:
            obs = jnp.clip(obs, -self._obs_config.clip, self._obs_config.clip)
        if self._obs_config.general_noise is not None:
            obs_noise = self._obs_config.general_noise * jax.random.uniform(
                state_info["rng"], obs.shape, minval=-1, maxval=1
            )
            obs = obs + obs_noise

        return obs

    def _get_proprioceptive_obs_list(
        self, pipeline_state: PipelineState, state_info: dict[str, Any]
    ) -> list[jax.Array]:
        """Computes a list of proprioceptive observations. Override this to extend or define
        custom proprioceptive observations."""

        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        obs_list = [
            jnp.array([local_rpyrate[2]]) * 0.25,  # yaw rate
            math.rotate(jnp.array([0, 0, -1]), inv_torso_rot),  # projected gravity
            pipeline_state.q[7:19] - self._default_pose,  # motor angles
            state_info["last_act"],  # last action
        ]
        return obs_list

    def _check_termination(self, pipeline_state: PipelineState) -> jax.Array:
        """Checks if the current state is defined as a terminal state. This is when:
        1) The robot flipped over
        2) The robot falls too low
        3) The joint limits are exceeded"""
        # done if joint limits are reached or robot is falling

        up = jnp.array([0.0, 0.0, 1.0])
        joint_angles = pipeline_state.q[7:19]

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
        state_info: dict[str, Any],
        action: jax.Array,
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        """Computes all the rewards for the current state of the environment and returns a dict
        of them. Updates any relevant state information. Override this to define custom rewards
        for your task."""
        x, xd = pipeline_state.x, pipeline_state.xd
        joint_vel = pipeline_state.qd[6:18]

        # foot contact data based on z-position
        foot_pos = pipeline_state.data.site_xpos[self._feet_site_id]
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state_info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state_info["last_contact"]
        first_contact = (state_info["feet_air_time"] > 0) * contact_filt_mm
        state_info["feet_air_time"] += self.dt

        rewards = {
            "lin_vel_z": self._reward_lin_vel_z(xd),
            "ang_vel_xy": self._reward_ang_vel_xy(xd),
            "orientation": self._reward_orientation(x),
            "torques": self._reward_torques(pipeline_state.data.qfrc_actuator),
            "action_rate": self._reward_action_rate(action, state_info["last_act"]),
            "feet_air_time": self._reward_feet_air_time(
                state_info["feet_air_time"], first_contact
            ),
            "foot_slip": self._reward_foot_slip(pipeline_state, contact_filt_cm),
            "termination": self._reward_termination(done, state_info["step"]),
        }

        state_info["last_vel"] = joint_vel
        state_info["last_contact"] = contact
        state_info["feet_air_time"] *= ~contact_filt_mm

        return rewards

    # ------------ reward functions ------------
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

    def _reward_feet_air_time(self, air_time: jax.Array, first_contact: jax.Array) -> jax.Array:
        # Reward air time.
        rew_air_time = jnp.sum((air_time - 0.1) * first_contact)
        return rew_air_time

    def _reward_foot_slip(
        self, pipeline_state: PipelineState, contact_filt: jax.Array
    ) -> jax.Array:
        # get velocities at feet which are offset from lower legs
        pos = pipeline_state.data.site_xpos[self._feet_site_id]  # feet position
        feet_offset = pos - pipeline_state.data.xpos[self._lower_leg_body_id]
        offset = Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jnp.sum(jnp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done

    def render(
        self,
        trajectory: list[PipelineState],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)
