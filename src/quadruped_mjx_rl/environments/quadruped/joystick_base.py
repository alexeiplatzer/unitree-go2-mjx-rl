"""Base environment for training quadruped joystick policies in MJX."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from quadruped_mjx_rl import math
from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    EnvSpec,
    Motion,
    PipelineState,
    State,
    Transform,
)
from quadruped_mjx_rl.environments.quadruped.base import (
    EnvironmentConfig as EnvCfg,
    QuadrupedBaseEnv,
    register_environment_config_class,
)
from quadruped_mjx_rl.robots import RobotConfig


@dataclass
class JoystickBaseEnvConfig(EnvCfg):

    @dataclass
    class ObservationNoiseConfig(EnvCfg.ObservationNoiseConfig):
        general_noise: float = 0.05

    observation_noise: ObservationNoiseConfig = field(default_factory=ObservationNoiseConfig)

    @dataclass
    class ControlConfig(EnvCfg.ControlConfig):
        action_scale: float | list[float] = 0.3

    control: ControlConfig = field(default_factory=ControlConfig)

    @dataclass
    class CommandConfig(EnvCfg.CommandConfig):
        resampling_time: int = 500

        @dataclass
        class RangesConfig:
            # min max [m/s]
            lin_vel_x_min: float = -0.6
            lin_vel_x_max: float = 1.5
            lin_vel_y_min: float = -0.8
            lin_vel_y_max: float = 0.8
            # min max [rad/s]
            ang_vel_yaw_min: float = -0.7
            ang_vel_yaw_max: float = 0.7

        ranges: RangesConfig = field(default_factory=RangesConfig)

    command: CommandConfig = field(default_factory=CommandConfig)

    @dataclass
    class DomainRandConfig(EnvCfg.DomainRandConfig):
        kick_vel: float = 0.05
        kick_interval: int = 10

    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)

    @dataclass
    class SimConfig(EnvCfg.SimConfig):
        ctrl_dt: float = 0.02
        sim_dt: float = 0.004

        @dataclass
        class OverrideConfig(EnvCfg.SimConfig.OverrideConfig):
            Kp: float = 35.0
            Kd: float = 0.5

        override: OverrideConfig = field(default_factory=OverrideConfig)

    sim: SimConfig = field(default_factory=SimConfig)

    @dataclass
    class RewardConfig(EnvCfg.RewardConfig):
        tracking_sigma: float = 0.25  # Used in tracking reward: exp(-error^2/sigma).
        termination_body_height: float = 0.18

        # The coefficients for all reward terms used for training. All
        # physical quantities are in SI units, if not otherwise specified,
        # i.e., joint positions are in rad, positions are measured in meters,
        # torques in Nm, and time in seconds, and forces in Newtons.
        @dataclass
        class ScalesConfig(EnvCfg.RewardConfig.ScalesConfig):
            # Tracking rewards are computed using exp(-delta^2/sigma)
            # sigma can be a hyperparameter to tune.

            # Track the base x-y velocity (no z-velocity tracking).
            tracking_lin_vel: float = 1.5

            # Track the angular velocity along the z-axis (yaw rate).
            tracking_ang_vel: float = 0.8

            # Regularization terms:
            lin_vel_z: float = -2.0  # Penalize base velocity in the z direction (L2 penalty).
            ang_vel_xy: float = -0.05  # Penalize base roll and pitch rate (L2 penalty).
            orientation: float = -5.0  # Penalize non-zero roll and pitch angles (L2 penalty).
            torques: float = -0.0002  # L2 regularization of joint torques, |tau|^2.
            action_rate: float = -0.01  # Penalize changes in actions; encourage smooth actions.
            feet_air_time: float = 0.2  # Encourage long swing steps (not high clearances).
            stand_still: float = -0.5  # Encourage no motion at zero command (L2 penalty).
            termination: float = -1.0  # Early termination penalty.
            foot_slip: float = -0.1  # Penalize foot slipping on the ground.

        scales: ScalesConfig = field(default_factory=ScalesConfig)

    rewards: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "JoystickBase"

    @classmethod
    def get_environment_class(cls) -> type[QuadrupedBaseEnv]:
        return QuadrupedJoystickBaseEnv


register_environment_config_class(JoystickBaseEnvConfig)


class QuadrupedJoystickBaseEnv(QuadrupedBaseEnv):
    """base environment for training the go2 quadruped joystick policy in MJX."""

    def __init__(
        self,
        environment_config: JoystickBaseEnvConfig,
        robot_config: RobotConfig,
        env_spec: EnvModel | EnvSpec,
    ):
        super().__init__(environment_config, robot_config, env_spec)

        self._kick_interval = environment_config.domain_rand.kick_interval
        self._kick_vel = environment_config.domain_rand.kick_vel

        self._resampling_time = environment_config.command.resampling_time
        self._command_ranges = environment_config.command.ranges

    @staticmethod
    def customize_model(
        model: EnvModel | EnvSpec, environment_config: JoystickBaseEnvConfig
    ) -> EnvModel:
        env_model = QuadrupedBaseEnv.customize_model(model, environment_config)
        env_model.dof_damping[6:] = environment_config.sim.override.Kd
        return env_model

    def sample_command(self, rng: jax.Array) -> jax.Array:
        key1, key2, key3 = jax.random.split(rng, 3)
        lin_vel_x = jax.random.uniform(
            key=key1,
            shape=(1,),
            minval=self._command_ranges.lin_vel_x_min,
            maxval=self._command_ranges.lin_vel_x_max,
        )
        lin_vel_y = jax.random.uniform(
            key=key2,
            shape=(1,),
            minval=self._command_ranges.lin_vel_y_min,
            maxval=self._command_ranges.lin_vel_y_max,
        )
        ang_vel_yaw = jax.random.uniform(
            key=key3,
            shape=(1,),
            minval=self._command_ranges.ang_vel_yaw_min,
            maxval=self._command_ranges.ang_vel_yaw_max,
        )
        new_command = jnp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_command

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)

        state.metrics["total_dist"] = jnp.zeros(())

        return state

    def step(self, state: State, action: jax.Array) -> State:
        rng, command_key, kick_noise = jax.random.split(state.info["rng"], 3)

        # give the robot a random kick for robustness
        kick = self._compute_kick(step_count=state.info["step"], kick_noise=kick_noise)
        state = self._kick_robot(state, kick)
        state.info["rng"] = rng
        state.info["kick"] = kick

        state = super().step(state, action)

        # sample new command if more than max timesteps achieved
        state.info["command"] = jnp.where(
            state.info["step"] > self._resampling_time,
            self.sample_command(command_key),
            state.info["command"],
        )

        # reset the step counter when resampling arrives
        state.info["step"] = jnp.where(
            state.info["step"] > self._resampling_time, 0, state.info["step"]
        )

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(
            state.pipeline_state.x.pos[self._torso_idx - 1]
        )[1]

        return state

    def _init_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
    ) -> jax.Array | dict[str, jax.Array]:
        state_info["rng"], command_key = jax.random.split(state_info["rng"])

        state_info["command"] = self.sample_command(command_key)
        state_info["last_vel"] = jnp.zeros(12)
        state_info["last_contact"] = jnp.zeros(shape=4, dtype=jnp.bool)
        state_info["feet_air_time"] = jnp.zeros(4)
        state_info["kick"] = jnp.array([0.0, 0.0])

        state_obs = QuadrupedJoystickBaseEnv._get_state_obs(self, pipeline_state, state_info)
        obs_history = jnp.zeros(state_obs.size * 15)  # keep track of the last 15 steps
        obs_history = QuadrupedJoystickBaseEnv._update_obs_history(self, obs_history, state_obs)
        return obs_history

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        previous_obs: jax.Array | dict[str, jax.Array],
    ) -> jax.Array | dict[str, jax.Array]:
        assert isinstance(previous_obs, jax.Array)
        obs_history = previous_obs
        state_obs = QuadrupedJoystickBaseEnv._get_state_obs(self, pipeline_state, state_info)
        obs_history = QuadrupedJoystickBaseEnv._update_obs_history(self, obs_history, state_obs)
        return obs_history

    def _update_obs_history(self, obs_history: jax.Array, current_obs: jax.Array) -> jax.Array:
        # stack observations through time
        return jnp.roll(obs_history, current_obs.size).at[: current_obs.size].set(current_obs)

    def _get_state_obs(
        self, pipeline_state: PipelineState, state_info: dict[str, ...]
    ) -> jax.Array:
        obs_list = QuadrupedJoystickBaseEnv._get_raw_obs_list(self, pipeline_state, state_info)
        obs = jnp.concatenate(obs_list)

        # clip, noise
        obs_noise = self._obs_noise_config.general_noise * jax.random.uniform(
            state_info["rng"], obs.shape, minval=-1, maxval=1
        )
        obs = jnp.clip(obs, -100.0, 100.0) + obs_noise
        return obs

    def _get_raw_obs_list(
        self, pipeline_state: PipelineState, state_info: dict[str, ...]
    ) -> list[jax.Array]:
        obs_list = [
            state_info["command"] * jnp.array([2.0, 2.0, 0.25]),  # command
            *QuadrupedBaseEnv._get_raw_obs_list(self, pipeline_state, state_info),
        ]
        return obs_list

    def _get_rewards(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        action: jax.Array,
        done: jax.Array,
    ):
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

        rewards = {
            "tracking_lin_vel": self._reward_tracking_lin_vel(state_info["command"], x, xd),
            "tracking_ang_vel": self._reward_tracking_ang_vel(state_info["command"], x, xd),
            "lin_vel_z": self._reward_lin_vel_z(xd),
            "ang_vel_xy": self._reward_ang_vel_xy(xd),
            "orientation": self._reward_orientation(x),
            "torques": self._reward_torques(pipeline_state.qfrc_actuator),
            "action_rate": self._reward_action_rate(action, state_info["last_act"]),
            "stand_still": self._reward_stand_still(state_info["command"], joint_angles),
            "feet_air_time": self._reward_feet_air_time(
                state_info["feet_air_time"], first_contact, state_info["command"]
            ),
            "foot_slip": self._reward_foot_slip(pipeline_state, contact_filt_cm),
            "termination": self._reward_termination(done, state_info["step"]),
        }

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

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jnp.sum(jnp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jnp.exp(-lin_vel_error / self._rewards_config.tracking_sigma)
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jnp.square(commands[2] - base_ang_vel[2])
        return jnp.exp(-ang_vel_error / self._rewards_config.tracking_sigma)

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
