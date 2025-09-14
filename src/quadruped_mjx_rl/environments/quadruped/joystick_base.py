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
    class RewardConfig(EnvCfg.RewardConfig):
        tracking_sigma: float = 0.25  # Used in tracking reward: exp(-error^2/sigma).

        @dataclass
        class ScalesConfig(EnvCfg.RewardConfig.ScalesConfig):
            # Tracking rewards are computed using exp(-delta^2/sigma)
            # sigma can be a hyperparameter to tune.

            # Track the base x-y velocity (no z-velocity tracking).
            tracking_lin_vel: float = 1.5

            # Track the angular velocity along the z-axis (yaw rate).
            tracking_ang_vel: float = 0.8

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
        env_model: EnvModel | EnvSpec,
    ):
        super().__init__(environment_config, robot_config, env_model)

        self._resampling_time = environment_config.command.resampling_time
        self._command_ranges = environment_config.command.ranges

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

    def step(self, state: State, action: jax.Array) -> State:
        state = super().step(state, action)

        # sample new command if more than max timesteps achieved
        state.info["rng"], command_key = jax.random.split(state.info["rng"], 2)
        state.info["command"] = jnp.where(
            state.info["step"] > self._resampling_time,
            self.sample_command(command_key),
            state.info["command"],
        )

        # reset the step counter when resampling arrives
        state.info["step"] = jnp.where(
            state.info["step"] > self._resampling_time, 0, state.info["step"]
        )

        return state

    def _init_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
    ) -> jax.Array | dict[str, jax.Array]:
        """Resamples the command in addition to initializing observation arrays."""
        state_info["rng"], command_key = jax.random.split(state_info["rng"])
        state_info["command"] = self.sample_command(command_key)
        return QuadrupedBaseEnv._init_obs(self, pipeline_state, state_info)

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
        rewards = QuadrupedBaseEnv._get_rewards(self, pipeline_state, state_info, action, done)

        x, xd = pipeline_state.x, pipeline_state.xd
        rewards["tracking_lin_vel"] = self._reward_tracking_lin_vel(
            state_info["command"], x, xd
        )
        rewards["tracking_ang_vel"] = self._reward_tracking_ang_vel(
            state_info["command"], x, xd
        )

        return rewards

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

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < self._resampling_time)
