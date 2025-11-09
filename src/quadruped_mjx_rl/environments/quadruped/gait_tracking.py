from dataclasses import dataclass, field
from collections.abc import Callable

import jax
from jax import numpy as jnp

from quadruped_mjx_rl.robots import RobotConfig
from quadruped_mjx_rl.environments.physics_pipeline import (
    EnvModel,
    EnvSpec,
)
from quadruped_mjx_rl.environments.quadruped.base import (
    EnvironmentConfig as EnvCfg,
    QuadrupedBaseEnv,
    register_environment_config_class,
)


def cos_wave(t, step_period, scale):
    _cos_wave = -jnp.cos(((2 * jnp.pi) / step_period) * t)
    return _cos_wave * (scale / 2) + (scale / 2)


def dcos_wave(t, step_period, scale):
    """
    Derivative of the cos wave, for reference velocity
    """
    return ((scale * jnp.pi) / step_period) * jnp.sin(((2 * jnp.pi) / step_period) * t)


def make_kinematic_ref(sinusoid, step_k, scale=0.3, dt=1 / 50):
    """
    Makes trotting kinematics for the 12 leg joints.
    `step_k` is the number of timesteps it takes to raise and lower a given foot.
    A gait cycle is `2 * step_k * dt`-seconds long.
    """

    _steps = jnp.arange(step_k)
    step_period = step_k * dt
    t = _steps * dt

    wave = sinusoid(t, step_period, scale)
    # Commands for one step of an active front leg
    fleg_cmd_block = jnp.concatenate(
        [jnp.zeros((step_k, 1)), wave.reshape(step_k, 1), -2 * wave.reshape(step_k, 1)], axis=1
    )
    # Our standing config reverses front and hind legs
    h_leg_cmd_bloc = -1 * fleg_cmd_block

    block1 = jnp.concatenate(
        [jnp.zeros((step_k, 3)), fleg_cmd_block, h_leg_cmd_bloc, jnp.zeros((step_k, 3))], axis=1
    )

    block2 = jnp.concatenate(
        [fleg_cmd_block, jnp.zeros((step_k, 3)), jnp.zeros((step_k, 3)), h_leg_cmd_bloc], axis=1
    )
    # In one step cycle, both pairs of active legs have inactive and active phases
    step_cycle = jnp.concatenate([block1, block2], axis=0)
    return step_cycle


def axis_angle_to_quaternion(v: jnp.ndarray, theta: jnp.float_):
    """
    axis angle representation: rotation of theta around v.
    """
    return jnp.concatenate(
        [jnp.cos(0.5 * theta).reshape(1), jnp.sin(0.5 * theta) * v.reshape(3)]
    )


@dataclass
class QuadrupedGaitTrackingEnvConfig(EnvCfg):

    @dataclass
    class RewardConfig(EnvCfg.RewardConfig):
        termination_body_height: float = 0.25

        @dataclass
        class ScalesConfig:
            tracking_lin_vel: float = (1.0,)
            orientation: float = (-1.0,)  # non-flat base
            height: float = (0.5,)
            lin_vel_z: float = (-1.0,)  # prevents the suicide policy
            torque: float = (-0.01,)
            feet_pos: float = (-1,)  # Bad action hard-coding.
            feet_height: float = (-1,)  # prevents it from just standing still
            joint_velocity: float = -0.001

        scales: ScalesConfig = field(default_factory=ScalesConfig)

    rewards: RewardConfig = field(default_factory=RewardConfig)


register_environment_config_class(QuadrupedGaitTrackingEnvConfig)


class QuadrupedGaitTrackingEnv(QuadrupedBaseEnv):
    def __init__(
        self,
        environment_config: QuadrupedGaitTrackingEnvConfig,
        robot_config: RobotConfig,
        env_model: EnvSpec | EnvModel,
        baseline_inference_fn: Callable,
    ):
        super().__init__(environment_config, robot_config, env_model)
