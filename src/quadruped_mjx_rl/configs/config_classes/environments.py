from typing import Generic, TypeVar

from dataclasses import dataclass, field

from brax.envs import PipelineEnv

EnvType = TypeVar("EnvType", bound=PipelineEnv)


@dataclass
class EnvironmentConfig(Generic[EnvType]):

    @dataclass
    class ObservationNoiseConfig:
        general_noise: float = 0.05

    observation_noise: ObservationNoiseConfig = field(default_factory=ObservationNoiseConfig)

    @dataclass
    class ControlConfig:
        action_scale: float = 0.3

    control: ControlConfig = field(default_factory=ControlConfig)

    @dataclass
    class CommandConfig:
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
    class DomainRandConfig:
        pass

    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)

    @dataclass
    class SimConfig:
        ctrl_dt: float = 0.02
        sim_dt: float = 0.004

        @dataclass
        class OverrideConfig:
            pass

        override: OverrideConfig = field(default_factory=OverrideConfig)

    sim: SimConfig = field(default_factory=SimConfig)

    @dataclass
    class RewardConfig:
        tracking_sigma: float = 0.25  # Used in tracking reward: exp(-error^2/sigma).
        termination_body_height: float = 0.18

        # The coefficients for all reward terms used for training. All
        # physical quantities are in SI units, if no otherwise specified,
        # i.e. joint positions are in rad, positions are measured in meters,
        # torques in Nm, and time in seconds, and forces in Newtons.
        @dataclass
        class ScalesConfig:
            # Tracking rewards are computed using exp(-delta^2/sigma)
            # sigma can be a hyperparameter to tune.

            # Track the base x-y velocity (no z-velocity tracking).
            tracking_lin_vel: float = 1.5
            # Track the angular velocity along the z-axis (yaw rate).
            tracking_ang_vel: float = 0.8

        scales: ScalesConfig = field(default_factory=ScalesConfig)

    rewards: RewardConfig = field(default_factory=RewardConfig)

    environment_class: str = "default"


environment_config_classes = {
    "default": EnvironmentConfig,
}
