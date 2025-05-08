from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax.base import State as PipelineState
from brax.base import System
from etils.epath import PathLike

from quadruped_mjx_rl.environments.base import configs_to_env_classes
from quadruped_mjx_rl.environments.base import environment_config_classes
from quadruped_mjx_rl.environments.ppo_teacher_student import QuadrupedJoystickTeacherStudentEnv
from quadruped_mjx_rl.environments.ppo_teacher_student import TeacherStudentEnvironmentConfig
from quadruped_mjx_rl.robotic_vision import VisionConfig
from quadruped_mjx_rl.robots import RobotConfig

_ENVIRONMENT_CLASS = "QuadrupedVision"


def adjust_brightness(img, scale):
    """Adjusts the brightness of an image by scaling the pixel values."""
    return jnp.clip(img * scale, 0, 1)


@dataclass
class QuadrupedVisionEnvConfig(TeacherStudentEnvironmentConfig):
    environment_class = _ENVIRONMENT_CLASS
    use_vision: bool = True

    @dataclass
    class ObservationNoiseConfig(TeacherStudentEnvironmentConfig.ObservationNoiseConfig):
        general_noise: float = 0.05
        brightness: tuple[float, float] = field(default_factory=lambda: [1.0, 1.0])

    obs_noise: ObservationNoiseConfig = field(default_factory=ObservationNoiseConfig)

    @dataclass
    class ControlConfig(TeacherStudentEnvironmentConfig.ControlConfig):
        action_scale: float = 0.3

    control: ControlConfig = field(default_factory=ControlConfig)

    @dataclass
    class CommandConfig(TeacherStudentEnvironmentConfig.CommandConfig):
        episode_length: int = 500

    command: CommandConfig = field(default_factory=CommandConfig)

    @dataclass
    class DomainRandConfig(TeacherStudentEnvironmentConfig.DomainRandConfig):
        kick_vel: float = 0.05
        kick_interval: int = 10

    domain_rand: DomainRandConfig = field(default_factory=DomainRandConfig)

    @dataclass
    class SimConfig:
        ctrl_dt: float = 0.02
        sim_dt: float = 0.004
        Kp: float = 35.0
        Kd: float = 0.5

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


environment_config_classes[_ENVIRONMENT_CLASS] = QuadrupedVisionEnvConfig


class QuadrupedVisionEnvironment(QuadrupedJoystickTeacherStudentEnv):

    def __init__(
        self,
        environment_config: QuadrupedVisionEnvConfig,
        robot_config: RobotConfig,
        init_scene_path: PathLike,
        vision_config: VisionConfig,
    ):
        super().__init__(environment_config, robot_config, init_scene_path)

        self._use_vision = environment_config.use_vision
        if self._use_vision:
            from madrona_mjx.renderer import BatchRenderer

            self.renderer = BatchRenderer(
                m=self.sys,
                gpu_id=vision_config.gpu_id,
                num_worlds=vision_config.render_batch_size,
                batch_render_view_width=vision_config.render_width,
                batch_render_view_height=vision_config.render_height,
                enabled_geom_groups=np.asarray(vision_config.enabled_geom_groups),
                enabled_cameras=np.asarray([0]),  # TODO: check cameras
                add_cam_debug_geo=False,
                use_rasterizer=vision_config.use_rasterizer,
                viz_gpu_hdls=None,
            )

    def _override_menagerie_params(
        self, sys: System, environment_config: QuadrupedVisionEnvConfig
    ) -> System:
        sys = super()._override_menagerie_params(sys, environment_config)

        # TODO: verify that this works
        floor_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        sys = sys.replace(geom_size=sys.geom_size.at[floor_id, :2].set([5.0, 5.0]))

        return sys

    def _init_obs(
        self, pipeline_state: PipelineState, state_info: dict[str, ...]
    ) -> dict[str, jax.Array]:
        if not self._use_vision:
            obs = super()._init_obs(pipeline_state, state_info)
        else:
            rng = state_info["rng"]
            rng_brightness, rng = jax.random.split(rng)
            state_info["rng"] = rng
            brightness = jax.random.uniform(
                rng_brightness,
                (1,),
                minval=self._obs_noise_config.brightness[0],
                maxval=self._obs_noise_config.brightness[1],
            )
            state_info["brightness"] = brightness

            render_token, rgb, _ = self.renderer.init(pipeline_state, self.sys)
            state_info["render_token"] = render_token

            obs_0 = jnp.asarray(rgb[0][..., :3], dtype=jnp.float32) / 255.0
            obs_0 = adjust_brightness(obs_0, brightness)

            # obs_1 = jnp.asarray(rgb[1][..., :3], dtype=jnp.float32) / 255.0
            # obs_1 = adjust_brightness(obs_1, brightness)

            obs = {"pixels/view_0": obs_0}

        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        last_obs: jax.Array | dict[str, jax.Array],
    ) -> dict[str, jax.Array]:
        if not self._use_vision:
            obs = super()._get_obs(pipeline_state, state_info, last_obs)
        else:
            _, rgb, _ = self.renderer.render(state_info["render_token"], pipeline_state)
            obs = jnp.asarray(rgb[0][..., :3], dtype=jnp.float32) / 255.0
            obs = adjust_brightness(obs, state_info["brightness"])
            obs = {"pixels/view_0": obs}
        return obs


configs_to_env_classes[QuadrupedVisionEnvConfig] = QuadrupedVisionEnvironment
