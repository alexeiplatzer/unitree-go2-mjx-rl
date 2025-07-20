# Typing
from dataclasses import dataclass, asdict
from collections.abc import Sequence

# Supporting
from etils.epath import PathLike

# Math
import jax
import jax.numpy as jnp
import numpy as np
from quadruped_mjx_rl import math

# Sim
import mujoco
from brax.base import System, State as PipelineState
from brax.envs.base import State, PipelineEnv
from brax.io.mjcf import load as load_system


# Configs
from quadruped_mjx_rl.config_utils import Configuration, register_config_base_class
from quadruped_mjx_rl.robots import RobotConfig


@dataclass
class EnvironmentConfig(Configuration):
    @dataclass
    class ObservationNoiseConfig:
        pass

    observation_noise: ObservationNoiseConfig

    @dataclass
    class ControlConfig:
        action_scale: float | list[float]

    control: ControlConfig

    @dataclass
    class CommandConfig:
        pass

    command: CommandConfig

    @dataclass
    class DomainRandConfig:
        pass

    domain_rand: DomainRandConfig

    @dataclass
    class SimConfig:
        ctrl_dt: float
        sim_dt: float

        @dataclass
        class OverrideConfig:
            Kp: float

        override: OverrideConfig

    sim: SimConfig

    @dataclass
    class RewardConfig:
        termination_body_height: float

        # The coefficients for all reward terms used for training. All
        # physical quantities are in SI units, if not otherwise specified,
        # i.e., joint positions are in rad, positions are measured in meters,
        # torques in Nm, and time in seconds, and forces in Newtons.
        @dataclass
        class ScalesConfig:
            pass

        scales: ScalesConfig

    rewards: RewardConfig

    @classmethod
    def config_base_class_key(cls) -> str:
        return "environment"

    @classmethod
    def environment_class_key(cls) -> str:
        return "QuadrupedBase"

    @classmethod
    def get_environment_class(cls) -> type["QuadrupedBaseEnv"]:
        return QuadrupedBaseEnv

    @classmethod
    def from_dict(cls, config_dict: dict) -> Configuration:
        environment_class_key = config_dict.pop("environment_class")
        environment_config_class = _environment_config_classes[environment_class_key]
        return super(EnvironmentConfig, environment_config_class).from_dict(config_dict)

    def to_dict(self) -> dict:
        config_dict = super().to_dict()
        config_dict["environment_class"] = type(self).environment_class_key()
        return config_dict


register_config_base_class(EnvironmentConfig)

_environment_config_classes = {}


def register_environment_config_class(environment_config_class: type[EnvironmentConfig]):
    _environment_config_classes[
        environment_config_class.environment_class_key()
    ] = environment_config_class


register_environment_config_class(EnvironmentConfig)


class QuadrupedBaseEnv(PipelineEnv):

    def __init__(
        self,
        environment_config: EnvironmentConfig,
        robot_config: RobotConfig,
        init_scene_path: PathLike,
    ):
        n_frames = int(environment_config.sim.ctrl_dt / environment_config.sim.sim_dt)
        sys = self.make_system(init_scene_path, environment_config)
        super().__init__(sys, n_frames=n_frames)

        self._rewards_config = environment_config.rewards
        self.reward_scales = asdict(environment_config.rewards.scales)

        self._termination_body_height = environment_config.rewards.termination_body_height

        self._obs_noise_config = environment_config.observation_noise

        self._action_scale = environment_config.control.action_scale

        initial_keyframe_name = robot_config.initial_keyframe
        initial_keyframe = sys.mj_model.keyframe(initial_keyframe_name)
        self._init_q = jnp.array(initial_keyframe.qpos)
        self._default_pose = initial_keyframe.qpos[7:]

        # joint ranges
        self.joints_lower_limits = jnp.array(robot_config.joints_lower_limits * 4)
        self.joints_upper_limits = jnp.array(robot_config.joints_upper_limits * 4)

        # find body definition
        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, name=robot_config.main_body_name
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

    @staticmethod
    def make_system(
        init_scene_path: PathLike, environment_config: EnvironmentConfig
    ) -> System:
        sys = load_system(init_scene_path)
        sys = sys.tree_replace({"opt.timestep": environment_config.sim.sim_dt})
        sys = sys.replace(
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(
                environment_config.sim.override.Kp
            ),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(
                -environment_config.sim.override.Kp
            ),
        )
        return sys

    def reset(self, rng: jax.Array) -> State:
        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "step": 0,
            "rewards": {k: jnp.zeros(()) for k in self.reward_scales.keys()},
            "last_act": jnp.zeros(self.action_size),
        }

        obs = self._init_obs(pipeline_state, state_info)

        reward, done = jnp.zeros(2)

        metrics = {
            f"reward/{k}": jnp.zeros(()) for k in self.reward_scales.keys()
        }

        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:

        pipeline_state = self._physics_step(state, action)

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)

        done = self._check_termination(pipeline_state)

        # reward
        rewards = self._get_rewards(pipeline_state, state.info, action, done)
        rewards = {k: v * self.reward_scales[k] for k, v in rewards.items()}
        reward = jnp.clip(sum(rewards.values()) * self.dt, 0.0, 10_000.0)

        # state management
        state.info["step"] += 1
        state.info["rewards"] = rewards
        state.info["last_act"] = action

        # reset the step counter when done
        state.info["step"] = jnp.where(
            done, 0, state.info["step"]
        )

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

    def _init_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
    ) -> jax.Array | dict[str, jax.Array]:
        obs = QuadrupedBaseEnv._get_state_obs(self, pipeline_state, state_info)
        return obs

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, ...],
        previous_obs: jax.Array | dict[str, jax.Array],
    ) -> jax.Array | dict[str, jax.Array]:
        obs = QuadrupedBaseEnv._get_state_obs(self, pipeline_state, state_info)
        return obs

    def _get_state_obs(
        self, pipeline_state: PipelineState, state_info: dict[str, ...]
    ) -> jax.Array:
        obs_list = QuadrupedBaseEnv._get_raw_obs_list(self, pipeline_state, state_info)
        obs = jnp.clip(jnp.concatenate(obs_list), -100.0, 100.0)
        return obs

    def _get_raw_obs_list(
        self, pipeline_state: PipelineState, state_info: dict[str, ...]
    ) -> list[jax.Array]:

        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        obs_list = [
            jnp.array([local_rpyrate[2]]) * 0.25,  # yaw rate
            math.rotate(jnp.array([0, 0, -1]), inv_torso_rot),  # projected gravity
            pipeline_state.q[7:] - self._default_pose,  # motor angles
            state_info["last_act"],  # last action
        ]
        return obs_list

    def _check_termination(self, pipeline_state: PipelineState) -> jax.Array:
        # done if joint limits are reached or robot is falling

        up = jnp.array([0.0, 0.0, 1.0])
        joint_angles = pipeline_state.q[7:]

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
    ) -> dict[str, jax.Array]:
        return {}

    def render(
        self,
        trajectory: list[PipelineState],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)
