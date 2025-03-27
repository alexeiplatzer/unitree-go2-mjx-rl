from dataclasses import dataclass, field, asdict
from ml_collections import config_dict
from etils.epath import PathLike

import jax
import jax.numpy as jnp
import numpy as np

import mujoco
from mujoco import mjx

from brax import math
from brax.io import mjcf
from brax.base import System, Motion, Transform
from brax.mjx.pipeline import init as pipeline_init
from brax.mjx.pipeline import step as pipeline_step

from mujoco_playground import MjxEnv, State

from ..robots import RobotConfig
from .configs import EnvironmentConfig, VisionConfig


def adjust_brightness(img, scale):
    """Adjusts the brightness of an image by scaling the pixel values."""
    return jnp.clip(img * scale, 0, 1)


@dataclass
class QuadrupedVisionEnvironmentConfig(EnvironmentConfig["Go2TeacherEnv"]):
    name: str = "quadruped_vision"
    use_vision: bool = True

    @dataclass
    class ObservationNoiseConfig:
        general_noise: float = 0.05
        brightness: tuple[float, float] = field(default_factory=lambda: [1.0, 1.0])

    obs_noise: ObservationNoiseConfig = field(default_factory=ObservationNoiseConfig)

    @dataclass
    class ControlConfig:
        action_scale: float = 0.3

    control: ControlConfig = field(default_factory=ControlConfig)

    @dataclass
    class CommandConfig:
        episode_length: int = 500

    command: CommandConfig = field(default_factory=CommandConfig)

    @dataclass
    class DomainRandConfig:
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


class QuadrupedVisionEnvironment(MjxEnv):

    def __init__(
        self,
        environment_config: QuadrupedVisionEnvironmentConfig,
        vision_config: VisionConfig,
        robot_config: RobotConfig,
        init_scene_path: PathLike,
    ):
        super().__init__(config_dict.ConfigDict(asdict(environment_config.sim)))

        sys = self.make_system(init_scene_path, environment_config)

        self._xml_path = init_scene_path  # TODO: most likely not xml
        self._mjx_model = sys
        self._mj_model = sys.mj_model

        # number of everything
        self._nv = sys.nv
        self._nq = sys.nq

        self.rewards = environment_config.rewards
        self.reward_scales = asdict(environment_config.rewards.scales)

        self._obs_noise_config = environment_config.obs_noise

        self._action_scale = environment_config.control.action_scale
        self._resampling_time = environment_config.command.episode_length
        self._kick_interval = environment_config.domain_rand.kick_interval
        self._kick_vel = environment_config.domain_rand.kick_vel
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
        self._torso_mass = sys.mj_model.body_subtreemass[self._torso_idx]

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

        self._use_vision = environment_config.use_vision
        if self._use_vision:
            from madrona_mjx.renderer import BatchRenderer
            self.renderer = BatchRenderer(
                m=self._mjx_model,
                gpu_id=vision_config.gpu_id,
                num_worlds=vision_config.render_batch_size,
                batch_render_view_width=vision_config.render_width,
                batch_render_view_height=vision_config.render_height,
                enabled_geom_groups=np.asarray(vision_config.enabled_geom_groups),
                enabled_cameras=None,  # Use all cameras.  # TODO: check cameras
                add_cam_debug_geo=False,
                use_rasterizer=vision_config.use_rasterizer,
                viz_gpu_hdls=None,
            )


    def make_system(
        self, init_scene_path: PathLike, environment_config: QuadrupedVisionEnvironmentConfig
    ) -> System:
        sys = mjcf.load(init_scene_path)
        sys = sys.tree_replace({"opt.timestep": self._sim_dt})

        # override menagerie params for smoother policy
        sys = sys.replace(
            dof_damping=sys.dof_damping.at[6:].set(environment_config.sim.Kd),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(environment_config.sim.Kp),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-environment_config.sim.Kp),
        )

        # TODO: verify that this works
        floor_id = mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        sys = sys.replace(
            geom_size=sys.geom_size.at[floor_id, :2].set([5.0, 5.0])
        )

        return sys

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def sample_command(self, rng: jax.Array) -> jax.Array:
        # TODO adapt from config / sample with curriculum
        # TODO try overfitting to smaller intervals
        lin_vel_x = [-0.6, 1.5]  # min max [m/s]
        lin_vel_y = [-0.8, 0.8]  # min max [m/s]
        ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, num=4)
        lin_vel_x = jax.random.uniform(key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1])
        lin_vel_y = jax.random.uniform(key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1])
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jnp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:
        rng, key = jax.random.split(rng)

        pipeline_state = pipeline_init(self._mjx_model, self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_act": jnp.zeros(12),
            "last_vel": jnp.zeros(12),
            "command": self.sample_command(key),
            "last_contact": jnp.zeros(shape=4, dtype=jnp.bool),
            "feet_air_time": jnp.zeros(4),
            "kick": jnp.array([0.0, 0.0]),
            "step": 0,
        }

        if not self._use_vision:
            obs_history = jnp.zeros(15 * 31)  # store 15 steps of history
            obs = self._get_obs(pipeline_state, state_info, obs_history)
        else:
            rng_brightness, rng = jax.random.split(rng)
            brightness = jax.random.uniform(
                rng_brightness,
                (1,),
                minval=self._obs_noise_config.brightness[0],
                maxval=self._obs_noise_config.brightness[1],
            )
            state_info.update({'brightness': brightness})

            render_token, rgb, _ = self.renderer.init(pipeline_state, self._mjx_model)
            state_info.update({'render_token': render_token})

            obs = jnp.asarray(rgb[0][..., :3], dtype=jnp.float32) / 255.0
            obs = adjust_brightness(obs, brightness)
            obs = {'pixels/view_0': obs}

        reward, done = jnp.zeros(2)

        metrics = {"total_dist": jnp.zeros(())}  # TODO: check correct initialization
        for k in self.reward_scales.keys():
            metrics[f"reward/{k}"] = jnp.zeros(())

        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng, kick_noise = jax.random.split(state.info["rng"], num=3)

        kick = self._compute_kick(step_count=state.info["step"], kick_noise=kick_noise)
        state = self._kick_robot(state, kick)

        # physics step
        motor_targets = self._default_pose + action * self._action_scale
        motor_targets = jnp.clip(
            motor_targets, self.joints_lower_limits, self.joints_upper_limits
        )
        # TODO: not sure if the states get messed up
        pipeline_state = pipeline_step(self._mjx_model, state.data, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # observation data
        if not self._use_vision:
            obs = self._get_obs(pipeline_state, state.info, state.obs["state_history"])
        else:
            _, rgb, _ = self.renderer.render(state.info['render_token'], pipeline_state)
            obs = jnp.asarray(rgb[0][..., :3], dtype=jnp.float32) / 255.0
            obs = adjust_brightness(obs, state.info['brightness'])
            obs = {'pixels/view_0': obs}

        done = self._check_termination(pipeline_state)

        # foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[self._feet_site_id]
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt

        # reward
        rewards = {
            "tracking_lin_vel": self._reward_tracking_lin_vel(state.info["command"], x, xd),
            "tracking_ang_vel": self._reward_tracking_ang_vel(state.info["command"], x, xd),
            "lin_vel_z": self._reward_lin_vel_z(xd),
            "ang_vel_xy": self._reward_ang_vel_xy(xd),
            "orientation": self._reward_orientation(x),
            "torques": self._reward_torques(pipeline_state.qfrc_actuator),
            "action_rate": self._reward_action_rate(action, state.info["last_act"]),
            "stand_still": self._reward_stand_still(
                state.info["command"],
                joint_angles,
            ),
            "feet_air_time": self._reward_feet_air_time(
                state.info["feet_air_time"],
                first_contact,
                state.info["command"],
            ),
            "foot_slip": self._reward_foot_slip(pipeline_state, contact_filt_cm),
            "termination": self._reward_termination(done, state.info["step"]),
            # "energy": self._reward_energy(pipeline_state.qfrc_actuator, pipeline_state.qvel[6:]),
            # "energy_expenditure": self._reward_energy_expenditure(
            #     pipeline_state.qfrc_actuator, pipeline_state.qvel[6:]
            # ),
            # "joint_ang_vel": self._reward_joint_ang_vel(pipeline_state.qvel[6:]),
        }
        rewards = {k: v * self.reward_scales[k] for k, v in rewards.items()}
        reward = jnp.clip(sum(rewards.values()) * self.dt, min=0.0, max=10_000.0)

        # state management
        state.info["rng"] = rng
        state.info["step"] += 1
        state.info["kick"] = kick
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact

        # sample new command if more than 500 timesteps achieved
        state.info["command"] = jnp.where(
            state.info["step"] > self._resampling_time,
            self.sample_command(cmd_rng),
            state.info["command"],
        )

        # reset the step counter when done
        state.info["step"] = jnp.where(
            done | (state.info["step"] > self._resampling_time), 0, state.info["step"]
        )
        done = jnp.float32(done)

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(x.pos[self._torso_idx - 1])[1]

        state.metrics.update({f"reward/{k}": v for k, v in rewards.items()})

        state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
        return state

    def _get_obs(
        self,
        pipeline_state,
        state_info: dict[str, ...],
        obs_history: jax.Array,
    ) -> dict[str, jax.Array]:
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        state = jnp.concatenate(
            [
                jnp.array([local_rpyrate[2]]) * 0.25,  # yaw rate
                math.rotate(jnp.array([0, 0, -1]), inv_torso_rot),  # projected gravity
                state_info["command"] * jnp.array([2.0, 2.0, 0.25]),  # command
                pipeline_state.q[7:] - self._default_pose,  # motor angles
                state_info["last_act"],  # last action
            ]
        )

        # stack observations through time
        history = jnp.roll(obs_history, state.size).at[: state.size].set(state)

        # privileged_state = jnp.concatenate(
        #     [
        #         self.sys.geom_friction.reshape(-1),
        #         # self.sys.dof_frictionloss,
        #         # self.sys.dof_damping,
        #         # self.sys.jnt_stiffness,
        #         # self.sys.actuator_forcerange,
        #         # self.sys.body_mass[0],
        #     ]
        # )

        obs = {
            "state": state,
            "state_history": history,
            # "privileged_state": privileged_state,
        }

        return obs

    # ----------- utility computations ------------
    def _compute_kick(self, step_count: jnp.int32, kick_noise: jax.Array) -> jax.Array:
        kick_interval = self._kick_interval
        kick_theta = jax.random.uniform(kick_noise, maxval=2 * jnp.pi)
        kick = jnp.array([jnp.cos(kick_theta), jnp.sin(kick_theta)])
        kick *= jnp.mod(step_count, kick_interval) == 0
        return kick

    def _kick_robot(self, state: State, kick: jax.Array) -> State:
        qvel = state.data.qvel
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        return state.tree_replace({"data.qvel": qvel})

    def _check_termination(self, pipeline_state) -> jax.Array:
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

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd) -> jax.Array:
        # Penalize z axis base linear velocity
        return jnp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jnp.sum(jnp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x) -> jax.Array:
        # Penalize non flat base orientation
        up = jnp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jnp.sum(jnp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jnp.sqrt(jnp.sum(jnp.square(torques))) + jnp.sum(jnp.abs(torques))

    def _reward_energy(self, torques: jax.Array, joint_vels: jax.Array) -> jax.Array:
        return jnp.sum(torques * joint_vels)

    def _reward_energy_expenditure(
        self, torques: jax.Array, joint_vels: jax.Array
    ) -> jax.Array:
        return jnp.sum(jnp.clip(torques * joint_vels, 0, 1e30))

    def _reward_joint_ang_vel(self, joint_vels: jax.Array) -> jax.Array:
        return jnp.sum(jnp.square(joint_vels))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        # Penalize changes in actions
        return jnp.sum(jnp.square(act - last_act))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x, xd
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jnp.sum(jnp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jnp.exp(-lin_vel_error / self.rewards.tracking_sigma)
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x, xd
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jnp.square(commands[2] - base_ang_vel[2])
        return jnp.exp(-ang_vel_error / self.rewards.tracking_sigma)

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
        self, pipeline_state, contact_filt: jax.Array
    ) -> jax.Array:
        # get velocities at feet which are offset from lower legs
        # pytype: disable=attribute-error
        pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
        feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
        # pytype: enable=attribute-error
        offset = Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jnp.sum(jnp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < self._resampling_time)

    # def render(
    #     self,
    #     trajectory: list[PipelineState],
    #     camera: str | None = None,
    #     width: int = 240,
    #     height: int = 320,
    # ) -> Sequence[np.ndarray]:
    #     camera = camera or "track"
    #     return super().render(trajectory, camera=camera, width=width, height=height)
