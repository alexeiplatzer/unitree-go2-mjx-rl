from ml_collections import ConfigDict
from etils.epath import Path
from collections.abc import Sequence
from typing import Any

# Math
import jax
import jax.numpy as jnp
import numpy as np

# Sim
import mujoco

# Brax
from brax import envs
from brax import math
from brax.base import System, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf


class Go2TeacherEnv(PipelineEnv):

    def __init__(
        self,
        environment_config: ConfigDict,
        robot_config: ConfigDict,
        init_scene_path: Path,
    ):
        self._dt = environment_config.sim.dt
        n_frames = int(environment_config.sim.dt / environment_config.sim.timestep)
        sys = self.make_system(init_scene_path, environment_config)
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        # get privileged info about environment
        # self.total_mass = sys.body_mass[0]
        # self.privileged_obs = jnp.concatenate(
        #     [
        #         # sys.geom_friction,
        #         sys.geom_friction,
        #         sys.dof_frictionloss,
        #         sys.dof_damping,
        #         sys.jnt_stiffness,
        #         sys.actuator_forcerange,
        #         sys.body_mass[0],
        #     ]
        # )
        # self._privileged_obs_size = self.privileged_obs.size

        self.rewards = environment_config.rewards

        self._obs_noise_config = environment_config.noise

        self._action_scale = environment_config.control.action_scale
        self._resampling_time = environment_config.command.resampling_time
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

        # number of everything
        self._nv = sys.nv
        self._nq = sys.nq

    def make_system(self, init_scene_path: Path, environment_config: ConfigDict) -> System:
        sys = mjcf.load(init_scene_path)
        sys = sys.tree_replace({"opt.timestep": environment_config.sim.timestep})

        # override menagerie params for smoother policy
        sys = sys.replace(
            dof_damping=sys.dof_damping.at[6:].set(0.5239),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
        )

        return sys

    # def get_privileged_obs(self):
    #     return jnp.concatenate([self.privileged_obs], axis=1)

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

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_act": jnp.zeros(12),
            "last_vel": jnp.zeros(12),
            "command": self.sample_command(key),
            "last_contact": jnp.zeros(shape=4, dtype=jnp.bool),
            "feet_air_time": jnp.zeros(4),
            "rewards": {k: 0.0 for k in self.rewards.scales.keys()},
            "kick": jnp.array([0.0, 0.0]),
            "step": 0,
            # "privileged_obs": jnp.zeros(self._privileged_obs_size),
        }

        obs_history = jnp.zeros(15 * 31)  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jnp.zeros(2)
        metrics = {"total_dist": 0.0}
        for k in state_info["rewards"]:
            metrics[k] = state_info["rewards"][k]
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
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs["state_history"])

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
        rewards = {k: v * self.rewards.scales[k] for k, v in rewards.items()}
        reward = jnp.clip(sum(rewards.values()) * self.dt, min=0.0, max=10_000.0)

        # state management
        state.info["rng"] = rng
        state.info["step"] += 1
        state.info["rewards"] = rewards
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

        # state.info["privileged_obs"] = self.get_privileged_obs()

        # reset the step counter when done
        state.info["step"] = jnp.where(
            done | (state.info["step"] > self._resampling_time), 0, state.info["step"]
        )
        done = jnp.float32(done)

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(x.pos[self._torso_idx - 1])[1]

        state.metrics.update(state.info["rewards"])

        state = state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done)
        return state

    def _get_obs(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
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

        privileged_state = jnp.concatenate(
            [
                self.sys.geom_friction.reshape(-1),
                # self.sys.dof_frictionloss,
                # self.sys.dof_damping,
                # self.sys.jnt_stiffness,
                # self.sys.actuator_forcerange,
                # self.sys.body_mass[0],
            ]
        )

        obs = {
            "state": state,
            "state_history": history,
            "privileged_state": privileged_state,
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
        qvel = state.pipeline_state.qvel
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        return state.tree_replace({"pipeline_state.qvel": qvel})

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

    # ------------ reward functions----------------
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

    def render(
        self,
        trajectory: list[PipelineState],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)
