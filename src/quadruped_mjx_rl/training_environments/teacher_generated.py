import brax
from brax import envs
import jax
import jax.numpy as jnp
import numpy as np

# Example: assume you have a robot defined in an MJCF file.
# The MJCF should define a legged robot with actuators, bodies, etc.
# We load it once at the top-level. You would replace 'path/to/robot.xml' with your MJCF.
from brax.io import mjcf

from brax.base import System
from brax.base import State as PipelineState
from brax.envs.base import PipelineEnv, State


class RewardThresholdCurriculum:
    def __init__(self, seed: int, x_vel: tuple, y_vel: tuple, yaw_vel: tuple):
        self.rng = np.random.RandomState(seed)
        self.x_range = x_vel
        self.y_range = y_vel
        self.yaw_range = yaw_vel
        self.weights = np.ones((1,))  # placeholder for some distribution logic

    def update(self, old_bins, lin_vel_rewards, ang_vel_rewards, lin_vel_thresh, ang_vel_thresh, local_range=0.5):
        # Implement curriculum logic here
        pass

    def sample(self, batch_size):
        # Sample new commands from the distribution
        x_cmd = self.rng.uniform(self.x_range[0], self.x_range[1], size=batch_size)
        y_cmd = self.rng.uniform(self.y_range[0], self.y_range[1], size=batch_size)
        yaw_cmd = self.rng.uniform(self.yaw_range[0], self.yaw_range[1], size=batch_size)
        bins = np.zeros(batch_size, dtype=int)
        return np.stack([x_cmd, y_cmd, yaw_cmd], axis=1), bins


class LeggedRobotEnv(env.Env):
    """
    Brax environment for a legged robot, adapted from an IsaacGym environment.
    Suitable for teacher-student PPO setup.
    """

    def __init__(self,
                 seed=0,
                 num_envs=1,
                 episode_length=1000,
                 dt=0.002,
                 action_scale=0.5,
                 # Curriculum and domain randomization configs:
                 curriculum_cfg=None,
                 domain_rand_cfg=None,
                 # Teacher-student configs:
                 observe_command=True,
                 observe_vel=True,
                 observe_only_lin_vel=False,
                 observe_only_ang_vel=False,
                 observe_yaw=False,
                 measure_heights=False,
                 **kwargs):
        self._num_envs = num_envs
        self._episode_length = episode_length
        self._action_scale = action_scale
        self._dt = dt
        self._current_step = 0

        self.observe_command = observe_command
        self.observe_vel = observe_vel
        self.observe_only_lin_vel = observe_only_lin_vel
        self.observe_only_ang_vel = observe_only_ang_vel
        self.observe_yaw = observe_yaw
        self.measure_heights = measure_heights

        self.curriculum_cfg = curriculum_cfg or {}
        self.domain_rand_cfg = domain_rand_cfg or {}

        # Load MJCF model
        sys = mjcf.load('path/to/robot.xml')
        self.sys = sys

        # Curriculum initialization
        self.curriculum = RewardThresholdCurriculum(
            seed=self.curriculum_cfg.get("seed", 0),
            x_vel=(self.curriculum_cfg.get("limit_vel_x", (-1.0, 1.0))),
            y_vel=(self.curriculum_cfg.get("limit_vel_y", (-0.5, 0.5))),
            yaw_vel=(self.curriculum_cfg.get("limit_vel_yaw", (-1.0, 1.0))),
        )
        self.env_command_bins = np.zeros((self._num_envs,), dtype=int)

        # Random initial params
        self._friction_coeffs = 1.0
        self._payloads = np.zeros((self._num_envs,))
        self._motor_strengths = np.ones((self._num_envs, self.sys.act_size()))

        # Commands
        self.commands = jnp.zeros((self._num_envs, 3))

        super().__init__(config=None, seed=seed)

    def reset(self, rng: jnp.ndarray):
        # Randomize environment parameters
        rng, rng_init = jax.random.split(rng)
        qpos = self.sys.default_q
        qvel = jnp.zeros_like(self.sys.default_qd)

        # Sample new commands from curriculum:
        new_commands, new_bins = self.curriculum.sample(self._num_envs)
        self.commands = jnp.array(new_commands)
        self.env_command_bins = new_bins

        # Domain randomization (example: randomize payload)
        if self.domain_rand_cfg.get("randomize_base_mass", False):
            rng, subrng = jax.random.split(rng)
            payload_range = self.domain_rand_cfg.get("added_mass_range", (0.0, 5.0))
            self._payloads = jax.random.uniform(subrng, (self._num_envs,)) * (payload_range[1] - payload_range[0]) + \
                             payload_range[0]

        # Apply payload as mass changes - in Brax, mass changes require reconstructing sys or a similar workaround.
        # For simplicity, we won't dynamically change it here, but you could by modifying sys.def.
        # Similar approach can be taken for friction.

        # Construct initial brax state
        pipeline_state = self.sys.init_qp(qpos, qvel)
        state = env.State(
            qp=pipeline_state,
            obs=self._get_observation(pipeline_state),
            reward=jnp.zeros(self._num_envs),
            done=jnp.zeros(self._num_envs, dtype=bool),
            metrics={}
        )
        self._current_step = 0
        return state, rng

    def step(self, state: env.State, action: jp.ndarray, rng: jp.ndarray):
        # Clip and scale actions
        action = jnp.clip(action, -1.0, 1.0) * self._action_scale

        # Compute torques: for simplicity assume direct torque control
        # If PD control is desired, it needs joint pos/vel from state.qp
        act = action * self._motor_strengths.mean(axis=1, keepdims=True)  # simplified

        # Run physics
        qp, info = self.sys.step(state.qp, act, self._dt)

        # Compute observations, rewards, done
        obs = self._get_observation(qp)
        rew = self._compute_reward(qp, action)
        done = jnp.where(self._current_step >= self._episode_length, True, False)

        metrics = {}
        extras = self._get_privileged_info(qp)

        new_state = state.replace(
            qp=qp,
            obs=obs,
            reward=rew,
            done=done,
            metrics=metrics
        )

        self._current_step += 1

        return new_state, rng, extras

    def _get_observation(self, qp: brax.QP):
        # Extract robot base orientation, velocities, etc.
        # Example for a single agent scenario
        # Brax states: qp.pos, qp.vel, etc. are shape (num_bodies, 3)
        base_pos = qp.pos[0]
        base_vel = qp.vel[0]
        # Orientation can be derived from qp.rot
        base_ang_vel = qp.ang[0]

        obs_list = []

        # Gravity projection not directly needed as Brax is top-down, but you can add a constant gravity vector if needed
        gravity_vec = jnp.array([0, 0, -1])

        if self.observe_command:
            obs_list.append(self.commands)  # shape (num_envs,3)

        # Observing velocities
        if self.observe_vel:
            obs_list.append(base_vel)
            obs_list.append(base_ang_vel)

        if self.observe_only_lin_vel:
            obs_list = [self.commands, base_vel] if self.observe_command else [base_vel]

        if self.observe_only_ang_vel:
            obs_list = [self.commands, base_ang_vel] if self.observe_command else [base_ang_vel]

        if self.observe_yaw:
            # extract yaw from orientation
            # assuming z-up and quaternion in qp.rot
            # For simplicity, assume rot is a unit quaternion [w,x,y,z]
            rot = qp.rot[0]
            # Compute yaw angle
            siny_cosp = 2.0 * (rot[0] * rot[3] + rot[1] * rot[2])
            cosy_cosp = 1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3])
            yaw = jnp.arctan2(siny_cosp, cosy_cosp)
            heading_error = jnp.clip(0.5 * yaw, -1.0, 1.0)
            obs_list.append(heading_error.reshape(-1, 1) if len(heading_error.shape) == 1 else heading_error)

        # Add joint angles and velocities
        joint_pos = self.sys.link_jnt(qp)  # link_jnt gives joint angles if defined correctly
        joint_vel = self.sys.link_jntd(qp)

        obs_list.append(joint_pos)
        obs_list.append(joint_vel)

        # Optionally measure heights or other environment info (if available)
        # Not straightforward in Brax without custom fields
        # For now skip height measurements
        # If implemented, obs_list.append(height_measurements)

        # Combine all
        obs = jnp.concatenate([x if x.ndim > 1 else x[jnp.newaxis, :]
        if x.ndim == 1 else x for x in obs_list], axis=-1)
        return obs

    def _compute_reward(self, qp: brax.QP, action: jp.ndarray):
        # Compute rewards similar to Isaac Gym code
        # e.g., tracking command velocities
        base_vel = qp.vel[0]
        base_ang_vel = qp.ang[0]

        lin_vel_error = jnp.sum((self.commands[:, :2] - base_vel[:2]) ** 2, axis=-1)
        ang_vel_error = (self.commands[:, 2] - base_ang_vel[2]) ** 2

        rew_lin = jnp.exp(-lin_vel_error / 0.25)  # example sigma
        rew_ang = jnp.exp(-ang_vel_error / 0.25)

        # Combine various terms
        reward = rew_lin + rew_ang

        return reward

    def _get_privileged_info(self, qp: brax.QP):
        # Return teacher-only observations, e.g., friction, restitution, payloads, etc.
        # In brax we don't have direct friction as an easy accessible variable per step,
        # but we can store them as attributes in the Env class and return them.
        # Just as an example:
        friction = self._friction_coeffs
        payload = self._payloads
        motor_strength = self._motor_strengths.mean(axis=1)
        priv_obs = jnp.stack([jnp.ones(self._num_envs) * friction,
                              payload,
                              motor_strength], axis=-1)
        return {"privileged_obs": priv_obs}

# Example usage:
# env = LeggedRobotEnv(
#     num_envs=1,
#     curriculum_cfg={"seed":42, "limit_vel_x":(-1.,1.), "limit_vel_y":(-0.5,0.5), "limit_vel_yaw":(-1.,1.)},
#     domain_rand_cfg={"randomize_base_mass": True, "added_mass_range":(0.,5.)}
# )

# rng = jax.random.PRNGKey(0)
# state, rng = env.reset(rng)
# for i in range(1000):
#     action = jnp.zeros((1, env.sys.act_size()))
#     state, rng, extras = env.step(state, action, rng)
#     if jnp.all(state.done):
#         break
