from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from quadruped_mjx_rl import math
from quadruped_mjx_rl.physics_pipeline import Motion, PipelineState
from quadruped_mjx_rl.environments.quadruped.base import (
    EnvironmentConfig,
    QuadrupedBaseEnv,
    register_environment_config_class,
)
from quadruped_mjx_rl.types import Action, Observation


@dataclass
class QuadrupedWanderEnvConfig(EnvironmentConfig):
    @dataclass
    class RewardConfig(EnvironmentConfig.RewardConfig):
        # We target a modest cruising speed
        target_speed: float = 0.8

        @dataclass
        class ScalesConfig(EnvironmentConfig.RewardConfig.ScalesConfig):
            # --- Primary Driver ---
            # Reward moving forward in the body frame.
            # This allows the robot to turn freely; it just has to move where it faces.
            forward_speed: float = 1.0

            # --- Terrain Penalties (The Avoidance Signals) ---
            # Heavily punish slipping to force avoidance of low-friction tiles
            foot_slip: float = -2.0

            # Punish high torques/energy.
            # Soft terrain (low stiffness) might require more torque to push off.
            torques: float = -0.0005
            energy: float = -0.005

            # Penalize tumbling/instability
            orientation: float = -5.0
            termination: float = -1.0

            # Allow turning, but punish rapid spinning so it doesn't just do donuts
            ang_vel_xy: float = -0.05
            ang_vel_yaw: float = -0.1

        scales: ScalesConfig = field(default_factory=ScalesConfig)

    rewards: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "QuadrupedWander"

    @classmethod
    def get_environment_class(cls) -> type[QuadrupedBaseEnv]:
        return QuadrupedWanderEnv


register_environment_config_class(QuadrupedWanderEnvConfig)


class QuadrupedWanderEnv(QuadrupedBaseEnv):
    """
    A Free-Roaming environment where the robot is rewarded for maintaining a
    forward velocity in its own body frame. It must avoid terrain that causes
    slipping or high energy consumption to maximize rewards.
    """

    def __init__(
        self,
        environment_config: QuadrupedWanderEnvConfig,
        robot_config,
        env_model,
    ):
        super().__init__(environment_config, robot_config, env_model)
        self._target_speed = environment_config.rewards.target_speed

    def _get_rewards(
        self,
        pipeline_state: PipelineState,
        state_info: dict[str, Any],
        action: Action,
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        # Get standard rewards/penalties from base
        rewards = super()._get_rewards(pipeline_state, state_info, action, done)

        # Extract useful states
        xd = pipeline_state.xd

        # 1. Forward Speed Reward (Body Frame)
        # Convert global velocity to local body frame
        inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        local_vel = math.rotate(xd.vel[0], inv_torso_rot)
        v_x = local_vel[0]

        # Reward: Clip at target speed.
        # We don't want it running infinitely fast, just cruising efficiently.
        # Logic: Reward increases linearly up to target_speed, then plateaus.
        rewards["forward_speed"] = jnp.minimum(v_x, self._target_speed)

        # 2. Energy Penalty
        # Torques are already in rewards["torques"], but let's add specific Energy (Power)
        # Power = Torque * Joint Velocity
        motor_torques = pipeline_state.data.qfrc_actuator[6:]  # Ignore free joint
        motor_vels = pipeline_state.qd[7:19]
        power = jnp.sum(jnp.abs(motor_torques * motor_vels))
        rewards["energy"] = power

        # 3. Add the donut-prevention penalty
        # We access the 'ang_vel_yaw' scale from self.reward_scales (inherited from base)
        # Note: Ensure 'ang_vel_yaw' is in your yaml config scales!
        rewards["ang_vel_yaw"] = self._reward_ang_vel_yaw(pipeline_state.xd)

        return rewards

    def _reward_ang_vel_yaw(self, xd: Motion) -> jax.Array:
        # Penalize Z-axis angular velocity (Yaw Rate)
        # xd.ang[0] is the angular velocity vector [x, y, z]
        return jnp.square(xd.ang[0, 2])