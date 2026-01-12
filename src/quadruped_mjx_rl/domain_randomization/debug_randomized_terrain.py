from dataclasses import dataclass

import jax
import mujoco

from quadruped_mjx_rl.types import PRNGKey
from quadruped_mjx_rl.physics_pipeline import PipelineModel, EnvModel
from quadruped_mjx_rl.domain_randomization.types import DomainRandomizationConfig


@dataclass
class SurfaceDomainRandomizationConfig(DomainRandomizationConfig):
    friction_min: float = 0.01
    friction_max: float = 2.00
    stiffness_min: float = 0.002
    stiffness_max: float = 0.100

    @classmethod
    def default_easier(cls) -> "SurfaceDomainRandomizationConfig":
        return SurfaceDomainRandomizationConfig(
            friction_min=1.00, friction_max=1.00, stiffness_min=0.02, stiffness_max=0.02
        )

    def domain_randomize(
        self,
        pipeline_model: PipelineModel,
        env_model: EnvModel,
        key: PRNGKey,
        num_worlds: int,
    ) -> tuple[PipelineModel, PipelineModel]:
        """Randomizes the mjx.Model floor surface properties (friction and stiffness)."""
        floor_id = mujoco.mj_name2id(env_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        @jax.vmap
        def rand(rng):
            friction_key, stiffness_key = jax.random.split(rng, 2)
            # friction
            friction = jax.random.uniform(
                friction_key, (1,), minval=self.friction_min, maxval=self.friction_max
            )
            friction = pipeline_model.model.geom_friction.at[floor_id, 0].set(friction)
            # actuator
            stiffness = jax.random.uniform(
                stiffness_key, (1,), minval=self.stiffness_min, maxval=self.stiffness_max
            )
            stiffness = pipeline_model.model.geom_solref.at[floor_id, 0].set(stiffness)
            return friction, stiffness

        key_envs = jax.random.split(key, num_worlds)
        friction, stiffness = rand(key_envs)

        in_axes = jax.tree.map(lambda x: None, pipeline_model)
        in_axes = in_axes.replace(
            model=in_axes.model.tree_replace(
                {
                    "geom_friction": 0,
                    "geom_solref": 0,
                }
            )
        )

        pipeline_model = pipeline_model.replace(
            model=pipeline_model.model.tree_replace(
                {
                    "geom_friction": friction,
                    "geom_solref": stiffness,
                }
            )
        )

        return pipeline_model, in_axes
