from dataclasses import dataclass

import jax

from quadruped_mjx_rl.types import PRNGKey
from quadruped_mjx_rl.physics_pipeline import PipelineModel, EnvModel
from quadruped_mjx_rl.domain_randomization.types import DomainRandomizationConfig


@dataclass
class PhysicsDomainRandomizationConfig(DomainRandomizationConfig):
    friction_max: float = 1.4
    friction_min: float = 0.6
    gain_max: float = 5.0
    gain_min: float = -5.0

    def domain_randomize(
        self,
        pipeline_model: PipelineModel,
        env_model: EnvModel,
        key: PRNGKey,
        num_worlds: int,
    ) -> tuple[PipelineModel, PipelineModel]:
        """Randomizes the mjx.Model."""

        @jax.vmap
        def rand(rng):
            friction_key, actuator_key = jax.random.split(rng, 2)
            # friction
            friction = jax.random.uniform(
                friction_key, (1,), minval=self.friction_min, maxval=self.friction_max
            )
            friction = pipeline_model.model.geom_friction.at[:, 0].set(friction)
            # actuator
            param = (
                jax.random.uniform(
                    actuator_key, (1,), minval=self.gain_min, maxval=self.gain_max
                )
                + pipeline_model.model.actuator_gainprm[:, 0]
            )
            gain = pipeline_model.model.actuator_gainprm.at[:, 0].set(param)
            bias = pipeline_model.model.actuator_biasprm.at[:, 1].set(-param)
            return friction, gain, bias

        key_envs = jax.random.split(key, num_worlds)
        friction, gain, bias = rand(key_envs)

        in_axes = jax.tree.map(lambda x: None, pipeline_model)
        in_axes = in_axes.replace(
            model=in_axes.model.tree_replace(
                {
                    "geom_friction": 0,
                    "actuator_gainprm": 0,
                    "actuator_biasprm": 0,
                }
            )
        )

        pipeline_model = pipeline_model.replace(
            model=pipeline_model.model.tree_replace(
                {
                    "geom_friction": friction,
                    "actuator_gainprm": gain,
                    "actuator_biasprm": bias,
                }
            )
        )

        return pipeline_model, in_axes
