from dataclasses import dataclass, field

import jax

from quadruped_mjx_rl.types import PRNGKey
from quadruped_mjx_rl.physics_pipeline import EnvModel, PipelineModel
from quadruped_mjx_rl.domain_randomization.types import DomainRandomizationConfig


@dataclass
class ObstaclePositionRandomizationConfig(DomainRandomizationConfig):
    obstacle_names: list[str] = field(default_factory=lambda: ["cylinder_0", "cylinder_1"])
    offset_min: float = -3.0
    offset_max: float = 3.0

    def domain_randomize(
        self,
        pipeline_model: PipelineModel,
        env_model: EnvModel,
        rng_key: PRNGKey,
        num_worlds: int,
    ) -> tuple[PipelineModel, PipelineModel]:
        obstacle_ids = [
            env_model.body(obstacle_name).id for obstacle_name in self.obstacle_names
        ]

        @jax.vmap
        def rand(rng: jax.Array):
            body_pos = pipeline_model.model.body_pos
            for obstacle_id in obstacle_ids:
                rng, offset_key = jax.random.split(rng, 2)
                offset = jax.random.uniform(
                    offset_key, shape=(2,), minval=self.offset_min, maxval=self.offset_max
                )
                body_pos = body_pos.at[obstacle_id, :2].set(body_pos[obstacle_id, :2] + offset)
            return body_pos

        key_envs = jax.random.split(rng_key, num_worlds)
        body_pos = rand(key_envs)

        in_axes = jax.tree_util.tree_map(lambda x: None, pipeline_model)
        in_axes = in_axes.replace(
            model=in_axes.model.tree_replace(
                {
                    "body_pos": 0,
                }
            )
        )

        pipeline_model = pipeline_model.replace(
            model=pipeline_model.model.tree_replace(
                {
                    "body_pos": body_pos,
                }
            )
        )
        return pipeline_model, in_axes
