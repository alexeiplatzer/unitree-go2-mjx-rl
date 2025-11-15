import jax

from quadruped_mjx_rl.types import PRNGKey
from quadruped_mjx_rl.environments.physics_pipeline import EnvModel, PipelineModel


def terrain_randomize(
    pipeline_model: PipelineModel,
    env_model: EnvModel,
    rng_key: PRNGKey,
    num_worlds: int,
) -> tuple[PipelineModel, PipelineModel]:
    cylinder_1_id = env_model.body("cylinder_0").id
    cylinder_2_id = env_model.body("cylinder_1").id

    @jax.vmap
    def rand(rng: jax.Array):
        _, key = jax.random.split(rng, 2)
        key_1, key_2, key = jax.random.split(key, 3)
        offset_1 = jax.random.uniform(key_1, shape=(2,), minval=-3.0, maxval=3.0)
        offset_2 = jax.random.uniform(key_2, shape=(2,), minval=-3.0, maxval=3.0)
        body_pos = pipeline_model.model.body_pos
        body_pos = body_pos.at[cylinder_1_id, :2].set(body_pos[cylinder_1_id, :2] + offset_1)
        body_pos = body_pos.at[cylinder_2_id, :2].set(body_pos[cylinder_2_id, :2] + offset_2)
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
