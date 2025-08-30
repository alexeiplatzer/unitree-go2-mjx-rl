import jax

from quadruped_mjx_rl.environments.physics_pipeline import PipelineModel


def domain_randomize(pipeline_model: PipelineModel, rng: jax.Array):
    """Randomizes the mjx.Model."""

    @jax.vmap
    def rand(rng):
        _, key = jax.random.split(rng, 2)
        # friction
        friction = jax.random.uniform(key, (1,), minval=0.6, maxval=1.4)
        friction = pipeline_model.model.geom_friction.at[:, 0].set(friction)
        # actuator
        _, key = jax.random.split(key, 2)
        gain_range = (-5, 5)
        param = (
            jax.random.uniform(key, (1,), minval=gain_range[0], maxval=gain_range[1])
            + pipeline_model.model.actuator_gainprm[:, 0]
        )
        gain = pipeline_model.model.actuator_gainprm.at[:, 0].set(param)
        bias = pipeline_model.model.actuator_biasprm.at[:, 1].set(-param)
        return friction, gain, bias

    friction, gain, bias = rand(rng)

    in_axes = jax.tree.map(lambda x: None, pipeline_model)
    in_axes = in_axes.replace(model=in_axes.model.tree_replace(
        {
            "geom_friction": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
        }
    ))

    pipeline_model = pipeline_model.replace(model=pipeline_model.model.tree_replace(
        {
            "geom_friction": friction,
            "actuator_gainprm": gain,
            "actuator_biasprm": bias,
        }
    ))

    return pipeline_model, in_axes
