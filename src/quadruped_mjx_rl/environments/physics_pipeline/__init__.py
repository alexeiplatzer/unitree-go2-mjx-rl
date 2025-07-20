GLOBAL_PHYSICS_PIPELINE = "brax"  # or brax

if GLOBAL_PHYSICS_PIPELINE == "local":
    from quadruped_mjx_rl.environments.physics_pipeline.base import Motion, Transform
    from mujoco import MjModel as EnvModel
    from mujoco.mjx import Model as PipelineModel, Data as PipelineState
    from quadruped_mjx_rl.environments.physics_pipeline.physics_pipeline import (
        pipeline_init, pipeline_step
    )
    from quadruped_mjx_rl.environments.physics_pipeline.rendering import render_array
    from quadruped_mjx_rl.environments.physics_pipeline.loading import (
        model_load, make_pipeline_model
    )
    from quadruped_mjx_rl.environments.physics_pipeline.environments import Env, State, Wrapper
    from quadruped_mjx_rl.environments.physics_pipeline.wrappers import (
        EvalMetrics,
        EvalWrapper,
        VmapWrapper,
        AutoResetWrapper,
        EpisodeWrapper,
        DomainRandomizationVmapWrapper,
    )
elif GLOBAL_PHYSICS_PIPELINE == "brax":
    from brax.base import Motion, Transform, System as PipelineModel
    from mujoco import MjModel as EnvModel
    from brax.mjx.base import State as PipelineState
    from brax.mjx.pipeline import init as pipeline_init, step as pipeline_step
    from brax.io.image import render_array
    from brax.io.mjcf import load_mjmodel as model_load, load_model as make_pipeline_model
    from brax.envs.base import Env, State, Wrapper
    from brax.envs.wrappers.training import (
        EvalMetrics,
        EvalWrapper,
        VmapWrapper,
        AutoResetWrapper,
        EpisodeWrapper,
        DomainRandomizationVmapWrapper
    )
else:
    raise NotImplementedError
