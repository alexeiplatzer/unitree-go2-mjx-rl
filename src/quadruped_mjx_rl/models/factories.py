import functools

from etils.epath import PathLike

from quadruped_mjx_rl import running_statistics
from quadruped_mjx_rl.models import io
from quadruped_mjx_rl.models.architectures import (
    TeacherStudentAgentParams,
    ActorCriticConfig,
    ModelConfig,
    TeacherStudentConfig,
    TeacherStudentVisionConfig,
    TeacherStudentRecurrentConfig,
)
from quadruped_mjx_rl.models.types import NetworkFactory


def get_networks_factory(
    model_config: ModelConfig,
) -> NetworkFactory:
    """Checks the model type from the configuration and returns the appropriate factory."""
    model_class = type(model_config).get_model_class()
    return functools.partial(model_class, model_config=model_config)


def load_inference_fn(
    model_path: PathLike,
    model_config: ModelConfig,
    action_size: int,
):
    """Utility function to quickly get a policy function from a model config
    and the saved pre-trained params path."""
    model_class = type(model_config).get_model_class()
    params = io.load_params(model_path)
    if not isinstance(params, model_class.agent_params_class()):
        raise ValueError(
            f"The restored params have the wrong type: {type(params)}"
            f" - while {model_class.agent_params_class()} were expected!"
        )
    network_factories = get_networks_factory(model_config)
    if isinstance(model_config, TeacherStudentConfig):
        teacher_student_networks = network_factories(
            observation_size={
                "proprioceptive": 1,
                "proprioceptive_history": 1,
                "environment_privileged": 1,
                "pixels/terrain/depth": 1,
                "pixels/frontal_ego/rgb": 1,
                "pixels/frontal_ego/rgb_adjusted": 1,
                "privileged_terrain_map": 1,
            },
            action_size=action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
        teacher_policy_factory, student_policy_factory = (
            teacher_student_networks.policy_metafactory()
        )
        teacher_inference_fn = teacher_policy_factory(params, deterministic=True)
        student_inference_fn = student_policy_factory(params, deterministic=True)
        return {
            "teacher": teacher_inference_fn,
            "student": student_inference_fn,
        }
    elif isinstance(model_config, ActorCriticConfig):
        ppo_nets = network_factories(
            observation_size=1,
            action_size=action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
        (policy_factory,) = ppo_nets.policy_metafactory()
        return policy_factory(params, deterministic=True)
    else:
        raise NotImplementedError
