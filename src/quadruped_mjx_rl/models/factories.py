# Supporting
import functools
from etils.epath import PathLike
from quadruped_mjx_rl.models import io

# Math
from quadruped_mjx_rl import running_statistics

# Definitions
from quadruped_mjx_rl.models.configs import (
    ModelConfig,
    ActorCriticConfig,
    TeacherStudentConfig,
    TeacherStudentVisionConfig,
)
from quadruped_mjx_rl.models.architectures import (
    raw_actor_critic as raw_networks,
    guided_actor_critic as guided_networks,
    ActorCriticNetworks,
    TeacherStudentNetworks,
    ActorCriticAgentParams,
    TeacherStudentAgentParams,
)


def get_networks_factory(
    model_config: ModelConfig,
) -> functools.partial[TeacherStudentNetworks] | functools.partial[ActorCriticNetworks]:
    """Checks the model type from the configuration and returns the appropriate factory."""
    if isinstance(model_config, TeacherStudentVisionConfig):
        networks_factory = functools.partial(
            guided_networks.make_teacher_student_networks,
            model_config=model_config,
            teacher_obs_key="pixels/view_terrain",
            student_obs_key="pixels/view_frontal_ego",
        )
    elif isinstance(model_config, TeacherStudentConfig):
        networks_factory = functools.partial(
            guided_networks.make_teacher_student_networks,
            model_config=model_config,
        )
    elif isinstance(model_config, ActorCriticConfig):
        networks_factory = functools.partial(
            raw_networks.make_actor_critic_networks,
            model_config=model_config,
        )
    else:
        raise NotImplementedError
    return networks_factory


def load_inference_fn(
    model_path: PathLike,
    model_config: ModelConfig,
    action_size: int,
):
    params = io.load_params(model_path)
    network_factories = get_networks_factory(model_config)
    if isinstance(model_config, TeacherStudentConfig):
        if not isinstance(params, TeacherStudentAgentParams):
            raise ValueError(
                f"The restored params have the wrong type: {type(params)}"
                f" - while TeacherStudentAgentParams were expected!"
            )
        teacher_student_networks = network_factories(
            observation_size={
                "state": 1,
                "state_history": 1,
                "privileged_state": 1,
                "pixels/view_terrain": 1,
                "pixels/view_frontal_ego": 1,
            },
            action_size=action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
        teacher_student_inference_factory = guided_networks.make_teacher_student_inference_fns(
            teacher_student_networks
        )
        teacher_inference_fn, student_inference_fn = teacher_student_inference_factory(
            params=params
        )
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
        ppo_inference_factory = raw_networks.make_inference_fn(ppo_nets)
        return ppo_inference_factory(params)
    else:
        raise NotImplementedError
