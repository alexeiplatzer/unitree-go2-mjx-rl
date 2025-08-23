
import functools

from etils.epath import PathLike

from quadruped_mjx_rl import running_statistics
from quadruped_mjx_rl.models import io
from quadruped_mjx_rl.models.architectures import (
    guided_actor_critic as guided_networks, raw_actor_critic as raw_networks,
    TeacherStudentAgentParams,
)
from quadruped_mjx_rl.models.configs import (
    ActorCriticConfig, ModelConfig, TeacherStudentConfig, TeacherStudentVisionConfig,
)
from quadruped_mjx_rl.models.networks import (
    NetworkFactory,
)


def get_networks_factory(
    model_config: ModelConfig,
) -> NetworkFactory:
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
    """Utility function to quickly get a policy function from a model config
    and the saved pre-trained params path."""
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
        teacher_policy_factory, student_policy_factory = teacher_student_networks.policy_metafactory()
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
        policy_factory, = ppo_nets.policy_metafactory()
        return policy_factory(params, deterministic=True)
    else:
        raise NotImplementedError
