
# Supporting
import functools
from etils.epath import PathLike
from quadruped_mjx_rl.models import io

# Math
from quadruped_mjx_rl.models import running_statistics

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
    StudentNetworks,
    TeacherNetworks,
)


def get_networks_factory(
    model_config: ModelConfig,
) -> (
    dict[str, functools.partial[TeacherNetworks] | functools.partial[StudentNetworks]]
    | functools.partial[ActorCriticNetworks]
):
    if isinstance(model_config, TeacherStudentVisionConfig):
        teacher_factory = functools.partial(
            guided_networks.make_teacher_networks,
            policy_hidden_layer_sizes=model_config.modules.policy,
            value_hidden_layer_sizes=model_config.modules.value,
            latent_representation_size=model_config.latent_size,
            encoder_hidden_layer_sizes=model_config.modules.encoder_dense,
            encoder_convolutional_layer_sizes=model_config.modules.encoder_convolutional,
            encoder_obs_key="pixels/view_terrain",
        )
        student_factory = functools.partial(
            guided_networks.make_student_networks,
            latent_representation_size=model_config.latent_size,
            adapter_hidden_layer_sizes=model_config.modules.adapter_dense,
            adapter_convolutional_layer_sizes=model_config.modules.adapter_convolutional,
            encoder_obs_key="pixels/view_frontal_ego",
        )
        return {"teacher": teacher_factory, "student": student_factory}
    if isinstance(model_config, TeacherStudentConfig):
        teacher_factory = functools.partial(
            guided_networks.make_teacher_networks,
            policy_hidden_layer_sizes=model_config.modules.policy,
            value_hidden_layer_sizes=model_config.modules.value,
            latent_representation_size=model_config.latent_size,
            encoder_hidden_layer_sizes=model_config.modules.encoder,
        )
        student_factory = functools.partial(
            guided_networks.make_student_networks,
            latent_representation_size=model_config.latent_size,
            adapter_hidden_layer_sizes=model_config.modules.adapter,
        )
        return {"teacher": teacher_factory, "student": student_factory}
    if isinstance(model_config, ActorCriticConfig):
        return functools.partial(
            raw_networks.make_networks,
            policy_hidden_layer_sizes=model_config.modules.policy,
            value_hidden_layer_sizes=model_config.modules.value,
        )
    raise NotImplementedError


def load_inference_fn(
    model_path: PathLike,
    model_config: ModelConfig,
    action_size: int,
):
    params = io.load_params(model_path)
    network_factories = get_networks_factory(model_config)
    if isinstance(model_config, TeacherStudentConfig):
        teacher_params, student_params = params
        teacher_nets = network_factories["teacher"](
            observation_size=1,
            privileged_observation_size=1,
            action_size=action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
        student_nets = network_factories["student"](
            observation_size=1,
            preprocess_observations_fn=running_statistics.normalize,
        )
        teacher_inference_factory = guided_networks.make_teacher_inference_fn(teacher_nets)
        assert isinstance(student_nets, StudentNetworks)
        student_inference_factory = guided_networks.make_student_inference_fn(
            teacher_nets, student_nets
        )
        return {
            "teacher": teacher_inference_factory(teacher_params),
            "student": student_inference_factory(teacher_params, student_params),
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
