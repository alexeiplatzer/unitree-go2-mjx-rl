from collections.abc import Callable
from etils.epath import PathLike
from functools import partial

from brax.training.acme import running_statistics
from brax.io import model

from brax.training.agents.ppo import networks as ppo_networks

from ..brax_alt.training.agents.teacher import networks as teacher_networks

from .configs import ModelConfig, ActorCriticConfig, TeacherStudentConfig


def get_networks_factory(
    model_config: ModelConfig,
):
    if isinstance(model_config, TeacherStudentConfig):
        teacher_factory = partial(
            teacher_networks.make_teacher_networks,
            policy_hidden_layer_sizes=model_config.modules.policy,
            value_hidden_layer_sizes=model_config.modules.value,
            latent_representation_size=model_config.latent_size,
            encoder_hidden_layer_sizes=model_config.modules.encoder,
        )
        student_factory = partial(
            teacher_networks.make_student_networks,
            latent_representation_size=model_config.latent_size,
            adapter_hidden_layer_sizes=model_config.modules.adapter,
        )
        return {"teacher": teacher_factory, "student": student_factory}
    elif isinstance(model_config, ActorCriticConfig):
        return partial(
            ppo_networks.make_ppo_networks,
            policy_hidden_layer_sizes=model_config.modules.policy,
            value_hidden_layer_sizes=model_config.modules.value,
        )
    else:
        raise NotImplementedError


def load_inference_fn(
    model_path: PathLike,
    model_config: ModelConfig,
    action_size: int,
):
    params = model.load_params(model_path)
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
        teacher_inference_factory = teacher_networks.make_teacher_inference_fn(teacher_nets)
        student_inference_factory = teacher_networks.make_student_inference_fn(
            teacher_nets, student_nets
        )
        return {
            "teacher": teacher_inference_factory(teacher_params),
            "student": student_inference_factory(teacher_params, student_params),
        }
    elif isinstance(model_config, ActorCriticConfig):
        ppo_nets = network_factories["ppo"](
            observation_size=1,
            action_size=action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
        ppo_inference_factory = ppo_networks.make_inference_fn(ppo_nets)
        return ppo_inference_factory(params)
    else:
        raise NotImplementedError
