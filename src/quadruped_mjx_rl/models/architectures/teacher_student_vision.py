from collections.abc import Mapping
from dataclasses import dataclass, field

from flax import linen

from quadruped_mjx_rl import types
from quadruped_mjx_rl.models import distributions
from quadruped_mjx_rl.models.architectures.actor_critic_base import ActorCriticConfig
from quadruped_mjx_rl.models.architectures.configs_base import register_model_config_class
from quadruped_mjx_rl.models.architectures.teacher_student_base import (
    TeacherStudentConfig,
    TeacherStudentNetworks,
)
from quadruped_mjx_rl.models.base_modules import ActivationFn, CNN, MLP
from quadruped_mjx_rl.models.networks_utils import make_network


@dataclass
class TeacherStudentVisionConfig(TeacherStudentConfig):
    @dataclass
    class ModulesConfig(ActorCriticConfig.ModulesConfig):
        encoder_convolutional: list[int] = field(default_factory=lambda: [32, 64, 64])
        encoder_dense: list[int] = field(default_factory=lambda: [256, 256])
        adapter_convolutional: list[int] = field(default_factory=lambda: [32, 64, 64])
        adapter_dense: list[int] = field(default_factory=lambda: [256, 256])

    modules: ModulesConfig = field(default_factory=ModulesConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudentVision"


register_model_config_class(TeacherStudentVisionConfig)


def make_teacher_student_vision_networks(
    *,
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    teacher_obs_key: str = "pixels/terrain/depth",
    student_obs_key: str = "pixels/frontal_ego/rgb_adjusted",
    common_obs_key: str = "proprioceptive",
    latent_obs_key: str = "latent",
    model_config: TeacherStudentVisionConfig = TeacherStudentVisionConfig(),
    activation: ActivationFn = linen.swish,
):
    """Make teacher-student network with preprocessor."""

    # Sanity check
    if not isinstance(model_config, TeacherStudentVisionConfig):
        raise TypeError("Model configuration must be a TeacherStudentVisionConfig instance.")
    if not isinstance(observation_size, Mapping):
        raise TypeError(
            f"Environment observations must be a dictionary (Mapping),"
            f" got {type(observation_size)}"
        )
    if teacher_obs_key not in observation_size:
        raise ValueError(
            f"Teacher observation key {teacher_obs_key} must be in environment observations."
        )
    if student_obs_key not in observation_size:
        raise ValueError(
            f"Student observation key {student_obs_key} must be in environment observations."
        )
    if common_obs_key not in observation_size:
        raise ValueError(
            f"Common observation key {common_obs_key} must be in environment observations."
        )

    observation_size = dict(observation_size) | {latent_obs_key: (model_config.latent_size,)}

    parametric_action_distribution = distributions.NormalTanhDistribution(
        event_size=action_size
    )

    teacher_encoder_module = CNN(
        num_filters=model_config.modules.encoder_convolutional,
        dense_layer_sizes=model_config.modules.encoder_dense + [model_config.latent_size],
        activation=activation,
        activate_final=True,
    )
    student_encoder_module = CNN(
        num_filters=model_config.modules.adapter_convolutional,
        dense_layer_sizes=model_config.modules.adapter_dense + [model_config.latent_size],
        activation=activation,
        activate_final=True,
    )
    # Visual observations are not preprocessed
    teacher_preprocess_keys = ()
    student_preprocess_keys = ()

    teacher_encoder_network = make_network(
        module=teacher_encoder_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=teacher_preprocess_keys,
        apply_to_obs_keys=(teacher_obs_key,),
        squeeze_output=False,
    )

    student_encoder_network = make_network(
        module=student_encoder_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=student_preprocess_keys,
        apply_to_obs_keys=(student_obs_key,),
        squeeze_output=False,
    )

    policy_module = MLP(
        layer_sizes=model_config.modules.policy + [parametric_action_distribution.param_size],
        activation=activation,
        activate_final=False,
    )

    value_module = MLP(
        layer_sizes=model_config.modules.value + [1],
        activation=activation,
        activate_final=False,
    )

    policy_network = make_network(
        module=policy_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=(common_obs_key,),
        apply_to_obs_keys=(common_obs_key, latent_obs_key),
        squeeze_output=False,
        concatenate_inputs=True,
    )

    value_network = make_network(
        module=value_module,
        obs_size=observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        preprocess_obs_keys=(common_obs_key,),
        apply_to_obs_keys=(common_obs_key, latent_obs_key),
        squeeze_output=True,
        concatenate_inputs=True,
    )

    policy_raw_apply = policy_network.apply
    value_raw_apply = value_network.apply

    def policy_apply(
        processor_params: types.PreprocessorParams,
        params: types.Params,
        obs: dict[str, types.ndarray],
        latent_encoding: types.ndarray,
    ):
        obs = obs | {latent_obs_key: latent_encoding}
        return policy_raw_apply(processor_params=processor_params, params=params, obs=obs)

    def value_apply(
        processor_params: types.PreprocessorParams,
        params: types.Params,
        obs: dict[str, types.ndarray],
        latent_encoding: types.ndarray,
    ):
        obs = obs | {latent_obs_key: latent_encoding}
        return value_raw_apply(processor_params=processor_params, params=params, obs=obs)

    policy_network.apply = policy_apply
    value_network.apply = value_apply

    return TeacherStudentNetworks(
        teacher_encoder_network=teacher_encoder_network,
        student_encoder_network=student_encoder_network,
        policy_network=policy_network,
        value_network=value_network,
        parametric_action_distribution=parametric_action_distribution,
    )
