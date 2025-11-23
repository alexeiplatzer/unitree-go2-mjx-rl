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
from quadruped_mjx_rl.models.architectures.teacher_student_vision import (
    TeacherStudentVisionConfig,
)
from quadruped_mjx_rl.models.base_modules import ActivationFn, CNN, MLP, LSTM, MixedModeRNN
from quadruped_mjx_rl.models.networks_utils import make_network


@dataclass
class TeacherStudentRecurrentConfig(TeacherStudentVisionConfig):
    @dataclass
    class ModulesConfig(TeacherStudentVisionConfig.ModulesConfig):
        encoder_convolutional: list[int] = field(default_factory=lambda: [16, 16, 16])
        adapter_convolutional: list[int] = field(default_factory=lambda: [32, 32, 32])
        adapter_visual_dense: list[int] = field(default_factory=lambda: [128, 128])
        visual_latent_size: int = 64
        adapter_recurrent_size: int = 16
        adapter_dense: list[int] = field(default_factory=lambda: [16])

    modules: ModulesConfig = field(default_factory=ModulesConfig)

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudentRecurrent"


register_model_config_class(TeacherStudentRecurrentConfig)


def make_teacher_student_recurrent_networks(
    *,
    observation_size: types.ObservationSize,
    action_size: int,
    preprocess_observations_fn: types.PreprocessObservationFn = (
        types.identity_observation_preprocessor
    ),
    teacher_obs_key: str = "privileged_terrain_map",
    student_visual_obs_key: str = "pixels/frontal_ego/rgb_adjusted",
    student_proprioceptive_obs_key: str = "proprioceptive",
    common_obs_key: str = "proprioceptive",
    latent_obs_key: str = "latent",
    model_config: TeacherStudentRecurrentConfig = TeacherStudentRecurrentConfig(),
    activation: ActivationFn = linen.swish,
):
    """Make teacher-student network with preprocessor."""

    # Sanity check
    if not isinstance(model_config, TeacherStudentRecurrentConfig):
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
    # if student_obs_key not in observation_size:
    #     raise ValueError(
    #         f"Student observation key {student_obs_key} must be in environment observations."
    #     )
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
    student_vision_encoder_module = CNN(
        num_filters=model_config.modules.adapter_convolutional,
        dense_layer_sizes=model_config.modules.adapter_visual_dense + [model_config.modules.visual_latent_size],
        activation=activation,
        activate_final=True,
    )
    student_recurrent_cell_module = LSTM(
        recurrent_layer_size=model_config.modules.adapter_recurrent_size,
        dense_layer_sizes=model_config.modules.adapter_dense + [model_config.latent_size],
    )
    student_encoder_module = MixedModeRNN(
        convolutional_module=student_vision_encoder_module,
        recurrent_module=student_recurrent_cell_module,
    )
    # Visual observations are not preprocessed
    teacher_preprocess_keys = ()
    student_preprocess_keys = (student_proprioceptive_obs_key,)

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
        apply_to_obs_keys=(student_visual_obs_key, student_proprioceptive_obs_key),
        squeeze_output=False,
        concatenate_inputs=False,
        recurrent=True,
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
