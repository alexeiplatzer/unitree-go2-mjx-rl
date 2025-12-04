import functools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import jax
from flax import linen
from flax.struct import dataclass as flax_dataclass

from quadruped_mjx_rl.environments import Env, State, is_obs_key_vision
from quadruped_mjx_rl.environments.vision.vision_wrappers import VisionWrapper
from quadruped_mjx_rl.models.types import (
    AgentNetworkParams,
    Params,
    AgentParams,
    PolicyFactory,
    PreprocessObservationFn,
    identity_observation_preprocessor,
)
from quadruped_mjx_rl.models.acting import generate_unroll
from quadruped_mjx_rl.types import Observation, PRNGKey, ObservationSize, Transition
from quadruped_mjx_rl.models import (
    AgentParams,
    distributions,
    FeedForwardNetwork,
    networks_utils,
    RecurrentNetwork,
)
from quadruped_mjx_rl.models.architectures.actor_critic_base import (
    ActorCriticConfig,
    ActorCriticNetworkParams,
    ActorCriticNetworks,
    ActorCriticArchitecture,
)
from quadruped_mjx_rl.models.architectures.configs_base import (
    ComponentNetworksArchitecture,
    register_model_config_class,
)
from quadruped_mjx_rl.models.base_modules import ActivationFn, MLP, ModuleConfigMLP, ModuleConfigCNN
from quadruped_mjx_rl.models.networks_utils import (
    make_network,
    policy_factory,
)
from quadruped_mjx_rl.models.types import RecurrentCarry


@dataclass
class TeacherStudentConfig(ActorCriticConfig):
    encoder_obs_key: str = "environment_privileged"
    student_obs_key: str = "proprioceptive_history"
    latent_obs_key: str = "latent"
    encoder: ModuleConfigMLP = field(
        default_factory=lambda: ModuleConfigMLP(layer_sizes=[256, 256])
    )
    student: ModuleConfigMLP = field(
        default_factory=lambda: ModuleConfigMLP(layer_sizes=[256, 256])
    )
    latent_encoding_size: int = 16

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudent"

    @classmethod
    def get_model_class(cls) -> type["TeacherStudentNetworks"]:
        return TeacherStudentNetworks

@dataclass
class TeacherStudentVisionConfig(TeacherStudentConfig):

    encoder_obs_key: str = "pixels/terrain/depth"
    student_obs_key: str = "pixels/frontal_ego/rgb_adjusted"
    encoder: ModuleConfigCNN = field(
        default_factory=lambda: ModuleConfigCNN(
            filter_sizes=[32, 64, 64], dense=ModuleConfigMLP(layer_sizes=[256, 256])
        )
    )
    student: ModuleConfigCNN = field(
        default_factory=lambda: ModuleConfigCNN(
            filter_sizes=[32, 64, 64], dense=ModuleConfigMLP(layer_sizes=[256, 256])
        )
    )
    proprio_substeps: int = 1

    @classmethod
    def config_class_key(cls) -> str:
        return "TeacherStudentVision"

    @classmethod
    def get_model_class(cls) -> type["TeacherStudentNetworks"]:
        return TeacherStudentNetworks


register_model_config_class(TeacherStudentConfig)
register_model_config_class(TeacherStudentVisionConfig)


@flax_dataclass
class TeacherStudentNetworkParams(ActorCriticNetworkParams):
    """Contains training state for the learner."""

    teacher_encoder: Params
    student_encoder: Params


@flax_dataclass
class TeacherStudentAgentParams(AgentParams[TeacherStudentNetworkParams]):
    """Contains training state for the full agent."""

    def restore_params(
        self,
        restore_params: "TeacherStudentAgentParams",
        restore_value: bool = False,
    ):
        value_params = (
            restore_params.network_params.value if restore_value else self.network_params.value
        )
        return self.replace(
            network_params=TeacherStudentNetworkParams(
                policy=restore_params.network_params.policy,
                value=value_params,
                teacher_encoder=restore_params.network_params.teacher_encoder,
                student_encoder=restore_params.network_params.student_encoder,
            ),
            preprocessor_params=restore_params.preprocessor_params,
        )


class TeacherStudentNetworks(ActorCriticArchitecture[TeacherStudentNetworkParams]):
    def __init__(
        self,
        *,
        model_config: TeacherStudentConfig,
        observation_size: ObservationSize,
        action_size: int,
        preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
        activation: ActivationFn = linen.swish,
    ):
        """Make teacher-student network with preprocessor."""
        if isinstance(model_config, TeacherStudentVisionConfig):
            self.use_vision_obs = True
            self.proprio_substeps = model_config.proprio_substeps
        else:
            self.use_vision_obs = False
            self.proprio_substeps = 1
        # Do not preprocess visual observations
        preprocess_encoder_obs_keys = (
            ()
            if is_obs_key_vision(model_config.encoder_obs_key)
            else (model_config.encoder_obs_key,)
        )
        preprocess_student_obs_keys = (
            ()
            if is_obs_key_vision(model_config.student_obs_key)
            else (model_config.student_obs_key,)
        )
        teacher_encoder_module = model_config.encoder.create(
            activation_fn=activation,
            activate_final=True,
            extra_final_layer_size=model_config.latent_encoding_size,
        )
        self.teacher_encoder_network = make_network(
            module=teacher_encoder_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=preprocess_encoder_obs_keys,
            apply_to_obs_keys=(model_config.encoder_obs_key,),
        )
        student_encoder_module = model_config.encoder.create(
            activation_fn=activation,
            activate_final=True,
            extra_final_layer_size=model_config.latent_encoding_size,
        )
        self.student_encoder_network = make_network(
            module=student_encoder_module,
            obs_size=observation_size,
            preprocess_observations_fn=preprocess_observations_fn,
            preprocess_obs_keys=preprocess_student_obs_keys,
            apply_to_obs_keys=(model_config.student_obs_key,),
        )

        self._latent_obs_key = model_config.latent_obs_key
        self.actor_critic = ActorCriticNetworks(
            model_config=model_config,
            observation_size=observation_size,
            action_size=action_size,
            preprocess_observations_fn=preprocess_observations_fn,
            activation=activation,
            extra_input_keys=(self._latent_obs_key,),
        )

    @staticmethod
    def agent_params_class() -> type[TeacherStudentAgentParams]:
        return TeacherStudentAgentParams

    def initialize(self, rng: PRNGKey) -> TeacherStudentNetworkParams:
        policy_key, value_key, teacher_key, student_key = jax.random.split(rng, 4)
        return TeacherStudentNetworkParams(
            policy=self.actor_critic.policy_network.init(policy_key),
            value=self.actor_critic.value_network.init(value_key),
            teacher_encoder=self.teacher_encoder_network.init(teacher_key),
            student_encoder=self.student_encoder_network.init(student_key),
        )

    def apply_teacher_encoder(
        self, params: TeacherStudentAgentParams, observation: Observation
    ) -> jax.Array:
        return self.teacher_encoder_network.apply(
            params.preprocessor_params, params.network_params.teacher_encoder, observation
        )

    def apply_student_encoder(
        self, params: TeacherStudentAgentParams, observation: Observation
    ) -> jax.Array:
        return self.student_encoder_network.apply(
            params.preprocessor_params, params.network_params.student_encoder, observation
        )

    def apply_policy(
        self, params: TeacherStudentAgentParams, observation: Observation
    ) -> jax.Array:
        latent_encoding = self.apply_teacher_encoder(params, observation)
        observation = observation | {self._latent_obs_key: latent_encoding}
        return self.actor_critic.policy_network.apply(
            params.preprocessor_params, params.network_params.policy, observation
        )

    def apply_value(
        self, params: TeacherStudentAgentParams, observation: Observation
    ) -> jax.Array:
        latent_encoding = self.apply_teacher_encoder(params, observation)
        observation = observation | {self._latent_obs_key: latent_encoding}
        return self.actor_critic.value_network.apply(
            params.preprocessor_params, params.network_params.policy, observation
        )

    def get_acting_policy_factory(self) -> PolicyFactory[TeacherStudentNetworkParams]:

        def make_policy(
            params: AgentParams[TeacherStudentNetworkParams], deterministic: bool = False
        ):
            return networks_utils.policy_factory(
                policy_apply=self.apply_policy,
                parametric_action_distribution=self.actor_critic.parametric_action_distribution,
                params=params,
                deterministic=deterministic,
            )

        return make_policy

    def generate_training_unroll(
        self,
        params: TeacherStudentAgentParams,
        env: Env | VisionWrapper,
        env_state: State,
        key: PRNGKey,
        unroll_length: int,
        extra_fields: Sequence[str] = (),
    ) -> tuple[State, Transition]:
        env_state, transitions, _ = generate_unroll(
            env=env,
            env_state=env_state,
            policy=self.get_acting_policy_factory()(params, deterministic=False),
            key=key,
            unroll_length=unroll_length,
            extra_fields=extra_fields,
            add_vision_obs=self.use_vision_obs,
            proprio_steps_per_vision_step=self.proprio_substeps,
        )
        return env_state, transitions
