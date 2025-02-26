from collections.abc import Callable

from brax.training.agents.ppo import networks as ppo_networks

from ..brax_alt.training.agents.teacher import networks as teacher_networks

from .configs import ModelConfig, name_to_model


def get_model_factories_by_name(name: str) -> tuple[Callable, Callable]:
    match name:
        case "actor_critic":
            return (
                ppo_networks.make_ppo_networks,
                ppo_networks.make_inference_fn,
            )
        case "teacher_student":
            return (
                teacher_networks.make_teacher_networks,
                teacher_networks.make_teacher_inference_fn,
            )
        case _:
            raise NotImplementedError
