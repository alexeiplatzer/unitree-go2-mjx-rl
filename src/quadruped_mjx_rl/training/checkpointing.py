from orbax import checkpoint as ocp
from flax.training import orbax_utils


def policy_params_fn(current_step, parameters, checkpoints_save_path):
    # save checkpoints
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(parameters)
    path = checkpoints_save_path / f"{current_step}"
    orbax_checkpointer.save(path, parameters, force=True, save_args=save_args)
