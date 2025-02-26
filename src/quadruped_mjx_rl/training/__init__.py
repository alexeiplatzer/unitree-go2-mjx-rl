from .configs import TrainingConfig


name_to_training_config = {
    "default": lambda: TrainingConfig(),
}
