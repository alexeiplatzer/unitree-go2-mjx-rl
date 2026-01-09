from pathlib import Path

PROJECT_ROOT_DIRECTORY = Path(__file__).parent.parent

CONFIGS_DIRECTORY = PROJECT_ROOT_DIRECTORY / "configs"

# Minimal information needed about the robot for training
ROBOT_CONFIGS_DIRECTORY = CONFIGS_DIRECTORY / "robot_configs"
ROBOT_CONFIGS_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Environment and Terrain configs
ENVIRONMENT_CONFIGS_DIRECTORY = CONFIGS_DIRECTORY / "environment_configs"
ENVIRONMENT_CONFIGS_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Model and Training configs
MODEL_CONFIGS_DIRECTORY = CONFIGS_DIRECTORY / "model_configs"
MODEL_CONFIGS_DIRECTORY.mkdir(parents=True, exist_ok=True)

# Full definitions of Robots and Some terrain scenes in MJCF
RESOURCES_DIRECTORY = PROJECT_ROOT_DIRECTORY / "resources"

# Experiment results, visualizations, configs, etc.
EXPERIMENTS_DIRECTORY = PROJECT_ROOT_DIRECTORY / "experiments"
EXPERIMENTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
