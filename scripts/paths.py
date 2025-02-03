from etils.epath import Path

PROJECT_ROOT_DIRECTORY = Path(__file__).parent.parent

CONFIGS_DIRECTORY = PROJECT_ROOT_DIRECTORY / "configs"

RESOURCES_DIRECTORY = PROJECT_ROOT_DIRECTORY / "resources"
scene_path = RESOURCES_DIRECTORY / "scene_mjx.xml"

TRAINED_POLICIES_DIRECTORY = CONFIGS_DIRECTORY / "trained_policies"
ckpt_path = TRAINED_POLICIES_DIRECTORY / "quadrupred_joystick/ckpts"
ckpt_path.mkdir(parents=True, exist_ok=True)
