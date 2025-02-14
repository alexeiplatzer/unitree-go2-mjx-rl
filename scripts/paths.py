from etils.epath import Path

PROJECT_ROOT_DIRECTORY = Path(__file__).parent.parent

CONFIGS_DIRECTORY = PROJECT_ROOT_DIRECTORY / "configs"
ppo_simple_config = CONFIGS_DIRECTORY / "ppo_simple.yaml"
ppo_enhanced_config = CONFIGS_DIRECTORY / "ppo_enhanced.yaml"
unitree_go2_config = CONFIGS_DIRECTORY / "unitree_go2_environment.yaml"
barkour_config = CONFIGS_DIRECTORY / "google_barkour_vb_environment.yaml"

RESOURCES_DIRECTORY = PROJECT_ROOT_DIRECTORY / "resources"
UNITREE_GO2_RESOURCES = RESOURCES_DIRECTORY / "unitree_go2"
unitree_go2_init_scene = UNITREE_GO2_RESOURCES / "scene_mjx.xml"
BARKOUR_RESOURCES = RESOURCES_DIRECTORY / "google_barkour_vb"
barkour_init_scene = BARKOUR_RESOURCES / "scene_mjx.xml"

TRAINED_POLICIES_DIRECTORY = PROJECT_ROOT_DIRECTORY / "trained_policies"
ckpt_path = TRAINED_POLICIES_DIRECTORY / "quadrupred_joystick/ckpts"
ckpt_path.mkdir(parents=True, exist_ok=True)

ANIMATIONS_DIRECTORY = PROJECT_ROOT_DIRECTORY / "animations"
ANIMATIONS_DIRECTORY.mkdir(parents=True, exist_ok=True)
