from etils.epath import Path

PROJECT_ROOT_DIRECTORY = Path(__file__).parent.parent

CONFIGS_DIRECTORY = PROJECT_ROOT_DIRECTORY / "configs"

RESOURCES_DIRECTORY = PROJECT_ROOT_DIRECTORY / "resources"
UNITREE_GO2_RESOURCES = RESOURCES_DIRECTORY / "unitree_go2"
unitree_go2_init_scene = UNITREE_GO2_RESOURCES / "scene_mjx.xml"
BARKOUR_RESOURCES = RESOURCES_DIRECTORY / "google_barkour_vb"
barkour_init_scene = BARKOUR_RESOURCES / "scene_mjx.xml"

EXPERIMENTS_DIRECTORY = PROJECT_ROOT_DIRECTORY / "experiments"
EXPERIMENT_CONFIGS_DIRECTORY = EXPERIMENTS_DIRECTORY / "configs"
EXPERIMENT_CONFIGS_DIRECTORY.mkdir(parents=True, exist_ok=True)

TRAINED_POLICIES_DIRECTORY = EXPERIMENTS_DIRECTORY / "trained_policies"
ckpt_path = TRAINED_POLICIES_DIRECTORY / "quadrupred_joystick/ckpts"
ckpt_path.mkdir(parents=True, exist_ok=True)

ROLLOUTS_DIRECTORY = EXPERIMENTS_DIRECTORY / "rollouts"
ROLLOUTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
