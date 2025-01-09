from etils import epath

PROJECT_ROOT_PATH = epath.Path(__file__).parent.parent

ckpt_path = PROJECT_ROOT_PATH / "quadrupred_joystick/ckpts"
ckpt_path.mkdir(parents=True, exist_ok=True)

model_path = PROJECT_ROOT_PATH / "mjx_brax_go2_policy"

configurations_path = epath.Path(__file__).parent.parent / "configurations"
