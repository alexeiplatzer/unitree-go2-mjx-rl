from src.quadruped_mjx_rl.training_environments import PROJECT_ROOT_PATH

ckpt_path = PROJECT_ROOT_PATH / 'quadrupred_joystick/ckpts'
ckpt_path.mkdir(parents=True, exist_ok=True)

model_path = PROJECT_ROOT_PATH / 'mjx_brax_go2_policy'