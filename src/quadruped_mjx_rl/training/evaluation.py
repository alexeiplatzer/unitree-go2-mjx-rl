from datetime import datetime

import matplotlib.pyplot as plt
from IPython.display import display


def make_progress_fn(num_timesteps, reward_max=40):
    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]
    max_y, min_y = reward_max, 0

    fig, ax = plt.subplots()
    handle = display(fig, display_id=True)

    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics["eval/episode_reward"])
        ydataerr.append(metrics["eval/episode_reward_std"])

        ax.clear()
        ax.set_xlim(0, num_timesteps * 1.25)
        ax.set_ylim(min_y, max_y)
        ax.set_xlabel("# environment steps")
        ax.set_ylabel("reward per episode")
        ax.set_title(f"y={y_data[-1]:.3f}")
        ax.errorbar(x_data, y_data, yerr=ydataerr, fmt="-o")

        fig.canvas.draw()
        handle.update(fig)

    return progress, times
