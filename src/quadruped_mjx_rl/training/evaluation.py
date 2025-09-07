from datetime import datetime

import matplotlib.pyplot as plt
from IPython.display import display


def make_progress_fn(
    num_timesteps: float,
    reward_max: int = 40,
    title: str = "Evaluation results",
    run_in_cell: bool = True,
    data_key: str = "eval/episode_reward",
    data_err_key: str | None = "eval/episode_reward_std",
    label_key: str = "reward",
    color: str = "blue",
):
    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]
    max_y, min_y = reward_max, 0

    fig, ax = plt.subplots()
    if run_in_cell:
        handle = display(fig, display_id=True)
        plt.close(fig)

    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics[data_key])
        ydataerr.append(metrics.get(data_err_key, 0))

        ax.clear()
        ax.set_xlim(0, num_timesteps * 1.25)
        ax.set_ylim(min_y, max_y)
        ax.set_xlabel("# environment steps")
        ax.set_ylabel(f"{label_key} per episode")
        ax.set_title(f"{title} - current {label_key} per episode: {y_data[-1]:.3f}")
        ax.errorbar(x_data, y_data, yerr=ydataerr, color=color)

        if run_in_cell:
            handle.update(fig)
        else:
            fig.show()

    return progress, times
