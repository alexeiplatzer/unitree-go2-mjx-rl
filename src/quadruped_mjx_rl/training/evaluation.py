from datetime import datetime

import matplotlib.pyplot as plt
from IPython.display import display


def make_progress_fn(
    *,
    run_in_cell: bool = True,
    title: str = "Evaluation results",
    color: str = "blue",
    num_timesteps: float = 1_000_000,
    label_key: str = "reward",
    data_key: str = "eval/episode_reward",
    data_err_key: str | None = "eval/episode_reward_std",
    data_max: int = 40,
    data_min: int = -10,
):
    """Returns a progress function that plots the chosen metric values over time.
    Args:
        run_in_cell: whether to update the plot dynamically in a jupyter cell display a new
            plot every time using native matplotlib display functionality.
        title: the title of the plot.
        color: the color of the data line.
        num_timesteps: the total number of timesteps over which the data is plotted.
        label_key: a string describing what is being plotted, used to name the y-axis.
        data_key: the string key of the metric value to plot in the metrics dictionary.
        data_err_key: the string key in the metric dict describing the error/deviation of the
            plotted value.
        data_max: the maximum y-axis value which plots the metric data.
        data_min: correspondingly the minimum of the y-axis.
    """
    x_data = []
    y_data = []
    y_data_err = []
    times = [datetime.now()]

    fig, ax = plt.subplots()
    if run_in_cell:
        handle = display(fig, display_id=True)
        plt.close(fig)

    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics[data_key])
        y_data_err.append(metrics.get(data_err_key, 0))

        ax.clear()
        ax.set_xlim(0, num_timesteps * 1.25)
        ax.set_ylim(data_min, data_max)
        ax.set_xlabel("# environment steps")
        ax.set_ylabel(f"{label_key} per episode")
        ax.set_title(f"{title} - current {label_key} per episode: {y_data[-1]:.3f}")
        ax.errorbar(x_data, y_data, yerr=y_data_err, color=color)

        if run_in_cell:
            handle.update(fig)
        else:
            fig.show()

    return progress, times
