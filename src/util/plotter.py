import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from pathlib import Path
from moviepy import ImageSequenceClip
from datetime import datetime
from src.evaluate.performance_metrics import PerformanceMetrics

FILE_DIR = Path(__file__).parent
PRJ_ROOT = Path(FILE_DIR).parent.parent


"""create wrapper function that will:
    take a dict of model: metrics
    create a subplot column for each model
    if each model has variance coeffecient data, create a third column
    rows corresponding to the number of metrics in each model key
    plot learning rate with moving average and convergence vlines
    plot exploration rate below each model's learning curve graph
    plot the rolling variance below
    in the middle column, top row, in between both models
    plot both model's variance coefficient"""


def evaluation_figure_a_b(metrics_a: PerformanceMetrics,
                          model_a_name: str = "REINFORCE",
                          metrics_b: PerformanceMetrics = None,
                          model_b_name: str = "TD3",
                          title="A-B training comparison:"):
    rows = 3
    cols = 2

    # combine the right-most subplots into one

    fig, axs = plt.subplots(rows, cols, figsize=(14, 9))
    gs = axs[2, 0].get_gridspec()
    # a/b multi-plot row
    for ax in axs[-1, 0:]:
        ax.remove()
    ax_ab = fig.add_subplot(gs[-1, 0:])
    ax_ab.grid(visible=True)
    ax_ab.legend()

    for col, (model_metrics, name) in enumerate([(metrics_a, model_a_name),
                                                (metrics_b, model_b_name)]):

        # TODO refactor as inner loop
        if model_metrics is None:
            continue
        # plot learning curves
        convergence = model_metrics.stable_convergence_time()
        y = model_metrics._episode_returns
        _lc_axis(x=None, y=y, target_ep=None, convergence_ep=convergence,
                 title=f"{name}: Learning Curve",
                 ax=axs[0, col])

    # TODO refactor as inner loop
    for col, (model_metrics, name) in enumerate([(metrics_a, model_a_name),
                                                (metrics_b, model_b_name)]):
        if model_metrics is None:
            continue
        # plot exploration rates
        exploration_sma = model_metrics.exploration_trend()
        x = np.arange(model_metrics._rolling_window_size,
                      model_metrics._rolling_window_size + len(exploration_sma))
        axs[1, col].plot(
            x, exploration_sma, label=f"rolling av window: {model_metrics._rolling_window_size}")
        axs[1, col].set_title(f"{name}: Exploration Rate")
        axs[1, col].legend()

    # TODO refactor as inner loop
    for (model_metrics, name) in [(metrics_a, model_a_name),
                                  (metrics_b, model_b_name)]:

        if model_metrics is None:
            continue
        # plot coefficient of variance
        _, var_coeff = model_metrics.get_stability_metrics()
        x = np.arange(model_metrics._rolling_window_size,
                      len(var_coeff) * model_metrics._rolling_window_size + 1,
                      100)
        print(f"{x=}")
        ax_ab.set_title(
            f"{name}: A/B Training stability - Normalised variance")
        ax_ab.plot(
            x, var_coeff, label=name)  # 'bo')
        ax_ab.legend()

    title_b_part = "vs " + model_b_name if model_b_name else ""
    title_a_part = f" {model_a_name} {title_b_part}"
    title = title + title_a_part

    save_plot(title, fig)

    plt.show()


def learning_rate_ma(x: Optional[np.array], y: np.array,
                     target_ep: int = None,
                     convergence_ep: int = None,
                     title: Optional[str] = "Reinforce learning curve"):

    fig, ax = plt.subplots(1, 1)

    _lc_axis(x, y, target_ep, convergence_ep, title, ax)

    save_plot(title, fig)

    plt.show()


def save_plot(title, fig):
    output_dir = PRJ_ROOT / "plots"
    output_dir.mkdir(exist_ok=True)
    out_file = title.replace(" ", "-").replace(",",
                                               "_").replace(":", "") + datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".png"

    fig.savefig(output_dir / out_file, dpi=600)


def _lc_axis(x, y, target_ep, convergence_ep, title, ax):
    if x is None:
        x = np.arange(0, len(y), 1)

    ax.plot(x, y, label="returns")

    # target first reached
    if target_ep:
        ax.axvline(target_ep, color='r', label="target first reached")

    # convergence point
    if convergence_ep:
        ax.axvline(convergence_ep, color='g', label="Stable convergence point")

    # SMA
    window = len(y) // 10
    if window == 0:
        raise ValueError("SMA window too small: %d" % window)
    weights = np.ones(window) / window
    sma = np.convolve(y, weights, mode='valid')
    window_scaled = window * (x[1] - x[0])
    sma_x = x[window-1:]

    ax.plot(sma_x, sma, label=f"moving average, window={window_scaled}")

    # TODO magic number - provide a consistent X axis for A/B comparison
    ax.set_xticks(np.arange(0, int(1e6), step=int(0.2e6)))
    ax.grid(visible=True)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("episode (n)")
    ax.set_ylabel("returns (n)")


def record_gif(gif_frames: list, fps=200, filename: str = "reinforce", epochs: int = None) -> None:

    out_dir = PRJ_ROOT / "recordings"
    epochs_str = str(epochs) + "epochs_" if epochs else ""
    file = filename + '-' + epochs_str + \
        datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".gif"
    out_dir.mkdir(exist_ok=True)

    clip = ImageSequenceClip(gif_frames, fps=fps)
    clip.write_gif(out_dir / file, fps=fps)


if __name__ == "__main__":
    model_a = PerformanceMetrics()
    model_b = PerformanceMetrics()
    evaluation_figure_a_b(model_a, metrics_b=model_b)
