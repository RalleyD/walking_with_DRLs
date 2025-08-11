import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional
from pathlib import Path
from moviepy import ImageSequenceClip
from datetime import datetime
from src.evaluate.performance_metrics import PerformanceMetrics
from src.custom_logger import CustomLogger

FILE_DIR = Path(__file__).parent
PRJ_ROOT = Path(FILE_DIR).parent.parent


########################################
# logger
########################################
logger = CustomLogger.get_project_logger()
########################################


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
                     title: Optional[str] = "Reinforce learning curve",
                     time_steps: int = int(1e6),
                     **kwargs):

    fig, ax = plt.subplots(1, 1)

    title = f"{title}: {', '.join(lyr + ':' + str(v) for lyr, v in kwargs.items())}"

    _lc_axis(x, y, target_ep, convergence_ep, title, ax, time_steps)

    save_plot(title, fig)

    plt.show()


def save_plot(title, fig):
    output_dir = PRJ_ROOT / "plots"
    output_dir.mkdir(exist_ok=True)
    out_file = title.replace(" ", "-").replace(",",
                                               "_").replace(":", "") + datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".png"

    fig.savefig(output_dir / out_file, dpi=600)


def _lc_axis(x, y, target_ep, convergence_ep, title, ax, time_steps: int):
    if x is None:
        x = np.arange(0, len(y), 1)

    ax.plot(x, y, label="average returns")

    # target first reached
    if target_ep:
        ax.axvline(target_ep, color='r', label="target first reached")

    # convergence point
    if convergence_ep:
        ax.axvline(convergence_ep, color='g', label="Stable convergence point")

    # SMA
    window = len(y) // 10
    if window > 0:
        weights = np.ones(window) / window
        sma = np.convolve(y, weights, mode='valid')
        window_scaled = int(window * (x[1] - x[0]))
        sma_x = x[window-1:]

        ax.plot(sma_x, sma, label=f"moving average, window={window_scaled}")
    else:
        logger.warning("SMA window too small: %d" % window)

    # TODO magic number - provide a consistent X axis for A/B comparison
    ax.set_xticks(np.arange(0, int(time_steps)+1, step=int(time_steps // 5)))
    ax.grid(visible=True)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("time steps (n)")
    ax.set_ylabel("returns (n)")


class PlotLearningCurve:
    """
    """

    def __init__(self, time_steps: int,
                 trials: int):
        """
        Args:
            time_steps (int): number of training time steps
            trials (int): number of training loop repititions
        """
        self._time_steps = time_steps
        self._trials = trials
        self._reinforce_x = np.array([])
        self._reinforce_y = np.array([])
        self._reinforce_y_std = np.array([])
        self._td3_x = np.array([])
        self._td3_y = np.array([])
        self._td3_y_std = np.array([])
        self._fig, self._ax = plt.subplots(1, 1)
        self._title = "Reinforce Vs TD3 Learning Curve"
        self._clrs = sns.color_palette("husl", 2)
        self._clr_idx = 0
        self._ax: plt.Axes = None

    def _set_model_attrs(self, model_name: str, x_data, y_data, y_std):
        setattr(self, f'_{model_name}_x', np.array(x_data))
        setattr(self, f'_{model_name}_y', np.array(y_data))
        setattr(self, f'_{model_name}_y_std', np.array(y_std))

    def _set_model_data(self, model_name: str, x_data, y_data, y_std):
        # valid models set
        valid_models = {"reinforce", "td3"}

        if model_name not in valid_models:
            raise ValueError(
                f"Incorrect Model Requested: {model_name} - models available {valid_models}")

        self._set_model_attrs(model_name.lower(), x_data, y_data, y_std)

    def _lc_axis(self, x, y, y_std, model_name: str):
        if x is None:
            x = np.arange(0, len(y), 1)

        print("length x: ", len(x))
        print("length y: ", len(y))
        print("length y std: ", len(y_std))
        print(f"{x=}")
        print(f"{y=}")
        print(f"{y_std=}")

        self._ax.plot(x, y, label=model_name,
                      c=self._clrs[self._clr_idx])

        if len(y_std):
            self._ax.fill_between(x, y-y_std, y+y_std,
                                  alpha=0.3,
                                  facecolor=self._clrs[self._clr_idx])

        self._clr_idx = self._clr_idx + 1 \
            if self._clr_idx < len(self._clrs) else 0

        # SMA
        # window = len(y) // 10
        # if window > 0:
        #     weights = np.ones(window) / window
        #     sma = np.convolve(y, weights, mode='valid')
        #     window_scaled = int(window * (x[1] - x[0]))
        #     sma_x = x[window-1:]

        #     ax.plot(sma_x, sma, label=f"moving average, window={window_scaled}")
        # else:
        #     logger.warning("SMA window too small: %d" % window)

        # TODO magic number - provide a consistent X axis for A/B comparison
        self._ax.set_xticks(
            np.arange(0, int(self._time_steps)+1,
                      step=int(self._time_steps // 5))
        )
        self._ax.grid(visible=True)
        self._ax.legend()
        self._ax.set_title(self._title)
        self._ax.set_xlabel("time steps (n)")
        self._ax.set_ylabel("returns (n)")

    def _plot_lc_with_confidence(self, alpha=0.5):
        if len(self._reinforce_y):
            y_reinforce_smoothed = self._smooth_curve(self._reinforce_y.copy())
            # call _lc_axis with x_data, smoothed y_data
            y_std_alpha = self._reinforce_y_std * alpha
            self._lc_axis(self._reinforce_x, y_reinforce_smoothed,
                          y_std_alpha, "reinforce")

        if len(self._td3_y):
            y_td3_smoothed = self._smooth_curve(self._td3_y)
            # call _lc_axis with x_data, y_std
            y_td3_std_alpha = self._td3_y_std * alpha
            self._lc_axis(self._td3_x, y_td3_smoothed, y_td3_std_alpha, "TD3")

    def _plot_lc_std(self):
        pass

    def _smooth_curve(self, y_data: np.ndarray, method='sma', window=100) -> np.ndarray:
        """
        Args:
            method (str): 'sma' -> simple moving average
        """
        # establish a ones array of size window
        # for equal weights divide each ones by the window size
        if len(y_data) <= window:
            window = int(len(y_data) // 10)

        weights = np.ones(window) / window
        # convolve the weights array with y_data
        sma = np.convolve(y_data, weights, mode='same')

        return sma

    def _save_plot(self, fig: plt.Figure):
        output_dir = PRJ_ROOT / "plots"
        output_dir.mkdir(exist_ok=True)
        out_file = self._title.replace(" ", "-").replace(
            ",", "_").replace(":", "") + \
            datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + \
            ".png"

        fig.savefig(output_dir / out_file, dpi=600)

    def set_reinforce_data(self, x_data, y_data, y_std):
        self._set_model_data("reinforce", x_data, y_data, y_std)

    def set_td3_data(self, x_data, y_data, y_std):
        self._set_model_data("td3", x_data, y_data, y_std)

    def plot_learning_curves(self):
        # instantiate fig and axis
        fig, self._ax = plt.subplots(1, 1)
        # plot_lc_with_confidence reinforce
        self._plot_lc_with_confidence()
        # save plot
        self._save_plot(fig)
        # show plot
        plt.show()

# def _plot_lc_with_confidence(ax, x_data, y_data, y_std, alpha=0.5, time_steps=int(1e6)):
#     """
#     Args:
#         alpha (float): how much of the std dev to preserve
#                         (for visual clarity)
#     """
#     # leverage _lc_axis function to do this
#     # 1 - smooth the curve
#     y_smoothed = _smooth_curve(y_data)
#     # 2 - call _lc_axis with x_data, smoothed y_data
#     _lc_axis(x_data, y_smoothed,
#              title="Reinforce Vs TD3 Learning Curve", time_steps=time_steps)
#     # 3 - call _lc_axis with x_data, y_std
#     _lc_axis(x_data, y_std, time_steps=time_steps)


# def plot_learning_curves(reinforce_x, reinforce_y, reinforce_std, td3_x, td3_y, td3_std):
#     # instantiate fig and axis
#     fig, ax = plt.subplots(1, 1)
#     # plot_lc_with_confidence reinforce
#     _plot_lc_with_confidence(ax, reinforce_x, reinforce_y, reinforce_std)
#     # repeat for td3
#     # save plot
#     # show plot


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
