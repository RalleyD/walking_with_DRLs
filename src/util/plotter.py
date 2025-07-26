import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from pathlib import Path
from moviepy import ImageSequenceClip
from datetime import datetime

FILE_DIR = Path(__file__).parent
PRJ_ROOT = Path(FILE_DIR).parent.parent


def learning_rate_ma(x: Optional[np.array], y: np.array, title: Optional[str] = "Reinforce learning curve"):


def learning_rate_ma(x: Optional[np.array], y: np.array,
                     target_ep: int = None,
                     convergence_ep: int = None,
                     title: Optional[str] = "Reinforce learning curve"):

    if x is None:
        x = np.arange(0, len(y), 1)

    fig, ax = plt.subplots(1, 1)
    ax.plot(x, y, label="raw returns")

    # target first reached
    if target_ep:
        ax.axvline(target_ep, color='r', label="target first reached")

    # convergence point
    if convergence_ep:
        ax.axvline(convergence_ep, color='g', label="Stable convergence point")

    # SMA
    window = len(y) // 10
    weights = np.ones(window) / window
    sma = np.convolve(y, weights, mode='valid')
    sma_x = np.arange(window, len(sma)+window, 1)

    ax.plot(sma_x, sma, label=f"moving average, {window=}")


    ax.grid(visible=True)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("episode (n)")
    ax.set_ylabel("returns (n)")

    output_dir = PRJ_ROOT / "plots"
    output_dir.mkdir(exist_ok=True)
    out_file = title.replace(" ", "-").replace(",",
                                               "_").replace(":", "") + datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".png"

    fig.savefig(output_dir / out_file, dpi=600)

    plt.show()


def record_gif(gif_frames: list, fps=200, filename: str = "reinforce", epochs: int = None) -> None:

    out_dir = PRJ_ROOT / "recordings"
    epochs_str = str(epochs) + "epochs_" if epochs else ""
    file = filename + '-' + epochs_str + \
        datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".gif"
    out_dir.mkdir(exist_ok=True)

    clip = ImageSequenceClip(gif_frames, fps=fps)
    clip.write_gif(out_dir / file, fps=fps)
