
import matplotlib.pyplot as plt
import numpy as np


def plot_amplitude_map(A_max: np.ndarray, cmap: str, amplitude: float, omega: float | str) -> None:
    plt.imshow(A_max, cmap=cmap)
    plt.colorbar()
    plt.title(f"(A = {amplitude}; Omega = {omega})\nMax amplitude map")
    plt.show()