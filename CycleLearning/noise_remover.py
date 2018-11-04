import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, ifft

size = 100


def fourier_visualisation(noisy, start_index):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    for i in range(start_index, start_index + size):
        y = noisy[i:i + size]
        freqs = fftfreq(size, 0.1)
        fft_vals = fft(y)
        ax2.plot(np.array(freqs), np.array(fft_vals), zs=i, zdir='y')
        # removes noise
        fft_vals_nobegin = fft_vals[20:]
        highest_amp = fft_vals[np.argmax(fft_vals_nobegin)]
        threshold = 5 * highest_amp  # (highest_amp /2)
        fft_vals[abs(fft_vals) < threshold] = 0
        ax.plot(np.array(freqs), np.array(fft_vals), zs=i, zdir='y')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(-1000, 1000)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_zlim(-1000, 1000)
    angle = 90
    ax.view_init(0, angle)
    ax2.view_init(0, angle)
    plt.show()
