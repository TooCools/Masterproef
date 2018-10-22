import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, ifft

size = 100
start_index = 2000


# def fourier(noisy, actual):
#     y = noisy[start_index:start_index + size]
#     freqs = fftfreq(size, 0.115)
#     fft_vals = fft(y)
#     fft_theo = 2 * np.abs(fft_vals / size)
#     mask = freqs > 0
#     actual_freqs = freqs[mask]
#     actual_fft = fft_theo[mask]
#     plt.figure(1)
#     plt.title='Noisy'
#     plt.plot(actual_freqs, actual_fft)
#     highest_amp = actual_fft[np.argmax(actual_fft)]
#     threshold = (highest_amp / 10)
#     actual_fft[actual_fft < threshold] = 0
#     plt.figure(2)
#     plt.title='Removed noise'
#     plt.plot(actual_freqs, actual_fft)
#     filtered = ifft(actual_fft)
#     plt.figure(3)
#     plt.title='Denoised'
#     plt.plot(range(1,50), filtered)
#     # plt.plot(range(1,50), actual)
#     plt.show()
#     print('peek', actual_fft[np.argmax(actual_fft)])


def fourier(noisy, actual):
    y = noisy[start_index:start_index + size]
    freqs = fftfreq(size, 0.115)
    fft_vals = fft(y)
    plt.figure(1)
    plt.title = 'Noisy'
    plt.plot(freqs, fft_vals, label='Noisy')
    plt.legend()
    fft_vals_nobegin = fft_vals[20:]
    highest_amp = fft_vals[np.argmax(fft_vals_nobegin)]
    threshold = 5 * highest_amp  # (highest_amp /2)
    fft_vals[abs(fft_vals) < threshold] = 0
    plt.figure(2)
    plt.title = 'Removed noise'
    plt.plot(freqs, fft_vals, label='Removed noise')
    plt.legend()
    filtered = ifft(fft_vals)
    plt.figure(3)
    plt.title = 'Filtered vs actual'
    plt.plot(range(0, 100), filtered, label="filtered")
    plt.legend()
    a = actual[start_index:start_index + size]
    plt.plot(range(0, 100), a, label="actual (no noise)")
    plt.legend()
    plt.figure(4)
    plt.title = 'Filtered vs noisy'
    plt.plot(range(0, 100), filtered, label="filtered")
    plt.legend()
    b = noisy[start_index:start_index + size]
    plt.plot(range(0, 100), b, label="actual (noisy)")
    plt.legend()
    plt.show()
