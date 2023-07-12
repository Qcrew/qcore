""" """

from inspect import isfunction

import numpy as np


def mag(data):
    """absolute value of two inputs x and y"""
    x, y = data
    return np.sqrt(x**2 + y**2)


def phase(data, freq, delay=0):
    """ """
    if freq is None:  # freq-dependent phase, freq got from incoming data at index = 0
        x, y, freq = data
    else:  # constant frequency calculation, freq to be passed in by the user
        x, y = data
    return np.angle(np.exp(1j * 2 * np.pi * delay * freq) * (x + 1j * y))


def fft(data, length=1):
    """ """
    data = data - np.average(data, axis=1)[:, None]
    fft_data = np.abs(np.fft.fft(data)) / length
    return fft_data[: int(length / 2 + 1)]


DATAFN_MAP = {
    k: v for k, v in locals().items() if not k == "isfunction" and isfunction(v)
}
