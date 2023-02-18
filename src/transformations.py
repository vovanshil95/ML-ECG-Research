from random import uniform

from scipy import signal
import numpy as np
from math import pi

SLOW_HEART_RATE = (0.5, 0.7)
FAST_HEART_RATE = (1.3, 1.5)


def change_frequency(sig, times):
    _newSig = np.zeros(sig.shape)
    margin = abs(sig.shape[1] - int(sig.shape[1] / times)) // 2
    for channel in range(sig.shape[0]):
        if times >= 1:
            _newChannel = signal.resample(sig[channel], int(sig.shape[1] / times))
            _newChannel = np.concatenate([np.zeros(margin), _newChannel, np.zeros(margin + 1)])[:sig.shape[1]]
        else:
            _newChannel = signal.resample(sig[channel], int(sig.shape[1] / times))[margin:-margin]
            _newChannel = _newChannel[:sig.shape[1]]

        _newSig[channel] = _newChannel

    return _newSig


def add_wave_noise(sig, periods, amplitude=None):

    channels = sig.shape[0]
    _newSig = np.zeros(sig.shape)

    if amplitude is None:
        amplitude = 1

    noise = np.sin(np.linspace(uniform(0, 2*pi), 2*pi*periods, sig.shape[1])) * amplitude

    for channel in range(channels):
        _newSig[channel] = sig[channel] + noise

    return _newSig


def add_gauss_noise(sig, deviation):
    noise = np.random.normal(0, deviation, sig.shape)
    return sig + noise


def tachycardia(sig):
    heart_rate = uniform(*FAST_HEART_RATE)
    return change_frequency(sig, heart_rate)


def bradycardia(sig):
    heart_rate = uniform(*SLOW_HEART_RATE)
    return change_frequency(sig, heart_rate)
