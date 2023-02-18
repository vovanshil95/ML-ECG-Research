from statistics import stdev, mean
from matplotlib import pyplot as plt
import wfdb
import numpy as np

path = 'data/a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0/WFDBRecords/01/010/JS00010'

signal, _ = wfdb.rdsamp(path)

signal = signal.T


def get_spaces(peak_indexes):

    spaces = []
    for i, _ in enumerate(peak_indexes[:-1]):
        spaces.append(peak_indexes[i + 1] - peak_indexes[i])

    return spaces


def filter_peaks(peak_indexes):

    spaces = get_spaces(peak_indexes)
    avg_space = mean(spaces)

    wrong_indexes = []
    for i, _ in enumerate(peak_indexes[:-1]):
        if abs(peak_indexes[i] - peak_indexes[i + 1]) < avg_space / 5:
            wrong_indexes.append(i + 1)

    peak_indexes = [peak for i, peak in enumerate(peak_indexes) if i not in wrong_indexes]

    return peak_indexes


def split_qrs(sig: list) -> list:

    ref_signal = sig[0]

    min_peaks_size = 5
    peaks = sorted(ref_signal)[-min_peaks_size:]
    mean_peak = mean(peaks)
    deviation = stdev(ref_signal)
    peak_indexes = []

    for i, sig in enumerate(ref_signal[:-1]):
        if sig >= ref_signal[i-1] and sig > ref_signal[i+1]:
            if abs(mean_peak - sig) < deviation * 2 or \
                    (len(peak_indexes) > 0 and abs(ref_signal[peak_indexes[-1]] - sig) < deviation / 2):
                peak_indexes.append(i)

    last_peak_indexes = None
    while peak_indexes != last_peak_indexes or last_peak_indexes is None:
        last_peak_indexes = peak_indexes.copy()
        peak_indexes = filter_peaks(peak_indexes)

    spaces = get_spaces(peak_indexes)
    new_peaks = []
    for i, _ in enumerate(peak_indexes[:-1]):
        mean_space = mean(spaces)
        space = abs(peak_indexes[i] - peak_indexes[i+1])
        if abs(mean_space * 2 - space) < abs(mean_space - space):
            new_peaks.append(np.argmax(ref_signal[peak_indexes[i]+10:peak_indexes[i+1]-10]) + peak_indexes[i]+10)

    peak_indexes.extend(new_peaks)
    mean_space = mean(get_spaces(peak_indexes))

    qrs_list_ = []

    for peak in peak_indexes:
        qrs_list_.append(ref_signal[peak-int(mean_space / 2):peak+int(mean_space / 2)])

    plt.plot(qrs_list_[1])
    plt.show()

    return qrs_list_


qrs_list = split_qrs(signal)
