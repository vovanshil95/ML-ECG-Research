import wfdb
import os
import requests
import zipfile
from random import shuffle

from config import data_path

DATASET_URL = 'https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip'
DATASET_NAME = 'mit-bih-arrhythmia-database-1.0.0'


def get_from_url():
    response = requests.get(DATASET_URL)
    open('download.zip', "wb").write(response.content)
    zip_ref = zipfile.ZipFile('download.zip', 'r')
    zip_ref.extractall('../' + data_path)
    os.remove('download.zip')


def split(signal, parts):
    channels = signal.shape[0]
    remainder = signal.shape[1] % parts
    signal = signal[:, :-remainder]
    signal = signal.reshape(parts, channels, signal.shape[1] // parts)
    return signal


def make_pairs(entries, pairs_key):
    pairs = []
    for i, el1 in enumerate(entries):
        for j, el2 in enumerate(entries[i:], start=i):
            pairs.append((el1, el2, pairs_key(el1, el2) if pairs_key else i == j))
    return pairs


def prepare_data(to_pairs=False, pairs_key=None):

    path = '../' + data_path + DATASET_NAME + '/'

    if not os.path.exists(path):
        get_from_url()

    signal_names = list(set(filter(lambda name: name.isnumeric(),
                                   map(lambda file_name: file_name[:3], os.listdir(path)))))
    shuffle(signal_names)

    segment_signals = []

    for signal_name in signal_names:
        signal, _ = wfdb.rdsamp(path + signal_name)

        signal = signal.T

        segmented_signal = split(signal, 15)
        segment_signals.extend(segmented_signal)

    if to_pairs:
        pairs = make_pairs(segment_signals, pairs_key)
        return pairs
    else:
        return segment_signals
