import pandas as pd
import wfdb
import os
import requests
import zipfile

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from config import data_path, dataset_url, dataset_name, test_size, valid_size, thin_out_ratio


def get_from_url():
    r = requests.get(dataset_url, stream=True)
    with open('../data/download.zip', 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        chunk_size = 1024
        with tqdm(total=total_length // chunk_size) as pbar:
            pbar.set_description('Dataset download')
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    pbar.update()

    zip_ref = zipfile.ZipFile('../data/download.zip', 'r')
    dataset_name_ = zip_ref.filelist[0].filename.split(os.sep)[0]
    zip_ref.extractall('../' + data_path)
    os.remove('../data/download.zip')

    return dataset_name_


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
            pairs.append((el1[0], el2[0], pairs_key(el1, el2) if pairs_key else i == j))
    return pairs


def prepare_data(to_pairs=False):

    dataset_name_ = get_from_url() if dataset_name is None else dataset_name

    path = data_path + dataset_name_ + '/'
    assert os.path.exists(path)

    df = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
    infarct_paths = path + df[df.scp_codes.str.contains('IMI')].filename_hr
    infarct_paths = infarct_paths.sample(int(infarct_paths.size * thin_out_ratio))
    normal_paths = path + df[df.scp_codes.str.contains('NORM')].filename_hr.sample(infarct_paths.size)

    signals = []
    sig_names = wfdb.rdsamp(normal_paths.iloc[0])[1]['sig_name']
    channels = [sig_names.index('V1'), sig_names.index('II')]

    for i in tqdm(range(infarct_paths.size)):
        signals.append((wfdb.rdsamp(normal_paths.iloc[i], channels=channels)[0].T, 0))
        signals.append((wfdb.rdsamp(infarct_paths.iloc[i], channels=channels)[0].T, 1))

    train_valid_signals, test_signals = train_test_split(signals, test_size=test_size)
    train_signals, valid_signals = train_test_split(train_valid_signals, test_size=valid_size/(1-test_size))

    if to_pairs:
        train_pairs = make_pairs(train_signals, lambda sig1, sig2: sig1[1] == sig2[1])
        valid_pairs = make_pairs(valid_signals, lambda sig1, sig2: sig1[1] == sig2[1])
        return train_pairs, valid_pairs, test_signals
    else:
        return signals
