import os.path
from random import shuffle, sample

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from preprocessing import prepare_data
from config import valid_size, shot_sizes, test_size, thin_out_ratio, data_path
from transformations import add_wave_noise, add_gauss_noise, tachycardia, bradycardia
from train import train_pairs
from model import model

NORMAL_DEVIATION = 0.01
WAVE_PERIODS = 5
WAVE_AMPLITUDE = 0.5


def make_twins(signal, transforms):
    signal1 = add_wave_noise(add_gauss_noise(signal, 0.01), WAVE_PERIODS, WAVE_AMPLITUDE)
    signal2 = add_wave_noise(add_gauss_noise(signal, 0.01), WAVE_PERIODS, WAVE_AMPLITUDE)

    signal1 = transforms[0](signal1) if transforms[0] else signal1
    signal2 = transforms[1](signal2) if transforms[1] else signal2

    return [signal1, signal2, transforms[0] == transforms[1]]


def make_test_data(signals):
    test_data = []
    for signal in signals:
        normal = add_wave_noise(add_gauss_noise(signal, 0.01), WAVE_PERIODS, WAVE_AMPLITUDE)
        tachycardic = tachycardia(signal)
        bradycardic = bradycardia(signal)
        test_data.extend([(normal, 0), (tachycardic, 1), (bradycardic, 2)])

    shuffle(test_data)

    return test_data

def few_shot(data, shot_size):

    result_table = [[]]
    classes = ['normal', 'tachycardia', 'bradycardia']
    result_table[0].append('shot_size')
    result_table[0].extend(['precision_' + class_ for class_ in classes])
    result_table[0].extend(['recall_' + class_ for class_ in classes])
    result_table[0].append('accuracy')

    for shot_size in shot_sizes:
        normal_shot = torch.Tensor(np.array(list(map(lambda el: el[0], filter(lambda el: el[1] == 0, data)))[:shot_size]))
        tachycardic_shot = torch.Tensor(np.array(list(map(lambda el: el[0], filter(lambda el: el[1] == 1, data)))[:shot_size]))
        bradycardic_shot = torch.Tensor(np.array(list(map(lambda el: el[0], filter(lambda el: el[1] == 2, data)))[:shot_size]))

        model.eval()
        normal_features = torch.mean(model.forward_once(normal_shot), dim=0)
        tachycardic_features = torch.mean(model.forward_once(tachycardic_shot), dim=0)
        bradycardic_features = torch.mean(model.forward_once(bradycardic_shot), dim=0)

        model_out = []
        real_out = []

        for entry in data:
            entry_features = model.forward_once(torch.unsqueeze(torch.tensor(entry[0]).float(), dim=0))
            features = (normal_features, tachycardic_features, bradycardic_features)
            most_similar = np.argmax(list(map(lambda f1, f2: cosine_similarity(f1, f2).detach().numpy(), \
                    features, [entry_features] * 3)))
            model_out.append(most_similar)
            real_out.append(entry[1])

        result_table.append([shot_size,
                             *precision_score(real_out, model_out, average=None),
                             *recall_score(real_out, model_out, average=None),
                             accuracy_score(real_out, model_out)])

    df = pd.DataFrame(result_table[1:], columns=result_table[0])
    print(df)

    if not os.path.exists('../result/'):
        os.makedirs('../result')
    df.to_csv('../result/result-values.csv')



def main():
    entries = []
    signals = prepare_data()
    signals = sample(signals, int(len(signals) * thin_out_ratio))

    train_eval_data, test_data = train_test_split(signals, test_size=test_size)

    for sig in train_eval_data:
        print(i := i + 1 if 'i' in dir() else 1, 'of', len(train_eval_data))
        normal_normal = make_twins(sig, (None, None))
        fast_fast = make_twins(sig, (tachycardia, tachycardia))
        slow_slow = make_twins(sig, (bradycardia, bradycardia))
        normal_fast = make_twins(sig, (None, tachycardia))
        normal_slow = make_twins(sig, (None, bradycardia))
        slow_fast = make_twins(sig, (bradycardia, tachycardia))
        entries.extend([normal_normal, fast_fast, slow_slow, normal_fast, normal_slow, slow_fast])

    train_entries, eval_entries = train_test_split(entries, test_size=valid_size)
    train_pairs(train_entries, eval_entries)

    test_data = make_test_data(test_data)
    few_shot(test_data, shot_sizes)




if __name__ == '__main__':
    main()
