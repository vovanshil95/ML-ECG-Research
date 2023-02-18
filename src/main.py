from random import shuffle

import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from sklearn.model_selection import train_test_split

from preprocessing import prepare_data
from config import valid_size, shot_size, test_size
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

    normal_shot = torch.Tensor(np.array(list(map(lambda el: el[0], filter(lambda el: el[1] == 0, data)))[:shot_size]))
    tachycardic_shot = torch.Tensor(np.array(list(map(lambda el: el[0], filter(lambda el: el[1] == 1, data)))[:shot_size]))
    bradycardic_shot = torch.Tensor(np.array(list(map(lambda el: el[0], filter(lambda el: el[1] == 2, data)))[:shot_size]))

    model.eval()
    normal_features = torch.mean(model.forward_once(normal_shot), dim=0)
    tachycardic_features = torch.mean(model.forward_once(tachycardic_shot), dim=0)
    bradycardic_features = torch.mean(model.forward_once(bradycardic_shot), dim=0)

    trues = 0
    outputs = []

    for entry in data:
        entry_features = model.forward_once(torch.unsqueeze(torch.tensor(entry[0]).float(), dim=0))
        features = (normal_features, tachycardic_features, bradycardic_features)
        most_similar = np.argmax(list(map(lambda f1, f2: cosine_similarity(f1, f2).detach().numpy(), \
                features, [entry_features] * 3)))
        outputs.append(most_similar)
        if entry[1] == most_similar:
            trues += 1

    print('accuracy on test data:', trues / len(data))



def main():
    entries = []
    signals = prepare_data()

    train_eval_data, test_data = train_test_split(signals, test_size=test_size)

    for sig in train_eval_data:
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
    few_shot(test_data, shot_size)




if __name__ == '__main__':
    main()
