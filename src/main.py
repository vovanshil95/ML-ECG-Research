from random import sample

from sklearn.model_selection import train_test_split

from preprocessing import prepare_data
from config import valid_size, shot_sizes, test_size, thin_out_ratio, data_path
from transformations import tachycardia, bradycardia
from train import train_pairs


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
