from random import sample, shuffle

import numpy as np


def load(entries, batch_size, thin_out_ratio=None):

    if thin_out_ratio is None:
        thin_out_ratio = 1

    entries = sample(entries, int(len(entries) * thin_out_ratio))

    batches = []

    for i in range(len(entries) // batch_size):
        batch = entries[i * batch_size: (i + 1) * batch_size]
        batches.append(batch)

    return batches


def load_pairs(pairs, batch_size, thin_out_ratio=None):

    if thin_out_ratio is None:
        thin_out_ratio = 1

    pairs = sample(pairs, int(len(pairs) * thin_out_ratio))
    shuffle(pairs)

    batches = []

    for i in range(len(pairs) // batch_size):
        batch = []
        all_batch = pairs[i * batch_size: (i + 1) * batch_size]
        batch.append(np.array([el[0] for el in all_batch]))
        batch.append(np.array([el[1] for el in all_batch]))
        batch.append(np.array([el[2] for el in all_batch]))
        batches.append(batch)

    return batches
