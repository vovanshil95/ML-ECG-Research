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


def load_pairs(pairs, batch_size, thin_out_ratio=1):

    pairs = sample(pairs, int(len(pairs) * thin_out_ratio))
    shuffle(pairs)

    for i in range(len(pairs) // batch_size):
        slice_ = pairs[i * batch_size: (i + 1) * batch_size]
        batch_x = np.transpose(np.array(list(map(lambda pair: pair[:2], slice_))), axes=(1, 0, 2, 3))
        batch_y = list(map(lambda pair: pair[2], slice_))

        yield [batch_x[0], batch_x[1], batch_y]
