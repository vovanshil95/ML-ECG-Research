from random import shuffle

import numpy as np


def load(entries, batch_size):

    for i in range(len(entries) // batch_size):
        batch = entries[i * batch_size: (i + 1) * batch_size]
        yield batch


def load_pairs(pairs, batch_size):

    shuffle(pairs)

    for i in range(len(pairs) // batch_size):
        slice_ = pairs[i * batch_size: (i + 1) * batch_size]
        batch_x = np.transpose(np.array(list(map(lambda pair: pair[:2], slice_))), axes=(1, 0, 2, 3))
        batch_y = list(map(lambda pair: pair[2], slice_))

        yield [batch_x[0], batch_x[1], batch_y]
