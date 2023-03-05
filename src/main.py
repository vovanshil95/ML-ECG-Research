import torch

from preprocessing import prepare_data
from train import pairs_train
from utils import few_shot
from results import get_embeddings, plot_embeddings
from config import load_model, load_from
from model import model

def main():

    if load_model:
        test_signals = prepare_data()
        model.load_state_dict(torch.load(load_from))
    else:
        train_pairs, eval_pairs, test_signals = prepare_data(to_pairs=True)
        pairs_train(train_pairs, eval_pairs)

    few_shot(test_signals)

    embaddings = get_embeddings(list(map(lambda sig: sig[0], test_signals)))
    plot_embeddings(embaddings, list(map(lambda sig: sig[1], test_signals)), 'infarct_embaddings')

if __name__ == '__main__':
    main()
