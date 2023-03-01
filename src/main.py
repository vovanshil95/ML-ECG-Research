from preprocessing import prepare_data
from train import pairs_train
from utils import few_shot

def main():
    train_pairs, eval_pairs, test_signals = prepare_data(to_pairs=True)
    pairs_train(train_pairs, eval_pairs)
    few_shot(test_signals)

if __name__ == '__main__':
    main()
