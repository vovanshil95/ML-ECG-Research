import torch
from tqdm import tqdm

import sys

from model import model
from config import max_epoch, batch_size, learning_rate, min_eval_loss, evaluation_freq, save_model, max_iterations
from loader import load_pairs


def evaluate(entries, criterion):
    batches = len(entries) // batch_size
    loss = 0
    model.eval()
    with torch.no_grad():
        for (signal1, signal2, real_out) in load_pairs(entries, batch_size):
            output = model(torch.Tensor(signal1), torch.Tensor(signal2))
            try:
                loss += criterion(output, torch.Tensor(real_out))
            except RuntimeError:
                loss += criterion(output * 0.999 + 0.0001, torch.Tensor(real_out))
    model.train()
    return loss / batches


def pairs_train(train_data, eval_data):
    if max_iterations is None:
        max_iterations_ = sys.maxsize
    else:
        max_iterations_ = max_iterations

    pairs = train_data
    batches = int(len(pairs)) // batch_size
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    with tqdm(total=min(batches * max_epoch, max_iterations_)) as pbar:
        for epoch in range(max_epoch):
            for i, (signal1, signal2, real_out) in enumerate(load_pairs(pairs, batch_size)):
                optimizer.zero_grad()
                output = model(torch.Tensor(signal1), torch.Tensor(signal2))
                try:
                    loss = criterion(output, torch.Tensor(real_out))
                except RuntimeError:
                    loss = criterion(output * 0.999 + 0.0001, torch.Tensor(real_out))
                loss.backward()
                optimizer.step()
                pbar.update()
                if i % (batches // evaluation_freq) == 0 or i == batches - 1:
                    eval_loss = evaluate(eval_data, criterion)
                    pbar.write(f"loss after {i + 1} iterations: {eval_loss}", )
                    if save_model:
                        torch.save(model.state_dict(), f'result/trained/trained{i // (batches // evaluation_freq)}')

                    if eval_loss < min_eval_loss:
                        return model

                if epoch * batches + i == max_iterations_:
                    return model
