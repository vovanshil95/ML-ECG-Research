import torch

from model import model
from config import max_epoch, batch_size, learning_rate, thin_out_ratio, evaluation_freq, save_model
from loader import load_pairs


def evaluate(entries, criterion):
    batches = len(entries) // batch_size
    loss = 0
    model.eval()
    with torch.no_grad():
        for (signal1, signal2, real_out) in load_pairs(entries, batch_size):
            output = model(torch.Tensor(signal1), torch.Tensor(signal2))
            loss += criterion(output, torch.Tensor(real_out))
    model.train()
    return loss / batches


def train_pairs(train_data, eval_data):
    pairs = train_data
    batches = int(len(pairs) * thin_out_ratio) // batch_size
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(max_epoch):
        for i, (signal1, signal2, real_out) in enumerate(load_pairs(pairs, batch_size)):
            optimizer.zero_grad()
            output = model(torch.Tensor(signal1), torch.Tensor(signal2))
            loss = criterion(output, torch.Tensor(real_out))
            loss.backward()
            optimizer.step()
            print(f"iteration {batches * epoch + i + 1} of {batches * max_epoch}")
            if i % (batches // evaluation_freq) == 0 or i == batches - 1:
                eval_loss = evaluate(eval_data, criterion)
                print(f"loss on iteration {i}: {eval_loss}")
                if save_model:
                    torch.save(model.state_dict(), f'result/trained/trained{i // (batches // evaluation_freq)}')

                if eval_loss < 0.08:
                    return model
