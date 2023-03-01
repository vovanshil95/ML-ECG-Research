import torch.nn as nn
import torch.nn.functional as f


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.c1 = nn.Conv1d(in_channels=2, out_channels=128, kernel_size=7, stride=2)
        self.c2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5, stride=2)
        self.c3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(in_features=576, out_features=128)
        self.rel = nn.CosineSimilarity(dim=1)

    def forward_once(self, x):
        x = f.max_pool1d(f.relu(self.c1(x)), 4)
        x = f.max_pool1d(f.relu(self.c2(x)), 4)
        x = f.max_pool1d(f.relu(self.c3(x)), 4)

        x = x.view(x.shape[0], x.shape[1] * x.shape[2])

        x = f.relu(self.fc1(x))

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        output = self.rel(output1, output2)

        return output


model = SiameseNet()
