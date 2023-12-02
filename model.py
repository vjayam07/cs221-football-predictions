import torch
from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer = nn.Linear(input_size, 1)

    def forward(self, input):
        output = torch.sigmoid(self.layer(input))
        return output

# class MLP(nn.Module):
#     def __init__(self, input_size):
#         super(MLP, self).__init__
#         self.layer = nn.Linear(input_size, 1)

#     def forward(self, input):
#         pass

class NaiveBayes(nn.Module):
    def __init__(self, input_size):
        pass

    def forward(self, input):
        pass