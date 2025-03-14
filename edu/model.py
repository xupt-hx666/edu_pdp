import torch.nn as nn
from args import args_parser

args = args_parser()


class EducationModel(nn.Module):
    def __init__(self, name):
        super(EducationModel, self).__init__()
        self.name = name

        # 基础层（共享部分）
        self.base_layers = nn.Sequential(
            nn.Linear(args.input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2))

        # 个性化层（客户端独立）
        self.personal_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, args.num_classes))

    def forward(self, x):
        x = x.view(-1, args.input_dim)
        x = self.base_layers(x)
        return self.personal_layers(x)

