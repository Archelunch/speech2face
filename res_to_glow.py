import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, in_features=256, out_features=3*128*128):
        self.layer1 = nn.Sequential(nn.Linear(in_features, in_features*16),
                                    nn.ReLU(),
                                    nn.Linear(in_features*16, out_features),
                                    nn.Tanh()
                                    )

    def forward(self, x):
        return self.layer1(x)
