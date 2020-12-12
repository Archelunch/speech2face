import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, in_features=256, out_features=384*2*2):
        super(Decoder, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_features, in_features*16),
                                    nn.ReLU(),
                                    nn.Linear(in_features*16, out_features),
                                    nn.Tanh()
                                    )

    def forward(self, x): 
        b_size, l = x.shape
        return self.layer1(x).reshape(b_size, 384, 2, 2)
