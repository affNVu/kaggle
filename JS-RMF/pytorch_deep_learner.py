import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=78,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=16),
            nn.ReLU(),
            nn.Identity(16),
            )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=16,out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32,out_features=78),
            nn.ReLU(),
            nn.Identity(78)
        )
        
    def forward(self, x):
        encoded_x = self.encoder(x)
        decoded_x = self.decoder(encoded_x)
        return decoded_x

    def encode(self, x):
        return self.encoder(x)
        
    def decode(self, x):
        return self.decoder(x)
    
 