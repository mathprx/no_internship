import torch
import torch.nn as nn
import sys
import os

# Get the parent directory and add it to the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from no_layers import FNOLayer, ChannelMLPLayer


class FNO(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 modes,
                 num_layers,
                 lifting_layers,
                 projection_layers,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.modes = modes
        self.num_layers = num_layers
        self.lifting_layers = lifting_layers
        self.projection_layers = projection_layers

        self.lifting = ChannelMLPLayer(in_channels, hidden_channels, lifting_layers)
        self.fno_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.fno_layers.append(FNOLayer(hidden_channels, hidden_channels, modes))
        self.projection = ChannelMLPLayer(hidden_channels, out_channels, projection_layers)
    
    def forward(self, x):
        """ x: (batch, in_channels, size) """
        x = self.lifting(x) # (batch, hidden_channels, size)
        for layer in self.fno_layers:
            x = layer(x)
        x = self.projection(x)
        return x # (batch, out_channels, size)


if __name__ == "__main__":
    fno = FNO(10, 20, 30, 40, 50, 60, 70)
    x = torch.randn(32, 10, 100)
    y = fno(x)
    print(y.shape) # (32, 20, 100)