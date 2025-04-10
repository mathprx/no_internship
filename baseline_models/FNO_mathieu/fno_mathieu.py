import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
from no_layers import SpectralConv1d
import modulus


device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class FNOMetaData(modulus.ModelMetaData):
    name: str = "fno_mathieu"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = False
    amp_gpu: bool = False

class FNO_mathieu(modulus.Module):
    def __init__(
            self,
            input_profile_num: int = 9, # number of input profile
            input_scalar_num: int = 17, # number of input scalars
            target_profile_num: int = 5, # number of target profile
            target_scalar_num: int = 8, # number of target scalars
            output_prune: bool = True, # whether or not we prune strato_lev_out levels
            strato_lev_out: int = 12, # number of levels to set to zero
            modes: int = 30, # number of Fourier modes to multiply, at most floor(N/2) + 1 where N=vertical_levels_num=60
            channel_dim: int = 93, # channel dimension
            out_network_dim: int = 128 # width of the output network
    ):
        super().__init__(meta=FNOMetaData())
        
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
        self.output_prune = output_prune
        self.strato_lev_out = strato_lev_out
        self.vertical_level_num = 60
        self.inputs_dim = input_profile_num * self.vertical_level_num + input_scalar_num
        self.targets_dim = target_profile_num * self.vertical_level_num + target_scalar_num
        

        self.modes1 = modes
        self.channel_dim = channel_dim
        self.out_network_dim = out_network_dim


        self.fc0 = nn.Linear(self.input_scalar_num + self.input_profile_num, self.channel_dim) 


        self.conv0 = SpectralConv1d(self.channel_dim, self.channel_dim, self.modes1)
        self.conv1 = SpectralConv1d(self.channel_dim, self.channel_dim, self.modes1)
        self.conv2 = SpectralConv1d(self.channel_dim, self.channel_dim, self.modes1)
        self.conv3 = SpectralConv1d(self.channel_dim, self.channel_dim, self.modes1)
        self.w0 = nn.Conv1d(self.channel_dim, self.channel_dim, 1)
        self.w1 = nn.Conv1d(self.channel_dim, self.channel_dim, 1)
        self.w2 = nn.Conv1d(self.channel_dim, self.channel_dim, 1)
        self.w3 = nn.Conv1d(self.channel_dim, self.channel_dim, 1)


        self.fc1 = nn.Linear(self.channel_dim, self.out_network_dim)
        self.fc2 = nn.Linear(self.out_network_dim, self.target_profile_num+self.target_scalar_num)


    def forward(self, x):
        '''
        x: (batch, input_profile_num*levels+input_scalar_num)
        '''

        # Set x : (batch, vertical_level, input_profile_num + input_scalar_num) by repeating scalar values
        x_scalar = x[:,None,self.input_profile_num*self.vertical_level_num:].repeat(1,self.vertical_level_num,1)
        x_profile = x[:,:self.input_profile_num*self.vertical_level_num].reshape(-1,self.input_profile_num,self.vertical_level_num).permute(0,2,1)
        x = torch.cat([x_profile, x_scalar], dim=2)


        # Lift to the channel dimension
        x = self.fc0(x)


        # Prepare input by permuting dimensions for Fourier layers
        x = x.permute(0, 2, 1)


        # First Fourier Layer
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)


        # Sacond Fourier Layer
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)


        # Third Fourier Layer
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)


        # Fourth Fourier Layer
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2


        # The part I still don't fully understand
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        # set back x :(batch_size, input_profile_num*levels+input_scalar_num)
        x3d = x[...,:self.target_profile_num].reshape(-1,self.target_profile_num*self.vertical_level_num)
        x2d = torch.mean(x[...,self.target_profile_num:], dim=1).reshape(-1,self.target_scalar_num)
        x = torch.cat([x3d, x2d], dim=-1)
        return x

