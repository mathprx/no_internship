import torch
import torch.nn as nn
import torch.nn.functional as F


class DataToFunction(nn.Module):
    def __init__(self, input_profile_num, input_scalar_num, vertical_levels):
        super().__init__()
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.vertical_levels = vertical_levels

    def forward(self, x):
        '''
        x: (batch, input_profile_num*vertical_levels + input_scalar_num)
        '''
        # Set x to (batch, input_profile_num + input_scalar_num, vertical_levels) by repeating scalar values
        x_profile = x[:, :self.input_profile_num * self.vertical_levels]
        x_profile = x_profile.reshape(-1, self.input_profile_num, self.vertical_levels)  # (batch, input_profile_num, vertical_levels)

        # Extract and repeat the scalar part along vertical_levels
        x_scalar = x[:, self.input_profile_num * self.vertical_levels:]
        x_scalar = x_scalar[:, :, None].repeat(1, 1, self.vertical_levels)  # (batch, input_scalar_num, vertical_levels)

        # Concatenate along the profile+scalar feature dimension
        x = torch.cat([x_profile, x_scalar], dim=1)  # (batch, input_profile_num + input_scalar_num, vertical_levels)
        return x


class FunctionToDataMean(nn.Module): # Works fine for operators
    def __init__(self, target_profile_num, target_scalar_num):
        super().__init__()
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
    
    def forward(self, x):
        """ x: (batch, target_profile_num + target_scalar_num, vertical_levels ) """
        # Set x to (batch, target_profile_num*vertical_levels + target_scalar_num)
        # Separate profile and scalar parts
        x_profile = x[:, :self.target_profile_num, :]           # (batch, target_profile_num, vertical_levels)
        x_scalar = x[:, self.target_profile_num:, :]            # (batch, target_scalar_num, vertical_levels)

        # Flatten the profile part across all vertical levels
        x_profile_flat = x_profile.reshape(x.shape[0], -1)       # (batch, target_profile_num * vertical_levels)

        # Average the scalar part over the vertical levels
        x_scalar_mean = x_scalar.mean(dim=2)                     # (batch, target_scalar_num)

        # Concatenate flattened profile + averaged scalar
        x = torch.cat([x_profile_flat, x_scalar_mean], dim=1)    # (batch, target_profile_num * vertical_levels + target_scalar_num)
        return x


class FunctionToDataNet(nn.Module): # Warning: this breaks the operator aspect but could be adapted to be a neural operator
    def __init__(self, target_profile_num, target_scalar_num, vertical_levels, out_network_dim):
        super().__init__()
        self.vertical_level = vertical_levels
        self.out_network_dim = out_network_dim
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
        self.layer1 = nn.Linear(self.vertical_level, self.out_network_dim)
        self.layer2 = nn.Linear(self.out_network_dim, 1)

    def forward(self, x):
        """ x: (batch, target_profile_num + target_scalar_num, vertical_levels) """
        # Set x to (batch, target_profile_num*vertical_levels + target_scalar_num)
        # Separate profile and scalar parts
        x_profile = x[:, :self.target_profile_num, :]           # (batch, target_profile_num, vertical_levels)
        x_scalar = x[:, self.target_profile_num:, :]            # (batch, target_scalar_num, vertical_levels)

        # Compute a scalar from the profiles corresponding to scalar
        x_scalar = self.layer1(x_scalar)                   # (batch, target_scalar_num, out_network_dim)
        x_scalar = F.relu(x_scalar)
        x_scalar = self.layer2(x_scalar)                   # (batch, target_scalar_num, 1)
        x_scalar = x_scalar.squeeze(2)                     # (batch, target_scalar_num)

        # Flatten the profile part across all vertical levels
        x_profile_flat = x_profile.reshape(x.shape[0], -1)       # (batch, target_profile_num * vertical_levels)

        # Concatenate flattened profile + averaged scalar
        x = torch.cat([x_profile_flat, x_scalar], dim=1)    # (batch, target_profile_num * vertical_levels + target_scalar_num)
        return x

class ChannelLinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layer = nn.Linear(self.in_channels, self.out_channels)
    
    def forward(self, x):
        """ x: (batch, in_channels, size) """
        x = x.permute(0, 2, 1)  # (batch, size, in_channels)
        x = self.layer(x) # (batch, size, out_channels)
        x = x.permute(0, 2, 1) # (batch, out_channels, size)
        return x

class ChannelMLPLayer(nn.Module):
    def __init__(self, 
                 in_channels, # Number of input channels
                 out_channels, # Number of output channels
                 hidden_channels_list # List of channels for each hidden layer
                 ):
        super().__init__()
        layers = torch.nn.ModuleList()
        n = len(hidden_channels_list)
        for i in range(n):
            if i == 0:
                layers.append(nn.Linear(in_channels, hidden_channels_list[0]))
            else:
                layers.append(nn.Linear(hidden_channels_list[i-1], hidden_channels_list[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_channels_list[n-1], out_channels))
        self.net = nn.Sequential(*layers)

    
    def forward(self, x):
        """ x: (batch, in_channels, size) """
        x = x.permute(0, 2, 1)  # (batch, size, in_channels)
        x = self.net(x) # (batch, size, out_channels)
        x = x.permute(0, 2, 1) # (batch, out_cannels , size)
        return x

class FFTLayer(nn.Module):
    def __init__(self, modes):
        super().__init__()
        self.modes = modes

    def forward(self,x):
        """ x: (batch, channels, size) """
        x = torch.fft.rfft(x, dim=2) # (batch, channels, size//2+1)
        if x.shape[2] > self.modes:
            x = x[:,:,:self.modes] # (batch, channels, modes)
        elif x.shape[2] < self.modes:
            x = F.pad(x, (0, self.modes - x.shape[2]), mode='constant', value=0) # (batch, channels, modes)
        return x

class InverseFFTLayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
    
    def forward(self,x):
        """ x: (batch, channels, modes) """
        x = torch.fft.irfft(x, n=self.size, dim=2) # (batch, channels, size)
        return x   


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dims[0]))
        for i in range(len(self.hidden_dims)-1):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]))
        self.layers.append(nn.Linear(self.hidden_dims[-1], self.output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SpectralConvulution(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))
    
    def complex_multiply(self, x, weights):
        # (batch, in_channels, modes), (in_channels, out_channels, modes) -> (batch, out_channels, modes)
        return torch.einsum("bix,iox->box", x, weights)
    
    def fft(self, x) :
        """ x: (batch, channels, size) """
        x = torch.fft.rfft(x, dim=2) # (batch, channels, size//2+1)
        if x.shape[2] > self.modes:
            x = x[:,:,:self.modes] # (batch, channels, modes)
        elif x.shape[2] < self.modes:
            x = F.pad(x, (0, self.modes - x.shape[2]), mode='constant', value=0) # (batch, channels, modes)
        return x  


    def ifft(self,x,size):
        """ x: (batch, channels, modes) """
        x = torch.fft.irfft(x, n=size, dim=2) # (batch, channels, size)
        return x    

    def forward(self, x):
        """ x: (batch, input_channels, size) """

        x_ft = self.fft(x) # (batch, input_channels, modes)
        x_ft = self.complex_multiply(x_ft, self.weights) # (batch, output_channels, modes)
        x = self.ifft(x_ft, x.shape[2]) # (batch, output_channels, size)

        return x


class FNOLayer(nn.Module):
    def __init__(self, input_channels, output_channels, modes):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.modes = modes
        self.spectral_conv = SpectralConvulution(self.input_channels, self.output_channels, self.modes)
        self.skip_layer = nn.Conv1d(self.input_channels, self.output_channels, 1)
    
    def forward(self, x):
        """ x: (batch, input_channels, size) """
        
        x_skip = self.skip_layer(x) # (batch, output_channels, size)
        x_fourier = self.spectral_conv(x) # (batch, output_channels, size)
        
        x = x_skip + x_fourier
        x = F.relu(x)

        return x



