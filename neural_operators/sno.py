import torch
import torch.nn as nn
from no_layers import InputLayer, OutputLayerMean, MLP
from no_layers import FFTLayer, InverseFFTLayer
from no_layers import ChebyshevLayer, InverseChebyshevLayer


class SNOFourier(nn.Module):
    def __init__(self, input_profile_num, input_scalar_num, vertical_level_num, target_profile_num, target_scalar_num, num_modes, out_network_dim):
        super().__init__()
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.vertical_level_num = vertical_level_num
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
        self.num_modes = num_modes
        self.out_network_dim = out_network_dim

        self.input_layer = InputLayer(input_profile_num, input_scalar_num, vertical_level_num)
        self.fft_layer = FFTLayer(num_modes)
        self.spectral_network = MLP(num_modes, num_modes, [100])
        self.ifft_layer = InverseFFTLayer(vertical_level_num)
        self.output_layer = OutputLayerMean(target_profile_num, target_scalar_num, vertical_level_num, out_network_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.fft_layer(x)
        x = self.spectral_network(x)
        x = self.ifft_layer(x)
        x = self.output_layer(x)
        return x


class SNOChebyshev(nn.Module):
    def __init__(self, input_profile_num, input_scalar_num, vertical_level_num, target_profile_num, target_scalar_num, out_network_dim):
        super().__init__()
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.vertical_level_num = vertical_level_num
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
        self.out_network_dim = out_network_dim

        self.input_layer = InputLayer(input_profile_num, input_scalar_num, vertical_level_num)
        self.chebyshev_layer = ChebyshevLayer(vertical_level_num)
        self.spectral_network = MLP(vertical_level_num, vertical_level_num, [100])
        self.chebyshev_layer = InverseChebyshevLayer(vertical_level_num)
        self.output_layer = OutputLayerMean(target_profile_num, target_scalar_num, vertical_level_num, out_network_dim)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.chebyshev_layer(x)
        x = self.spectral_network(x)
        x = self.output_layer(x)
        return x
