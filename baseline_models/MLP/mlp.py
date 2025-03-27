import torch
import torch.nn as nn
from dataclasses import dataclass
import modulus


device = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class MLPMetaData(modulus.ModelMetaData):
    name: str = "mlp"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = False
    amp_gpu: bool = False

class MLP(modulus.Module):
    def __init__(
            self,
            input_profile_num: int = 9, # number of input profile
            input_scalar_num: int = 17, # number of input scalars
            target_profile_num: int = 5, # number of target profile
            target_scalar_num: int = 8, # number of target scalars
            output_prune: bool = True, # whether or not we prune strato_lev_out levels
            strato_lev_out: int = 12, # number of levels to set to zero
            loc_embedding: bool = False, # whether or not to use location embedding
            embedding_type: str = "positional", # type of location embedding
            hidden_layers_dim : list = [] # list of the hidden layers size
    ):
        super().__init__(meta=MLPMetaData())
        
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
        self.output_prune = output_prune
        self.strato_lev_out = strato_lev_out
        self.loc_embedding = loc_embedding
        self.embedding_type = embedding_type
        self.vertical_level_num = 60
        self.inputs_dim = input_profile_num * self.vertical_level_num + input_scalar_num
        self.targets_dim = target_profile_num * self.vertical_level_num + target_scalar_num
        self.hidden_layers_dim = hidden_layers_dim
        
        layers = []
        previous_dim = self.inputs_dim
        for hidden_dim in range(self.hidden_layers_dim):
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(previous_dim, self.targets_dim))

        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        y = self.model(x)
        
        if self.output_prune:
            y = y.clone()
            for i in range(4):
                start = i * 60 + 60
                y[:, start:start + self.strato_lev_out] = y[:, start:start + self.strato_lev_out].clone().zero_()
        
        return y
