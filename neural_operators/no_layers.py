import torch
import torch.nn as nn
import torch.nn.functional as F


class InputLayer(nn.Module):
    def __init__(self, input_profile_num, input_scalar_num, vertical_level_num):
        super().__init__()
        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.vertical_level_num = vertical_level_num

    def forward(self, x):
        '''
        x: (batch, input_profile_num*vertical_level_num + input_scalar_num)
        '''
        # Set x to (batch, vertical_level, input_profile_num + input_scalar_num) by repeating scalar values
        x_scalar = x[:,None,self.input_profile_num*self.vertical_level_num:].repeat(1,self.vertical_level_num,1)
        x_profile = x[:,:self.input_profile_num*self.vertical_level_num].reshape(-1,self.input_profile_num,self.vertical_level_num).permute(0,2,1)
        x = torch.cat([x_profile, x_scalar], dim=2)


class OutputLayerMean(nn.Module): # Works fine for operators
    def __init__(self, target_profile_num, target_scalar_num, vertical_level_num):
        super().__init__()
        self.vertical_level = vertical_level_num
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
    
    def forward(self, x):
        """ x: (batch, vertical_level_num , target_profile_num + target_scalar_num) """
        # Set x to (batch, target_profile_num*vertical_level_num + target_scalar_num)
        x_profile = x[:,:,:self.target_profile_num]
        x_scalar = x[:,:,self.target_profile_num:]
        x_profile_flat = x_profile.reshape(x.shape[0], -1) # (batch, vertical_level * input_profile_num)
        x_scalar_mean = x_scalar.mean(dim=1) # (batch, target_scalar_num)
        x = torch.cat([x_profile_flat, x_scalar_mean], dim=1)  # (batch, vertical_level * input_profile_num + input_scalar_num)
        return x


class OutputLayerNet(nn.Module): # Warning: this breaks the operator aspect but could be adapted to be a neural operator
    def __init__(self, target_profile_num, target_scalar_num, vertical_level_num, out_network_dim):
        super().__init__()
        self.vertical_level = vertical_level_num
        self.out_network_dim = out_network_dim
        self.target_profile_num = target_profile_num
        self.target_scalar_num = target_scalar_num
        self.layer1 = nn.Linear(self.vertical_level*self.target_scalar_num, self.out_network_dim)
        self.layer2 = nn.Linear(self.out_network_dim, self.target_scalar_num)

    def forward(self, x):
        """ x: (batch, vertical_level_num , target_profile_num + target_scalar_num) """
        # Set x to (batch, target_profile_num*vertical_level_num + taget_scalar_num)
        x_profile = x[:,:,:self.target_profile_num]
        x_scalar = x[:,:,self.target_profile_num:]
        x_profile_flat = x_profile.reshape(x.shape[0], -1)
        x_scalar = x_scalar.reshape(x.shape[0], -1)
        x_scalar = self.layer1(x_scalar)
        x_scalar = F.relu(x_scalar)
        x_scalar = self.layer2(x_scalar)
        x = torch.cat([x_profile_flat, x_scalar], dim=1)
        return x

