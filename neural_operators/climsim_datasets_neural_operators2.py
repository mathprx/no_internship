from torch.utils.data import Dataset
import numpy as np
import torch
import glob
import h5py


class DatasetNO(Dataset):
    def __init__(self, 
                 input_path, 
                 target_path, 
                 input_sub, 
                 input_div, 
                 out_scale, 
                 qinput_prune, 
                 output_prune, 
                 strato_lev,
                 qn_lbd,
                 decouple_cloud = False, 
                 aggressive_pruning = False,
                 strato_lev_qc = 30,
                 strato_lev_qinput = 22, 
                 strato_lev_tinput = -1,
                 strato_lev_out = 12,
                 input_clip = True,
                 input_clip_rhonly = True,
                 input_profile_num = 17,
                 input_scalar_num = 9,
                 output_profile_num = 8,
                 output_scalar_num = 5,
                 vertical_levels = 60):
        """
        Args:
            input_path (str): Path to the .npy file containing the inputs.
            target_path (str): Path to the .npy file containing the targets.
            input_sub (np.ndarray): Input data mean.
            input_div (np.ndarray): Input data standard deviation.
            out_scale (np.ndarray): Output data standard deviation.
            qinput_prune (bool): Whether to prune the input data.
            output_prune (bool): Whether to prune the output data.
            strato_lev (int): Number of levels in the stratosphere.
            qn_lbd (np.ndarray): Coefficients for the exponential transformation of qn.
        """
        self.input_array = np.load(input_path)
        self.target_array = np.load(target_path)
        self.input_sub = input_sub
        self.input_div = input_div
        self.out_scale = out_scale
        self.qinput_prune = qinput_prune
        self.output_prune = output_prune
        self.strato_lev = strato_lev
        self.qn_lbd = qn_lbd
        self.decouple_cloud = decouple_cloud
        self.aggressive_pruning = aggressive_pruning
        self.strato_lev_qc = strato_lev_qc
        self.strato_lev_out = strato_lev_out
        self.input_clip = input_clip
        if strato_lev_qinput <0:
            self.strato_lev_qinput = strato_lev
        else:
            self.strato_lev_qinput = strato_lev_qinput
        self.strato_lev_tinput = strato_lev_tinput
        self.input_clip_rhonly = input_clip_rhonly

        if self.strato_lev_qinput <self.strato_lev:
            raise ValueError('strato_lev_qinput should be greater than or equal to strato_lev, otherwise inconsistent with E3SM')
        assert len(self.input_array) == len(self.target_array)

        self.input_profile_num = input_profile_num
        self.input_scalar_num = input_scalar_num
        self.output_profile_num = output_profile_num
        self.output_scalar_num = output_scalar_num
        self.vertical_levels = vertical_levels

        assert 60%vertical_levels == 0, "vertical_levels should be a divisor of 60 for now"

    def __len__(self):
        return len(self.input_array)

    def __getitem__(self, idx):
        x = self.input_array[idx]
        y = self.target_array[idx]
        x[120:180] = 1 - np.exp(-x[120:180] * self.qn_lbd)
        # Avoid division by zero in input_div and set corresponding x to 0
        # input_div_nonzero = self.input_div != 0
        # x = np.where(input_div_nonzero, (x - self.input_sub) / self.input_div, 0)
        x = (x - self.input_sub) / self.input_div
        #make all inf and nan values 0
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0

        y = y * self.out_scale
        if self.decouple_cloud:
            x[120:240] = 0
            x[60*14:60*16] =0
            x[60*19:60*21] =0
        elif self.aggressive_pruning:
            # for profiles, only keep stratosphere temperature. prune all other profiles in stratosphere
            x[60:60+self.strato_lev_qinput] = 0 # prune RH
            x[120:120+self.strato_lev_qc] = 0
            x[180:180+self.strato_lev_qinput] = 0
            x[240:240+self.strato_lev] = 0 # prune u
            x[300:300+self.strato_lev] = 0 # prune v
            x[360:360+self.strato_lev] = 0
            x[420:420+self.strato_lev] = 0
            x[480:480+self.strato_lev] = 0
            x[540:540+self.strato_lev] = 0
            x[600:600+self.strato_lev] = 0
            x[660:660+self.strato_lev] = 0
            x[720:720+self.strato_lev] = 0
            x[780:780+self.strato_lev_qinput] = 0
            x[840:840+self.strato_lev_qc] = 0 # prune qc_phy
            x[900:900+self.strato_lev_qinput] = 0
            x[960:960+self.strato_lev] = 0
            x[1020:1020+self.strato_lev] = 0
            x[1080:1080+self.strato_lev_qinput] = 0
            x[1140:1140+self.strato_lev_qc] = 0 # prune qc_phy in previous time step
            x[1200:1200+self.strato_lev_qinput] = 0
            x[1260:1260+self.strato_lev] = 0
            x[1515] = 0 #SNOWHICE
        elif self.qinput_prune:
            # x[:,60:60+self.strato_lev] = 0
            x[120:120+self.strato_lev] = 0

        if self.strato_lev_tinput >0:
            x[0:self.strato_lev_tinput] = 0
        
        if self.input_clip:
            if self.input_clip_rhonly:
                x[60:120] = np.clip(x[60:120], 0, 1.2)
            else:
                x[60:120] = np.clip(x[60:120], 0, 1.2) # for RH, clip to (0,1.2)
                x[360:720] = np.clip(x[360:720], -0.5, 0.5) # for dyn forcing, clip to (-0.5,0.5)
                x[720:1320] = np.clip(x[720:1320], -3, 3) # for phy tendencies  clip to (-3,3)

        if self.output_prune:
            y[60:60+self.strato_lev_out] = 0
            y[120:120+self.strato_lev_out] = 0
            y[180:180+self.strato_lev_out] = 0
            y[240:240+self.strato_lev_out] = 0

        # Convert to torch tensors
        x, y =  torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        # Set x to (input_profile_num + input_scalar_num, data_levels (=60)) by repeating scalar values
        x_profile = x[:self.input_profile_num * 60]
        x_profile = x_profile.reshape(self.input_profile_num, 60)  # (input_profile_num, data_levels (=60))

        # Extract and repeat the scalar part along data_levels (=60)
        x_scalar = x[self.input_profile_num * 60:]
        x_scalar = x_scalar[:, None].repeat(1, 60)  # (input_scalar_num, data_levels (=60))

        # Concatenate along the profile+scalar feature dimension
        x = torch.cat([x_profile, x_scalar], dim=0)  # (input_profile_num + input_scalar_num, data_levels (=60))

        # change the vertical resolution of x AND y to vertical_levels (downsample)
        step = 60 // self.vertical_levels
        x = x[:,::step]

        y = torch.concatenate((y[:self.output_profile_num*60:step],y[self.output_profile_num*60:]))
    
        return x,y