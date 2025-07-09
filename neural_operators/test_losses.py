import torch 
from omegaconf import OmegaConf
import os
from get_data import get_data
from FNO.fno import FNO
import matplotlib.pyplot as plt


cfg = OmegaConf.load("FNO/config.yaml")
save_path = os.path.join(cfg.save_path, cfg.expname)

data, train_loader, val_loader = get_data(cfg)

model = FNO(
    in_channels=data.input_scalar_num + data.input_profile_num,
    out_channels=data.target_scalar_num + data.target_profile_num,
    hidden_channels=cfg.hidden_channels,
    modes=cfg.modes,
    num_layers=cfg.num_layers,
    lifting_layers=cfg.lifting_layers,
    projection_layers=cfg.projection_layers)

model_path = os.path.join(save_path, 'model.pth')
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
# Create a new state dictionary with adjusted keys
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key[2:] if key.startswith('0.') else key
    new_state_dict[new_key] = value

print(state_dict.keys())
model.load_state_dict(new_state_dict)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (f"Total number of trainable parameters: {pytorch_total_params}")


for key, value in new_state_dict.items():
    # Create a sample tensor
    tensor = value

    # Convert to NumPy array
    data = tensor.numpy()
    data = data.flatten()  # Flatten the array for histogram plotting
    data = abs(data)  # Take absolute values for better visualization

    # Plot using matplotlib
    plt.figure(figsize=(8, 5))
    plt.hist(data)
    plt.title('Distribution of ' + key + ' Values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('parameters_distribution/tensor_distribution'+ key.replace('.', '_') + '.png')


