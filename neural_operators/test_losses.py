import torch 
from omegaconf import OmegaConf
import os
from get_data import get_data
from FNO.fno import FNO
from no_layers import FunctionToDataMean
import matplotlib.pyplot as plt
import tqdm
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

cfg = OmegaConf.load("FNO/config.yaml")
save_path = os.path.join(cfg.save_path, cfg.expname)

data, train_loader, val_loader = get_data(cfg)

fno = FNO(
    in_channels=data.input_scalar_num + data.input_profile_num,
    out_channels=data.target_scalar_num + data.target_profile_num,
    hidden_channels=cfg.hidden_channels,
    modes=cfg.modes,
    num_layers=cfg.num_layers,
    lifting_layers=cfg.lifting_layers,
    projection_layers=cfg.projection_layers)

fno_path = os.path.join(save_path, 'model.pth')
state_dict = torch.load(fno_path) #, map_location=torch.device('cpu'))
# Create a new state dictionary with adjusted keys
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key[2:] if key.startswith('0.') else key
    new_state_dict[new_key] = value

fno.load_state_dict(new_state_dict)

function_to_scalar = FunctionToDataMean(data.target_profile_num, data.target_scalar_num)

model = torch.nn.Sequential(fno, function_to_scalar)
model.to(device)
model.eval()


pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (f"Total number of trainable parameters: {pytorch_total_params}")

if False :
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
        plt.close()  # Close the figure to free memory

with torch.no_grad():
    train_loss = 0.0
    cnt = 0
    for inputs, labels in tqdm.tqdm(train_loader, desc="Training Loss Calculation"):
        if cnt > 4:
            break
        cnt += 1
        inputs = inputs.to(device)
        labels = labels.to(device)
        t = time.time()
        outputs = model(inputs).to(device)
        t_1 = time.time()
        print(f"Time taken for forward pass: {t_1 - t:.4f} seconds")
        batch_train_loss = torch.nn.functional.mse_loss(outputs, labels).item()
        train_loss += batch_train_loss
    train_loss = train_loss / 5 #len(train_loader)

    val_loss = 0.0
    cnt = 0
    for inputs, labels in tqdm.tqdm(val_loader, desc="Validation Loss Calculation"):
        if cnt > 4:
            break
        cnt += 1
        inputs = inputs.to(device)
        labels = labels.to(device)
        t = time.time()
        outputs = model(inputs).to(device)
        t_1 = time.time()
        print(f"Time taken for forward pass: {t_1 - t:.4f} seconds")
        batch_val_loss = torch.nn.functional.mse_loss(outputs, labels).item()
        val_loss += batch_val_loss
    val_loss = val_loss / 5 #len(val_loader)

print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
