import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import xarray as xr
import sys
import os
from omegaconf import OmegaConf
from fno import FNO, FunctionToDataMean
# Get the parent directory and add it to the path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from get_data import get_data

# Load the configuration file
cfg = OmegaConf.load("config.yaml")


if __name__ == "__main__":

    data, train_loader, val_loader = get_data(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FNO( input_channels=data.input_scalar_num + data.input_profile_num,
                out_channels=data.output_scalar_num + data.output_profile_num,
                hidden_channels=cfg.hidden_channels,
                modes=cfg.modes,
                num_layers=cfg.num_layers,
                lifting_layers=cfg.lifting_layers,
                projection_layers=cfg.projection_layers).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.MSELoss()

    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step.step_size, gamma=cfg.scheduler.step.gamma)
    

    train_losses = []
    val_losses = []

    print("Starting training...")
    
    for epoch in range(cfg.epochs):

        # Training loop
        total_train_loss = 0
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_train_loss / len(train_loader)


        # Validation loop
        total_eval_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_eval_loss += loss.item()
        avg_eval_loss = total_eval_loss / len(val_loader)

        # Step the scheduler (some schedulers may require the validation loss)
        scheduler.step()
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_eval_loss)

        print(f"Epoch {epoch+1}/{cfg.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_eval_loss:.4f}")

    # Save the model
    save_path = os.path.join(cfg.save_path, cfg.expname)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
    data.save_norm(save
    OmegaConf.save(cfg, os.path.join(save_path, 'config.yaml'))
    with open('train_loss.txt', 'w') as f:
        for l in train_losses:
            f.write(f"{l}\n")
    
    with open('val_loss.txt', 'w') as f:
        for l in val_losses:
            f.write(f"{l}\n")

    print(f"Model, losses and normalization saved to {save_path}")
    print("Training complete.")
    


    

