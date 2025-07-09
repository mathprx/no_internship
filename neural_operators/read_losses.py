import matplotlib.pyplot as plt
import os
from omegaconf import OmegaConf

cfg = OmegaConf.load("FNO/config.yaml")
save_path = os.path.join(cfg.save_path, cfg.expname)

train_path = os.path.join(save_path, 'train_loss.txt')
train_losses = []
with open(train_path, 'r') as f:
    for line in f:
        train_losses.append(float(line.strip()))

val_path = os.path.join(save_path, 'val_loss.txt')
validation_losses = []  # Assuming you have a similar file for validation losses
with open(os.path.join(save_path, 'val_loss.txt'), 'r') as f:
    for line in f:
        validation_losses.append(float(line.strip()))

# Afficher les données avec Matplotlib
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(validation_losses, label='Validation Loss', color='orange')
plt.title('Train and Validation Losses : '+ cfg.expname)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('Train_Validation_Losses.png')
