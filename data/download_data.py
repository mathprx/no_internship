import huggingface_hub as hf

repo_id = "LEAP/ClimSim_low-res"

# Dowloading the 1st to 3rd day of each month of year 2 for training and the 15th and 20th day of each month for validation and testing
allowed_paterns = [
    "*/0002*0002-*-0[1-3]-*.nc", # Training data
    "*/0002*0002-*-15-*.nc", # Validation data
    "*/0002*0002-*-20-*.nc", # Testing data
]

hf.snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=allowed_paterns,
    local_dir="data/raw_data",
    )
