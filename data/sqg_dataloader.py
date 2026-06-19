import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SQGDataset(Dataset):
    def __init__(self, data_path, stats_dir=None):
        self.files = sorted(Path(data_path).glob("*.npy"))
        assert len(self.files) > 0, f"No .npy files found in {data_path}"
        
        # Force the stats path so val/test also use train's mean and std
        stats_path = Path(stats_dir) if stats_dir else Path(data_path)
        self.mean = torch.load(stats_path / "data_mean.pt", weights_only=True).float()  
        self.std  = torch.load(stats_path / "data_std.pt", weights_only=True).float()
        
        # Each file has 100 time steps
        self.steps_per_file = 100

    def __len__(self):
        return len(self.files) * self.steps_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.steps_per_file
        step_idx = idx % self.steps_per_file
        
        # mmap_mode='r' greatly speeds up reading and avoids blowing up memory
        x = np.load(self.files[file_idx], mmap_mode='r')[step_idx]
        
        # Because mmap is used, copy into memory with .copy() before converting to a Tensor
        x = torch.as_tensor(x.copy(), dtype=torch.float32)
        
        # Standardize
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return x

def get_sqg_dataloaders(base_dir, batch_size=64, num_workers=4):
    """
    Return the train, val, and test DataLoaders in one call
    """
    base_path = Path(base_dir)
    train_dir = base_path / "train"
    val_dir = base_path / "validation"
    test_dir = base_path / "test"
    
    # All splits use the statistics from the train directory
    train_dataset = SQGDataset(train_dir, stats_dir=train_dir)
    val_dataset   = SQGDataset(val_dir,   stats_dir=train_dir)
    test_dataset  = SQGDataset(test_dir,  stats_dir=train_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
