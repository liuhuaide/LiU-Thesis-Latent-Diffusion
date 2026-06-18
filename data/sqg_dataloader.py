import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SQGDataset(Dataset):
    def __init__(self, data_path, stats_dir=None):
        self.files = sorted(Path(data_path).glob("*.npy"))
        assert len(self.files) > 0, f"No .npy files found in {data_path}"
        
        # 强制统计量路径：保证验证集和测试集也使用 train 的均值和方差
        stats_path = Path(stats_dir) if stats_dir else Path(data_path)
        self.mean = torch.load(stats_path / "data_mean.pt", weights_only=True).float()  
        self.std  = torch.load(stats_path / "data_std.pt", weights_only=True).float()
        
        # 每个文件有 100 个时间步
        self.steps_per_file = 100

    def __len__(self):
        return len(self.files) * self.steps_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.steps_per_file
        step_idx = idx % self.steps_per_file
        
        # mmap_mode='r' 极大地加速数据读取，避免内存爆炸
        x = np.load(self.files[file_idx], mmap_mode='r')[step_idx]
        
        # 由于使用了 mmap，必须用 .copy() 复制到内存后才能转为 Tensor
        x = torch.as_tensor(x.copy(), dtype=torch.float32)
        
        # 标准化
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return x

def get_sqg_dataloaders(base_dir, batch_size=64, num_workers=4):
    """
    一键返回 train, val, test 的 DataLoader
    """
    base_path = Path(base_dir)
    train_dir = base_path / "train"
    val_dir = base_path / "validation"
    test_dir = base_path / "test"
    
    # 所有数据集统一使用 train 目录下的统计量
    train_dataset = SQGDataset(train_dir, stats_dir=train_dir)
    val_dataset   = SQGDataset(val_dir,   stats_dir=train_dir)
    test_dataset  = SQGDataset(test_dir,  stats_dir=train_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
