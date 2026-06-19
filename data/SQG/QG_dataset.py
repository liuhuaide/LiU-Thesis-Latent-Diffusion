import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
# import data.SQG.constants as SQGConstants # comment out if you don't need the constants

class SQGStateDataset(Dataset):
    def __init__(self, data_path, mean, std, nx=64, h=3):
        """
        Args:
            data_path (str): Path to folder containing files.
            mean, std: normalization stats.
        """
        self.mean = mean
        self.std = std

        # Modified: add fault tolerance to make sure the file is found
        file_pattern = os.path.join(data_path, f"sqg_N{nx}_{h}hrly_*.npy")
        file_list = sorted(glob.glob(file_pattern))
        
        if not file_list:
            # Fallback: try the 3hrly file (your filenames use 3hrly)
            file_pattern = os.path.join(data_path, f"sqg_N{nx}_3hrly_*.npy")
            file_list = sorted(glob.glob(file_pattern))

        if not file_list:
            raise ValueError(
                f"No files found in {data_path} matching sqg_N{nx}_{h}hrly_*.npy")

        # Load all data from selected files and concatenate along first axis
        data_list = [np.load(f) for f in file_list]
        self.data = torch.tensor(np.concatenate(
            data_list, axis=0), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        # Normalize the data
        x = (x - self.mean) / self.std
        return x


class SQGAssimDataset(Dataset):
    def __init__(self, data_path, mean, std):
        """
        Args:
            data_path (str): Path to data file.
            mean, std: normalization stats.
        """
        self.mean = mean
        self.std = std
        # Maybe it is better to load the .nc file directly?, Then we can access more stats about the data directly
        self.data = np.load(data_path+'.npy')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x


class SQGForecastDataset(Dataset):
    def __init__(self,
                 data_path,
                 subsample_step=1,
                 nx=64,
                 pred_length=1,
                 init_states=2,
                 max_length=100,
                 split="train",
                 standardize=True,
                 subset_ds=False,
                 ):
        """
        Args:
            data_path (str): Path to data file (should be the 'train', 'validation' or 'test' folder).
            mean, std: normalization stats.
            nx (int): Spatial resolution.
            pred_length (int): Number of time steps to predict.
            split (str): "train", "validation", or "test".
            standardize (bool): Whether to standardize the data.
            subset (bool): Whether to use a small subset of the data for debugging.
        """
        self.pred_length = pred_length
        self.init_states = init_states
        self.subsample_step = subsample_step
        self.max_length = max_length
        self.standardize = standardize
        
        # [Edit 1] Fix path handling: use the data_path passed in directly
        # Note: assumes data_path points to the dataset/train folder, where the .pt files live
        self.standardization_path = data_path 

        if standardize:
            # Use try-except to avoid a hard crash when the file is missing and give a clearer message
            try:
                self.data_mean = torch.load(os.path.join(
                    self.standardization_path, "data_mean.pt"), weights_only=True).view(1, -1, 1, 1)
                self.data_std = torch.load(os.path.join(
                    self.standardization_path, "data_std.pt"), weights_only=True).view(1, -1, 1, 1)
            except FileNotFoundError:
                print(f"WARNING: Stats files not found in {self.standardization_path}. Trying parent directory...")
                # Try the parent directory (in case data_path is a subdirectory)
                parent_dir = os.path.dirname(self.standardization_path)
                self.data_mean = torch.load(os.path.join(
                    parent_dir, "data_mean.pt"), weights_only=True).view(1, -1, 1, 1)
                self.data_std = torch.load(os.path.join(
                    parent_dir, "data_std.pt"), weights_only=True).view(1, -1, 1, 1)

        self.subset_ds = subset_ds

        # [Edit 2] Allow "validation" as a split name
        assert split in ("train", "val", "validation", "test"), f"Unknown dataset split: {split}"
        
        if split == "train":
            self.random_subsample = True
        else:
            self.random_subsample = False

        # Note: if data_path is already .../train, joining split would give .../train/train
        # Since the data_path passed in is already the specific subdirectory, use data_path directly
        self.sample_dir_path = data_path 
        
        self.trajectory_files = sorted(glob.glob(os.path.join(
            data_path, f"sqg_N{nx}_3hrly_*.npy")))
        
        if not self.trajectory_files:
            raise ValueError(
                f"No files found in {data_path} matching sqg_N{nx}_3hrly_*.npy")
        
        if init_states == 0:
            self.sample_length = pred_length * subsample_step
        else:
            self.sample_length = 1 + \
                (init_states-1 + pred_length) * subsample_step

        print(f"Init states: {init_states}")
        print(f"pred_length: {pred_length}")
        print(f"Sample length: {self.sample_length}")
        print(f"Subsample step: {self.subsample_step}")
        print(f"Max length: {self.max_length}")

        assert (
            self.sample_length <= 100 
        ), f"Requesting too long time series samples. Requested length ({self.sample_length}) exceeds max length (100)."

        self.trajectories_per_file = self.max_length // self.sample_length
        print(f"Trajectories per file: {self.trajectories_per_file}")

        if subset_ds:
            # Limit to 1 file
            self.trajectory_files = self.trajectory_files[:1]
            self.trajectories_per_file = min(
                4, self.trajectories_per_file)  # Only 4 samples per file

        print(f"Found {len(self.trajectory_files)} trajectory files.")
        print(f"Using {self.trajectories_per_file} trajectories per file.")

    def __len__(self):
        return len(self.trajectory_files) * self.trajectories_per_file

    def __getitem__(self, idx):
        # We want to find non-overlapping trajectories
        file_idx = idx // self.trajectories_per_file
        start_idx = idx - file_idx * self.trajectories_per_file
        
        if self.random_subsample:
            end_idx = start_idx + self.sample_length
            overflow = end_idx - self.max_length
            max_start_idx = start_idx - overflow if overflow < 0 else start_idx
            max_start_idx = min(max_start_idx, start_idx + self.sample_length)
            if max_start_idx > start_idx:
                start_idx = torch.randint(
                    start_idx, max_start_idx, ()).item()
                
        sample_path = self.trajectory_files[file_idx]
        try:
            full_sample = torch.tensor(
                np.load(sample_path), dtype=torch.float32
            )
        except ValueError:
            print(f"Failed to load {sample_path}")

        sample = full_sample[start_idx: start_idx +
                             self.sample_length: self.subsample_step]

        if self.standardize:
            # Standardize sample
            sample = (sample - self.data_mean) / self.data_std

        init_states = sample[:self.init_states]
        target_states = sample[self.init_states:]

        # B, T, C, H, W
        return init_states, target_states


if __name__ == "__main__":
    # [Edit 3] Fixes for the test code
    # 1. Removed h=3 (it is not in __init__)
    # 2. Change data_path to your own actual path so you can run the test directly
    dataset = SQGForecastDataset(
        data_path="/local/data2/huali824/mt-huaide-liu/data/SQG/dataset/train",
        nx=64,
        pred_length=1,
        init_states=2,
        split="train",
        standardize=True,
        subset_ds=False, # Note: the original code used 'subset'; renamed to 'subset_ds' here
    )
    print(f"Dataset length: {len(dataset)}")
    init_states, target_states = dataset[0]
    print(f"Input shape: {init_states.shape}")
    print(f"Target shape: {target_states.shape}")
