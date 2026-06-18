# Standard library
import os
import subprocess
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from data.SQG.QG_dataset import SQGForecastDataset


def save_stats(
    static_dir_path, means, squares, filename_prefix
):
    means = (
        torch.stack(means) if len(means) > 1 else means[0]
    )  # (N_batch, d_features,)
    squares = (
        torch.stack(squares) if len(squares) > 1 else squares[0]
    )  # (N_batch, d_features,)
    print(f"means shape: {means.shape}, squares shape: {squares.shape}")
    mean = torch.mean(means, dim=0)  # (d_features,)
    second_moment = torch.mean(squares, dim=0)  # (d_features,)
    std = torch.sqrt(second_moment - mean**2)  # (d_features,)

    print(f"Saving computed stats to {static_dir_path}...")
    print(f"{filename_prefix} mean: {mean}")
    print(f"{filename_prefix} std.: {std}")
    print(f"mean shape: {mean.shape}, std. shape: {std.shape}")

    torch.save(
        mean.cpu(), os.path.join(static_dir_path, f"{filename_prefix}_mean.pt")
    )
    torch.save(
        std.cpu(), os.path.join(static_dir_path, f"{filename_prefix}_std.pt")
    )


def main():
    """
    Pre-compute parameter weights to be used in loss function
    """
    parser = ArgumentParser(description="Training arguments")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/local/data2/huali824/mt-huaide-liu/data/SQG/dataset/train",
        help="Path to save the computed statistics (default: /local/data2/huali824/mt-huaide-liu/data/SQG/dataset/train)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size when iterating over the dataset",
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )

    args = parser.parse_args()

    # Load dataset without any subsampling
    ds = SQGForecastDataset(
        data_path=args.data_path,
        nx=64,
        pred_length=1,
        init_states=1,
        split="train",
        standardize=False,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    print("Computing mean and std.-dev. for parameters...")
    means, squares = [], []

    for init_batch, target_batch in tqdm(loader):
        # (N_batch, N_t, N_grid, d_features)
        batch = torch.cat((init_batch, target_batch), dim=1)
        # Flux at 1st windowed position is index 1 in forcing
        # (N_batch, d_features,)
        means.append(torch.mean(batch, dim=(1, 3, 4)).cpu())
        squares.append(
            torch.mean(batch**2, dim=(1, 3, 4)).cpu()
        )  # (N_batch, d_features,)

    means = [torch.cat(means, dim=0)]  # (N_batch, d_features,)
    squares = [torch.cat(squares, dim=0)]  # (N_batch, d_features,)

    save_stats(
        args.data_path,
        means,
        squares,
        "data",
    )

    print("Computing mean and std.-dev. for one-step differences...")
    ds_standard = SQGForecastDataset(
        data_path=args.data_path,
        nx=64,
        pred_length=1,
        init_states=1,
        split="train",
        standardize=True,
    )  # Re-load with standardization

    loader_standard = torch.utils.data.DataLoader(
        ds_standard,
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    diff_means, diff_squares = [], []

    for init_batch, target_batch in tqdm(loader_standard):
        batch = torch.cat((init_batch, target_batch), dim=1)

        # (N_batch', N_t, N_grid, d_features),
        # N_batch' = args.step_length*N_batch
        batch_diffs = batch[:, 1:] - batch[:, :-1]
        # (N_batch', N_t-1, N_grid, d_features)
        diff_means.append(torch.mean(batch_diffs, dim=(1, 3, 4)).cpu())
        # (N_batch', d_features,)
        diff_squares.append(torch.mean(batch_diffs**2, dim=(1, 3, 4)).cpu())
        # (N_batch', d_features,)

    diff_means = [torch.cat(diff_means, dim=0)]  # (N_batch', d_features,)
    diff_squares = [torch.cat(diff_squares, dim=0)]  # (N_batch', d_features,)

    save_stats(args.data_path,
               diff_means,
               diff_squares,
               "diff")


if __name__ == "__main__":
    main()
