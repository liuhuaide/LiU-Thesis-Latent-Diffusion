import os
import wandb
import torch
import numpy as np
from plotting.plotting import save_spectrum, save_video
from metrics.metrics import rmse, crps_ens, spread_skill_ratio, spread_squared, log_spectral_distance

# TODO: Maybe move this to a logging section of the package


def init_wandb_metrics(wandb_logger, val_steps):
    """
    Set up wandb metrics to track
    """
    experiment = wandb_logger.experiment
    experiment.define_metric("val_mean_loss", summary="min")
    for step in val_steps:
        experiment.define_metric(f"val_loss_unroll{step}", summary="min")
