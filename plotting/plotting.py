import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.animation as animation
import os

import data.SQG.constants as SQGConstants


def save_spectrum(rad_truth, rad_forecast, rad_prior, time, dir):
    # Plot mean spectra across batch
    plt.figure(figsize=(4, 3))
    # plt.xlim(0, rad_truth.shape[1]-1)
    x = np.arange(1, rad_truth.shape[1]-1)
    plt.loglog(x, rad_truth.mean(0).numpy()[1:-1], label="Truth")
    plt.loglog(x, rad_forecast.mean(0).numpy()[1:-1], label="Assimilation")
    plt.loglog(x, rad_prior.mean(0).numpy()[1:-1], label="Ens. mean")
    plt.xlabel("Wavenumber")
    plt.ylabel("Power")
    plt.legend()
    plt.title("Spectral Power")
    plt.tight_layout()

    fname = f'{dir}/spectrum_{time}.png'
    plt.savefig(fname, dpi=300)
    plt.close()
    return fname


def save_video(forecast, truth, oberrstdev, obs_p, obs_fn, dir, level=0, cmap='jet', lres=None, radar=False, observer=None):
    if observer is not None and radar:
        obs = torch.zeros_like(torch.tensor(truth))
        for ntime in range(truth.shape[0]):
            truth_t = truth[ntime] / SQGConstants.scalefact
            obs_t, obs_mask_t, _ = observer.observe(torch.tensor(truth_t), t=ntime)
            obs[ntime, obs_mask_t[0]] = obs_t
        
        truth = truth[:, level, :, :]  # shape: [time, lat, lon]
        obs = obs[:, level, :, :]  # shape: [time, lat, lon]
        obs = obs.numpy()
    else:
        truth = truth[:, level, :, :]  # shape: [time, lat, lon]
        if lres is None:
            lres = truth.shape[-1]
        obs = obs_fn(torch.tensor(truth)).reshape(truth.shape[0], lres, lres).numpy() 
        obs += np.random.randn(*obs.shape) * oberrstdev  # shape: [time, lat, lon]
    
    forecast = forecast[:, :, level, :, :]  # shape: [time, ens, lat, lon]
    print(
        f"forecast.shape: {forecast.shape}, truth.shape: {truth.shape}, obs.shape: {obs.shape}")
    rmse = ((torch.tensor(forecast[:]) - torch.tensor(truth[:]
                                                      ).unsqueeze(1)).pow(2).mean(dim=(2, 3)).sqrt())
    rmse_mean = ((torch.tensor(forecast[:]).mean(
        dim=1) - torch.tensor(truth[:])).pow(2).mean(dim=(1, 2)).sqrt())

    ens_mean = forecast.mean(axis=1)  # shape: [time, lat, lon]
    ens_std = forecast.std(axis=1)  # shape: [time, lat, lon]
    members = forecast[:, :4]  # shape: [time, ens, lat, lon]

    # TODO make this range smaller
    vmin = -20 if cmap == 'jet' else np.min(truth)
    vmax = 20 if cmap == 'jet' else np.max(truth)

    vmin_obs = np.min(obs)
    vmax_obs = np.max(obs)

    fig, axs = plt.subplots(2, 4, figsize=(13, 7), constrained_layout=True)

    # Initial images
    img = []
    titles = ['Truth', 'Obs', 'Mean', 'Std',
              'Member 1', 'Member 2', 'Member 3', 'Member 4']

    img.append(axs[0, 0].imshow(truth[0], cmap=cmap, vmin=vmin, vmax=vmax))
    img.append(axs[0, 1].imshow(
        obs[0], cmap=cmap, vmin=vmin_obs, vmax=vmax_obs))
    img.append(axs[0, 2].imshow(ens_mean[0], cmap=cmap, vmin=vmin, vmax=vmax))
    img.append(axs[0, 3].imshow(
        ens_std[0], cmap=cmap, vmin=0, vmax=ens_std.max()))

    for i in range(4):
        img.append(axs[1, i].imshow(members[0, i],
                   cmap=cmap, vmin=vmin, vmax=vmax))

    # Titles and formatting
    for i, ax in enumerate(axs.flatten()):
        ax.set_title(titles[i], fontsize=14)
        ax.axis('off')

    # --- Add static colorbars for Truth, Obs, and Std ---
    def set_cbar(im):
        ax = im.axes
        cax = inset_axes(ax,
                         width="100%",
                         height="5%",
                         loc='upper center',
                         bbox_to_anchor=(0, 0.15, 1, 1),
                         bbox_transform=ax.transAxes,
                         borderpad=0)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.xaxis.set_ticks_position('top')
        cbar.ax.xaxis.set_label_position('top')
        return cbar

    # These don't seem to work right now
    # set_cbar(img[0])  # Truth
    # set_cbar(img[1])  # Obs
    # set_cbar(img[3])  # Std

    sutitle = f't = {0}'
    fig.suptitle(sutitle, fontsize=14)  # , y=1.1)

    def update(frame):
        img[0].set_array(truth[frame])
        img[1].set_array(obs[frame])
        img[2].set_array(ens_mean[frame])
        img[3].set_array(ens_std[frame])
        for i in range(4):
            img[4 + i].set_array(members[frame, i])

        # Update the title with the current time
        sutitle = f't={frame}'  # TODO: It does not seem like this is added?
        fig.suptitle(sutitle, fontsize=14, y=1.1)

        titles = ['Truth',
                  f'Obs, $\sigma$={oberrstdev}, p={np.round(obs_p*100, 1)}%',
                  f'Mean ({rmse_mean[frame]:.3f})',
                  'Std',
                  ]
        for i in range(4):
            titles.append(f'Member {i+1} ({rmse[frame, i]:.3f})')

        for i, ax in enumerate(axs.flatten()):
            ax.set_title(titles[i], fontsize=14)
            ax.axis('off')

        return img

    # plt.tight_layout()

    fname = f'{dir}/animation.mp4'
    ani = animation.FuncAnimation(
        fig, update, frames=truth.shape[0], interval=10, blit=True)
    ani.save(fname, writer='ffmpeg', fps=10, dpi=300)
    plt.close(fig)
    return fname


def plot_ensemble_prediction(
    traj_rescaled,
    target_rescaled,
    ens_mean,
    ens_std,
    var_name,
    title,
    step,
    var_vrange=None,
    dir=None,
    cmap='jet',
):

    # [ens, var, lat, lon] or [var, lat, lon]
    n_ens = traj_rescaled.shape[0]
    if var_vrange is not None:
        vmin, vmax = var_vrange
    else:
        vmin = target_rescaled.min()
        vmax = target_rescaled.max()

    # Determine number of member rows
    n_member_rows = 1 if n_ens <= 3 else 2
    nrows = 1 + n_member_rows
    ncols = 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    # Title for the entire figure
    fig.suptitle(title)

    # First row: Ground truth, ens_mean, ens_std
    ax = axes[0, 0]
    ax.set_title(f'Ground Truth')
    im0 = ax.imshow(
        target_rescaled, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')

    ax = axes[0, 1]
    ax.set_title(f'Ensemble Mean')
    im1 = ax.imshow(ens_mean,
                    cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')

    ax = axes[0, 2]
    ax.set_title(f'Ensemble Std')
    im2 = ax.imshow(ens_std,
                    cmap=cmap, vmin=ens_std.min(), vmax=ens_std.max())
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
    ax.axis('off')

    # Second row: up to 3 ensemble members
    for i in range(min(3, n_ens)):
        ax = axes[1, i]
        ax.set_title(f'Ensemble Member {i+1}')
        im = ax.imshow(
            traj_rescaled[i], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis('off')

    # Third row: next 3 ensemble members (if available)
    if n_member_rows == 2:
        for i in range(3, min(6, n_ens)):
            ax = axes[2, i-3]
            ax.set_title(f'Ensemble Member {i+1}')
            im = ax.imshow(
                traj_rescaled[i], cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')
        # Hide unused axes if less than 6 members
        for j in range(n_ens-3, 3):
            axes[2, j].axis('off')

    plt.tight_layout()
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
        fname = f'{dir}/ensemble_{var_name}_step{step}.png'
        plt.savefig(fname, dpi=300)
    return fig


# @matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_latent_samples(prior_samples, vi_samples, title=None):
    """
    Plot samples of latent variable drawn from prior and
    variational distribution

    prior_samples: (samples, d_latent, X, Y)
    vi_samples: (samples, d_latent, X, Y)

    Returns:
    fig: the plot figure
    """
    num_samples, latent_dim, img_side_size, _ = prior_samples.shape
    plot_dims = min(latent_dim, 3)  # Plot first 3 dimensions

    # Get common scale for values
    vmin = min(
        vals[:, :plot_dims].min().cpu().item()
        for vals in (prior_samples, vi_samples)
    )
    vmax = max(
        vals[:, :plot_dims].max().cpu().item()
        for vals in (prior_samples, vi_samples)
    )

    # Create figure
    fig, axes = plt.subplots(num_samples, 2 * plot_dims, figsize=(20, 16))

    # Plot samples
    for row_i, (axes_row, prior_sample, vi_sample) in enumerate(
        zip(axes, prior_samples, vi_samples)
    ):

        for dim_i in range(plot_dims):
            prior_sample_reshaped = (
                prior_sample[dim_i]
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            vi_sample_reshaped = (
                vi_sample[dim_i]
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            # Plot every other as prior and vi
            prior_ax = axes_row[2 * dim_i]
            vi_ax = axes_row[2 * dim_i + 1]
            prior_ax.imshow(prior_sample_reshaped, vmin=vmin, vmax=vmax)
            vi_im = vi_ax.imshow(vi_sample_reshaped, vmin=vmin, vmax=vmax)

            if row_i == 0:
                # Add titles at top of columns
                prior_ax.set_title(f"d{dim_i} (prior)", size=15)
                vi_ax.set_title(f"d{dim_i} (vi)", size=15)

    # Remove ticks from all axes
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(vi_im, ax=axes, aspect=60, location="bottom")
    cbar.ax.tick_params(labelsize=15)

    if title:
        fig.suptitle(title, size=20)

    return fig
