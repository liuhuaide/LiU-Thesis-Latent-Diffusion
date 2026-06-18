# Latent EDM diffusion model for time series forecasting in latent space
# Operates on pre-encoded latent representations from a frozen autoencoder.
#
# Key differences from pixel-space EDM (edm.py):
#   - img_resolution = latent spatial size (16 for all compression rates)
#   - in_channels  = lat_channels * init_states + lat_channels
#   - out_channels = lat_channels
#   - Noise shape  = (B, lat_channels, H_lat, W_lat)
#   - scalefact is set to 1 (metrics are in latent space during training;
#     pixel-space metrics are computed in the separate eval pipeline)

# Third-party
import os
import torch
import wandb
import math
import numpy as np
import matplotlib.pyplot as plt

from plotting import plotting
from metrics import metrics
from forecasting.models.ar_prob_model import ARProbModel
from networks.diffusion_networks import SongUNet
import data.SQG.constants as SQGConstants


class LatentEDM(ARProbModel):
    """
    EDM-based probabilistic auto-regressive forecasting model
    operating in the latent space of a pre-trained autoencoder.

    The model trains and predicts entirely in latent space.
    Decoding back to pixel space is handled by the evaluation pipeline.
    """

    def __init__(self, args):
        super().__init__(args)

        # Override scalefact: latent metrics have no physical scale,
        # so we set it to 1 to avoid meaningless rescaling.
        # Pixel-space metrics are computed in the eval pipeline instead.
        self.scalefact = torch.tensor(1.0)

        self.sigma_max = 88
        self.sigma_min = 0.002
        self.sigma_data = 1
        self.rho = 7

        # Latent space dimensions
        lat_channels = args.lat_channels  # e.g., 2/4/8/16 depending on compression
        self.lat_channels = lat_channels

        self.backbone = SongUNet(
            img_resolution=torch.as_tensor(args.nx),  # 16 for all latent spaces
            in_channels=lat_channels * args.init_states + lat_channels,
            out_channels=lat_channels,
            embedding_type="fourier",
            resample_filter=args.resample_filter,
            channel_mult=args.channel_mult,
            encoder_type=args.encoder_type,
            attn_resolutions=args.attn_resolutions,
            channel_mult_emb=args.channel_mult_emb,
            channel_mult_noise=args.channel_mult_noise,
        )

        self.loss = metrics.get_metric(args.loss)

    def denoise(self, x, sigma, class_labels=None):
        """
        EDM denoising with preconditioning.
        Same formulation as pixel-space EDM, but dimensions are latent.

        x: (B, lat_channels, H_lat, W_lat) — noisy latent
        sigma: noise level
        class_labels: (B, lat_channels * init_states, H_lat, W_lat) — conditioning
        """
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / \
            (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.backbone((c_in * x).to(dtype), c_noise.flatten(),
                            class_labels=class_labels)
        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def predict_step(self, init_states):
        """
        Step state one step ahead using prediction model, X_t -> X_t+1
        init_states: (B, init_states, lat_channels, H_lat, W_lat)

        Returns:
        next_state: (B, lat_channels, H_lat, W_lat), predicted latent state
        """
        # Flatten init_states into conditioning input
        # (B, init_states, lat_channels, H, W) -> (B, init_states*lat_channels, H, W)
        input_grid = init_states.reshape(init_states.shape[0],
                                         -1,
                                         init_states.shape[3],
                                         init_states.shape[4])

        # Sample initial noise in latent dimensions
        latents = torch.randn_like(input_grid[:, :self.lat_channels, :, :])

        # Run through sampler
        if self.args.sampler == "heun":
            next_state, _ = self.heun_sampler(latents=latents,
                                              class_labels=input_grid,
                                              sigma_min=self.sigma_min * 1.5,
                                              sigma_max=self.sigma_max / 1.1,
                                              num_steps=self.args.sampler_steps)
        elif self.args.sampler == "edm":
            next_state, _ = self.edm_sampler(latents=latents,
                                             class_labels=input_grid,
                                             sigma_min=self.sigma_min * 1.5,
                                             sigma_max=self.sigma_max / 1.1,
                                             num_steps=self.args.sampler_steps)
        elif self.args.sampler == "ddpm":
            next_state, _ = self.ddpm_sampler(latents=latents,
                                              class_labels=input_grid,
                                              sigma_min=self.sigma_min * 1.5,
                                              sigma_max=self.sigma_max / 1.1,
                                              num_steps=self.args.sampler_steps)

        # Add residual if needed
        if self.args.pred_residual:
            next_state = (next_state * self.diff_std) + self.diff_mean
            next_state = init_states[:, -1] + next_state

        return next_state

    def predict_step_train(self, init_states, true_state):
        """
        Predict state one time step ahead X_t -> X_t+1

        init_states: (B, N_steps, lat_channels, H_lat, W_lat)
        true_state: (B, lat_channels, H_lat, W_lat)

        Returns:
        next_state: (B, lat_channels, H_lat, W_lat)
        loss: scalar per-sample loss
        """
        # Flatten init_states into conditioning input
        input_grid = init_states.reshape(init_states.shape[0],
                                         -1,
                                         init_states.shape[3],
                                         init_states.shape[4])

        # Sample noise level from F inverse
        rnd_uniform = torch.rand(
            [true_state.shape[0], 1, 1, 1], device=true_state.device)
        rho_inv = 1 / self.rho
        sigma_max_rho = self.sigma_max ** rho_inv
        sigma_min_rho = self.sigma_min ** rho_inv
        sigma = (sigma_max_rho + rnd_uniform *
                 (sigma_min_rho - sigma_max_rho)) ** self.rho

        y = true_state

        # Make y residual if needed
        if self.args.pred_residual:
            y = y - init_states[:, -1]
            y = (y - self.diff_mean) / self.diff_std

        n = torch.randn_like(y) * sigma
        noisy_input = y + n

        # Denoise conditioned on init_states
        next_state = self.denoise(noisy_input, sigma, input_grid)

        # Add residual if needed
        if self.args.pred_residual:
            next_state = (next_state * self.diff_std) + self.diff_mean
            next_state = init_states[:, -1] + next_state

        # EDM loss weighting
        weight = (sigma ** 2 + self.sigma_data ** 2) / \
            (sigma * self.sigma_data) ** 2

        loss = weight.squeeze() * self.loss(next_state, true_state)

        return next_state, loss

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        Override to handle arbitrary number of latent channels.
        The base class hardcodes ["0", "1"] for 2 pixel PV channels.
        """
        log_dict = {}
        prefix = self.args.eval if self.args.eval is not None else prefix
        full_log_name = f"{prefix}_{metric_name}"

        if self.args.eval == "test":

            np.savetxt(
                os.path.join(wandb.run.dir, f"{full_log_name}.csv"),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        metric_np = metric_tensor.cpu().numpy()
        metric_mean = torch.mean(metric_tensor, dim=1).cpu().numpy()

        # Dynamic variable names for latent channels
        n_vars = metric_tensor.shape[1]
        metric_names = [str(i) for i in range(n_vars)] + ["Mean"]
        metric_np = np.column_stack([metric_np, metric_mean])

        for i, varname in enumerate(metric_names):
            wandb.log({f"{full_log_name}_{varname}_lineplot": wandb.plot.line_series(
                xs=list(range(metric_np.shape[0])),
                ys=[metric_np[:, i].tolist()],
                keys=[wandb.run.name],
                title=f"{full_log_name} {varname}",
                xname="Time Step"
            )})

        if full_log_name in self.args.metrics_watch:
            for var_i in range(metric_tensor.shape[1]):
                log_dict.update(
                    {
                        f"{full_log_name}_{var_i}_step_{step}": metric_tensor[
                            step - 1, var_i
                        ]
                        for step in self.args.val_steps_to_log
                    }
                )

        return log_dict

    # =========================================================================
    # Samplers — inherited from pixel-space EDM, identical math
    # =========================================================================

    def edm_sampler(
        self, latents, class_labels=None, randn_like=torch.randn_like,
        num_steps=20, sigma_min=0.03, sigma_max=80, rho=7,
        S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05,
    ):
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        step_indices = torch.arange(num_steps)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
                   * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps, device=latents.device),
                             torch.zeros_like(t_steps[:1], device=latents.device)])

        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next

            gamma = min(S_churn / num_steps, np.sqrt(2) -
                        1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur, device=latents.device)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * \
                S_noise * randn_like(x_cur, device=latents.device)

            denoised = self.denoise(x_hat, t_hat, class_labels=class_labels)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            if i < num_steps - 1:
                denoised = self.denoise(x_next, t_next, class_labels=class_labels)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next, None

    def heun_sampler(
        self, latents, class_labels=None, randn_like=torch.randn_like,
        num_steps=20, sigma_min=0.03, sigma_max=80, rho=7,
    ):
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        step_indices = torch.arange(num_steps)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
                   * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps, device=latents.device),
                             torch.zeros_like(t_steps[:1], device=latents.device)])

        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            denoised = self.denoise(x_cur, t_cur, class_labels=class_labels)
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur

            if i < num_steps - 1:
                denoised = self.denoise(x_next, t_next, class_labels=class_labels)
                d_prime = (x_next - denoised) / t_next
                x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next, None

    def ddpm_sampler(
        self, latents, class_labels=None, randn_like=torch.randn_like,
        num_steps=20, sigma_min=0.03, sigma_max=80, rho=7,
        S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05, r=0.5,
    ):
        time_steps = torch.arange(
            0, num_steps, device=latents.device) / (num_steps - 1)
        sigmas = (sigma_max ** (1 / rho) + time_steps *
                  (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        x = sigmas[0] * latents

        for i in range(len(sigmas) - 1):
            gamma = (
                min(S_churn / num_steps, math.sqrt(2) - 1)
                if S_min <= sigmas[i] <= S_max
                else 0.0
            )
            noise = S_noise * randn_like(latents, device=latents.device)

            sigma_hat = sigmas[i] * (gamma + 1)
            if gamma > 0:
                x = x + (sigma_hat**2 - sigmas[i] ** 2) ** 0.5 * noise
            denoised = self.denoise(x, sigma_hat, class_labels=class_labels)

            if i == len(sigmas) - 2:
                d = (x - denoised) / sigma_hat
                x = x + d * (sigmas[i + 1] - sigma_hat)
            else:
                lambda_hat = -torch.log(sigma_hat)
                lambda_next = -torch.log(sigmas[i + 1])
                h = lambda_next - lambda_hat
                lambda_mid = lambda_hat + r * h
                sigma_mid = torch.exp(-lambda_mid)

                u = sigma_mid / sigma_hat * x - \
                    (torch.exp(-r * h) - 1) * denoised
                denoised_2 = self.denoise(u, sigma_mid, class_labels=class_labels)
                D = (1 - 1 / (2 * r)) * denoised + 1 / (2 * r) * denoised_2
                x = sigmas[i + 1] / sigma_hat * x - (torch.exp(-h) - 1) * D

        return x, None
