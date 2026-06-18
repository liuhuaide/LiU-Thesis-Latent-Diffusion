# EDM diffusion model for time series forecasting
# Third-party
import torch
import wandb
import math
import matplotlib.pyplot as plt

from plotting import plotting
from metrics import metrics
from forecasting.models.ar_prob_model import ARProbModel
from networks.diffusion_networks import SongUNet
import data.SQG.constants as SQGConstants


class EDM(ARProbModel):
    """
    A EDM-based probabilistic auto-regressive weather forecasting model
    """

    def __init__(self, args):
        super().__init__(args)

        self.sigma_max = 88
        self.sigma_min = 0.002
        self.sigma_data = 1
        self.rho = 7

        self.backbone = SongUNet(
            img_resolution=torch.as_tensor(args.nx),
            in_channels=2*args.init_states + 2,  # +2 for latent channels
            out_channels=2,
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
        init_states: (B, init_states, feature_dim, X, Y)

        Returns:
        next_state: (B, d_state, X, Y), predicted state X_{t+1} at time t+1
        """

        input_grid = init_states.reshape(init_states.shape[0],
                                         -1,
                                         init_states.shape[3],
                                         init_states.shape[4])

        latents = torch.randn_like(input_grid[:, :2, :, :])

        # Run through sampler
        if self.args.sampler == "heun":
            next_state, _ = self.heun_sampler(latents=latents,
                                              class_labels=input_grid,
                                              sigma_min=self.sigma_min*1.5,
                                              sigma_max=self.sigma_max / 1.1,
                                              num_steps=self.args.sampler_steps)
        elif self.args.sampler == "edm":
            next_state, _ = self.edm_sampler(latents=latents,
                                             class_labels=input_grid,
                                             sigma_min=self.sigma_min*1.5,
                                             sigma_max=self.sigma_max / 1.1,
                                             num_steps=self.args.sampler_steps)
        elif self.args.sampler == "ddpm":
            next_state, _ = self.ddpm_sampler(latents=latents,
                                              class_labels=input_grid,
                                              sigma_min=self.sigma_min*1.5,
                                              sigma_max=self.sigma_max / 1.1,
                                              num_steps=self.args.sampler_steps)

        # Add residual if needed
        if self.args.pred_residual:
            # TODO: Enable support for pred_residual
            next_state = (next_state * self.diff_std) + \
                self.diff_mean  # Unormalize residual
            next_state = init_states[:, -1] + next_state

        return next_state

    def predict_step_train(self, init_states, true_state):
        """
        Predict state one time step ahead X_t -> X_t+1

        init_states: (B, N_steps, d_state, X, Y)
        true_state: (B, d_state, X, Y)

        Returns:
        next_state: (B, d_state, X, Y), predicted state X_{t+1} at time t+1
        loss: (B, d_state, X, Y), loss for the prediction
        """

        input_grid = init_states.reshape(init_states.shape[0],
                                         -1,
                                         init_states.shape[3],
                                         init_states.shape[4])

        # Sample from F inverse
        rnd_uniform = torch.rand(
            [true_state.shape[0], 1, 1, 1], device=true_state.device)
        rho_inv = 1 / self.rho
        sigma_max_rho = self.sigma_max ** rho_inv
        sigma_min_rho = self.sigma_min ** rho_inv
        sigma = (sigma_max_rho + rnd_uniform *
                 (sigma_min_rho - sigma_max_rho)) ** self.rho
        # (B, N_grid, d_input), true_states[4, 19, n_grid, d_state], assuming 19 is for 19 rollouts
        y = true_state
        # Make y residual if needed
        if self.args.pred_residual:
            y = y - init_states[:, -1]
            y = (y - self.diff_mean) / \
                self.diff_std  # Normalize residual

        n = torch.randn_like(y) * sigma
        noisy_input = y+n

        # Shape (B, d_state, N_x, N_y)
        next_state = self.denoise(
            noisy_input, sigma, input_grid)

        # Add residual if needed
        if self.args.pred_residual:
            next_state = (next_state * self.diff_std) + \
                self.diff_mean  # Unormalize residual
            next_state = init_states[:, -1] + next_state

        # (B, 1, 1, 1), weight for the loss function
        weight = (sigma ** 2 + self.sigma_data ** 2) / \
            (sigma * self.sigma_data) ** 2

        loss = weight.squeeze() * self.loss(next_state, true_state)

        return next_state, loss


# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models"."""

    # ----------------------------------------------------------------------------
    # Proposed EDM sampler (Algorithm 2).

    def edm_sampler(
        self, latents, class_labels=None, randn_like=torch.randn_like,
        num_steps=20, sigma_min=0.03, sigma_max=80, rho=7,
        S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05,
    ):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
                   * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps, device=latents.device), torch.zeros_like(
            t_steps[:1], device=latents.device)])  # t_N = 0

        # Main sampling loop.
        x_next = latents * t_steps[0]
        # 0, ..., N-1
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            # diff_steps.append(x_cur)

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) -
                        1) if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(
                t_cur + gamma * t_cur, device=latents.device)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * \
                S_noise * randn_like(x_cur, device=latents.device)

            # Euler step.
            denoised = self.denoise(
                x_hat, t_hat, class_labels=class_labels)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(
                    x_next, t_next, class_labels=class_labels)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * \
                    (0.5 * d_cur + 0.5 * d_prime)

        return x_next, None

    # ----------------------------------------------------------------------------
    # Proposed Heun sampler (Algorithm 1).
    def heun_sampler(
        self, latents, class_labels=None, randn_like=torch.randn_like,
        num_steps=20, sigma_min=0.03, sigma_max=80, rho=7,
    ):

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.sigma_min)
        sigma_max = min(sigma_max, self.sigma_max)

        # Time step discretization.
        step_indices = torch.arange(num_steps)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1)
                   * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([torch.as_tensor(t_steps, device=latents.device), torch.zeros_like(
            t_steps[:1], device=latents.device)])  # t_N = 0

        # Main sampling loop.
        x_next = latents * t_steps[0]
        # 0, ..., N-1
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            denoised = self.denoise(
                x_cur, t_cur, class_labels=class_labels)
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                denoised = self.denoise(
                    x_next, t_next, class_labels=class_labels)
                d_prime = (x_next - denoised) / t_next
                x_next = x_cur + (t_next - t_cur) * \
                    (0.5 * d_cur + 0.5 * d_prime)

        return x_next, None

# ----------------------------------------------------------------------------

    # Sampler used in GenCast (taken from a reimplementation of the paper before the official code was released).
    # TODO: Check if this is correct.
    def ddpm_sampler(
        self, latents, class_labels=None, randn_like=torch.randn_like,
        num_steps=20, sigma_min=0.03, sigma_max=80, rho=7,
        S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05, r=0.5,
    ):

        time_steps = torch.arange(
            0, num_steps, device=latents.device) / (num_steps - 1)
        sigmas = (sigma_max ** (1 / rho) + time_steps *
                  (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

        # initialize noise
        x = sigmas[0] * latents

        for i in range(len(sigmas) - 1):
            # stochastic churn from Karras et al. (Alg. 2)
            gamma = (
                min(S_churn / num_steps, math.sqrt(2) - 1)
                if S_min <= sigmas[i] <= S_max
                else 0.0
            )
            # noise inflation from Karras et al. (Alg. 2)
            noise = S_noise * randn_like(latents, device=latents.device)

            sigma_hat = sigmas[i] * (gamma + 1)
            if gamma > 0:
                x = x + (sigma_hat**2 - sigmas[i] ** 2) ** 0.5 * noise
            denoised = self.denoise(
                x, sigma_hat, class_labels=class_labels)

            if i == len(sigmas) - 2:
                # final Euler step
                d = (x - denoised) / sigma_hat
                x = x + d * (sigmas[i + 1] - sigma_hat)
            else:
                # DPMSolver++2S  step (Alg. 1 in Lu et al.) with alpha_t=1.
                # t_{i-1} is t_hat because of stochastic churn!
                lambda_hat = -torch.log(sigma_hat)
                lambda_next = -torch.log(sigmas[i + 1])
                h = lambda_next - lambda_hat
                lambda_mid = lambda_hat + r * h
                sigma_mid = torch.exp(-lambda_mid)

                u = sigma_mid / sigma_hat * x - \
                    (torch.exp(-r * h) - 1) * denoised
                denoised_2 = self.denoise(
                    u, sigma_mid, class_labels=class_labels)
                D = (1 - 1 / (2 * r)) * denoised + 1 / (2 * r) * denoised_2
                x = sigmas[i + 1] / sigma_hat * x - (torch.exp(-h) - 1) * D

        return x, None
