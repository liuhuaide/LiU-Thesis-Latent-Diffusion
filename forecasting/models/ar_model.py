# Standard library
import os

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import wandb

# Local
from metrics import metrics
import data.SQG.constants as SQGConstants


class ARModel(pl.LightningModule):
    """
    Generic auto-regressive forecasting model.
    Abstract class that can be extended.
    """

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.standardization_path = args.data_path
        self.register_buffer("scalefact", torch.tensor(SQGConstants.scalefact))

        self.register_buffer(
            "data_mean",
            torch.load(os.path.join(self.standardization_path,
                       "data_mean.pt"), map_location="cpu", weights_only=True)
            .view(1, -1, 1, 1),
        )
        self.register_buffer(
            "data_std",
            torch.load(os.path.join(self.standardization_path,
                       "data_std.pt"), map_location="cpu", weights_only=True)
            .view(1, -1, 1, 1),
        )
        self.register_buffer(
            "diff_mean",
            torch.load(os.path.join(self.standardization_path,
                       "diff_mean.pt"), map_location="cpu", weights_only=True)
            .view(1, -1, 1, 1),
        )
        self.register_buffer(
            "diff_std",
            torch.load(os.path.join(self.standardization_path,
                       "diff_std.pt"), map_location="cpu", weights_only=True)
            .view(1, -1, 1, 1),
        )

        self.val_metrics = {
            "mae": [],
            "mse": [],
        }
        self.test_metrics = {
            "mae": [],  # NOTE: Can't be removed at the moment
            "mse": [],
        }

        # For making restoring of optimizer state optional
        self.restore_opt = args.restore_opt

        # For example plotting
        self.n_example_pred = args.n_example_pred
        self.plotted_examples = 0

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(), lr=self.args.lr, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=self.args.epochs,
            eta_min=self.args.min_lr if hasattr(self.args, "min_lr") else 0.0,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def predict_step(self, init_states):
        """
        Step state one step ahead using prediction model, X_t -> X_t+1
        init_states: (B, init_states, feature_dim, X, Y)

        Returns: 
        next_state: (B, d_state, X, Y), predicted state X_{t+1} at time t+1
        """
        raise NotImplementedError("No prediction step implemented")

    def unroll_prediction(self, init_states, pred_steps):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, init_states, feature_dim, X, Y)
        """
        prediction_list = []

        for i in range(pred_steps):

            pred_state = self.predict_step(init_states)
            prediction_list.append(pred_state)

            # Update conditioning states
            # TODO: Will this work with init_states < 2?
            init_states = torch.cat(
                (init_states[:, 1:], pred_state.unsqueeze(1)), dim=1
            )

        prediction = torch.stack(
            prediction_list, dim=1
        )

        return prediction

    def predict_step_train(self, init_states, true_state):
        """
        Predict state one time step ahead X_t -> X_t+1

        init_states: (B, N_steps, d_state, X, Y)
        true_state: (B, d_state, X, Y)

        Returns:
        next_state: (B, d_state, X, Y), predicted state X_{t+1} at time t+1
        loss: (B, d_state, X, Y), loss for the prediction
        """

        raise NotImplementedError(
            "You need to implement the predict_step_train method in ARProbModel")

    def unroll_prediction_train(self, init_states, true_states, pred_steps):
        """
        Roll out prediction taking multiple autoregressive steps with model
        init_states: (B, 2, num_grid_nodes, d_f)
        forcing_features: (B, pred_steps, num_grid_nodes, d_static_f)
        true_states: (B, pred_steps, num_grid_nodes, d_f)
        """
        prediction_list = []
        loss_list = []

        for i in range(pred_steps):

            pred_state, loss = self.predict_step_train(
                init_states, true_states[:, i])
            prediction_list.append(pred_state)
            loss_list.append(loss)

            # Update conditioning states
            # TODO: Will this work with init_states < 2?
            init_states = torch.cat(
                (init_states[:, 1:], pred_state.unsqueeze(1)), dim=1
            )

        prediction = torch.stack(
            prediction_list, dim=1
        )
        loss = torch.stack(
            loss_list, dim=1
        )  # (B, pred_steps, d_state, X, Y)

        return prediction, loss

    def training_step(self, batch):
        """
        Train on single batch
        """
        init_states, true_states = batch
        prediction, loss = self.unroll_prediction_train(
            init_states, true_states, self.args.ar_steps)

        # Compute loss
        batch_loss = torch.mean(
            loss
        )  # mean over unrolled times and batch

        log_dict = {"train_loss": batch_loss}
        self.log_dict(
            log_dict, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True
        )
        return batch_loss

    def validation_step(self, batch, batch_idx):
        """
        Run validation on single batch
        """
        init_states, true_states = batch
        prediction = self.unroll_prediction(
            init_states, self.args.ar_steps_eval)

        time_step_loss = torch.mean(
            self.loss(
                prediction,
                true_states,
            ),
            dim=0,
        )  # (time_steps-1)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        val_log_dict = {
            f"val_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
        }
        val_log_dict["val_mean_loss"] = mean_loss
        self.log_dict(
            val_log_dict, on_step=False, on_epoch=True, sync_dist=True
        )

        # Store MSEs
        entry_mses = metrics.mse(
            prediction,
            true_states,
            mean_vars=False,
        )
        self.val_metrics["mse"].append(entry_mses)

    def on_validation_epoch_end(self):
        """
        Compute val metrics at the end of val epoch
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.val_metrics, prefix="val")

        # Clear lists with validation metrics values
        for metric_list in self.val_metrics.values():
            metric_list.clear()

    # pylint: disable-next=unused-argument
    def test_step(self, batch, batch_idx):
        """
        Run test on single batch
        """
        init_states, true_states = batch
        prediction = self.unroll_prediction(
            init_states, self.args.ar_steps_eval)

        time_step_loss = torch.mean(
            self.loss(
                prediction,
                true_states,
            ),
            dim=0,
        )  # (time_steps-1,)
        mean_loss = torch.mean(time_step_loss)

        # Log loss per time step forward and mean
        test_log_dict = {
            f"test_loss_unroll{step}": time_step_loss[step - 1]
            for step in self.args.val_steps_to_log
        }
        test_log_dict["test_mean_loss"] = mean_loss

        self.log_dict(
            test_log_dict, on_step=False, on_epoch=True, sync_dist=True
        )

        # Compute all evaluation metrics for error maps
        # Note: explicitly list metrics here, as test_metrics can contain
        # additional ones, computed differently, but that should be aggregated
        # on_test_epoch_end
        for metric_name in ("mse", "mae"):
            metric_func = metrics.get_metric(metric_name)
            batch_metric_vals = metric_func(
                prediction,
                true_states,
                mean_vars=False,
            )
            self.test_metrics[metric_name].append(batch_metric_vals)

        # TODO: Fix spatial loss maps
        # Save per-sample spatial loss for specific times
        # spatial_loss = self.loss(
        #     prediction, target, average_grid=False
        # )  # (B, pred_steps, num_grid_nodes)
        # log_spatial_losses = spatial_loss[
        #     :, [step - 1 for step in self.args.val_steps_to_log]
        # ]
        # self.spatial_loss_maps.append(log_spatial_losses)

        # NOTE: Fix plotting of examples. Should be turned off in ARProbModel
        # Plot example predictions (on rank 0 only)
        # if (
        #     self.trainer.is_global_zero
        #     and self.plotted_examples < self.n_example_pred
        # ):
        #     # Need to plot more example predictions
        #     n_additional_examples = min(
        #         prediction.shape[0], self.n_example_pred - self.plotted_examples
        #     )

        #     self.plot_examples(
        #         batch, n_additional_examples, prediction=prediction
        #     )

    def plot_examples(self, batch, n_examples, prediction=None):
        """
        Plot the first n_examples forecasts from batch

        batch: batch with data to plot corresponding forecasts for
        n_examples: number of forecasts to plot
        prediction: (B, pred_steps, num_grid_nodes, d_f), existing prediction.
            Generate if None.
        """
        raise NotImplementedError("No plotting function implemented")

    # TODO: Add plot_video function

    def create_metric_log_dict(self, metric_tensor, prefix, metric_name):
        """
        Put together a dict with everything to log for one metric.
        Also saves plots as pdf and csv if using test prefix.

        metric_tensor: (pred_steps, d_f), metric values per time and variable
        prefix: string, prefix to use for logging
        metric_name: string, name of the metric

        Return:
        log_dict: dict with everything to log for given metric
        """
        log_dict = {}

        prefix = self.args.eval if self.args.eval is not None else prefix
        full_log_name = f"{prefix}_{metric_name}"

        if self.args.eval == "test":
            # Save errors as csv
            np.savetxt(
                os.path.join(wandb.run.dir, f"{full_log_name}.csv"),
                metric_tensor.cpu().numpy(),
                delimiter=",",
            )

        if metric_tensor.ndim == 1:
            metric_tensor = metric_tensor.unsqueeze(0)
        metric_np = metric_tensor.cpu().numpy()

        # Get mean for the metric over all variables
        metric_mean = torch.mean(
            metric_tensor, dim=1).cpu().numpy()  # (pred_steps,)

        # Add the mean to the log dict and the metric name "Mean"
        metric_names = ["0", "1"] + ["Mean"]
        metric_np = np.column_stack(
            [metric_np, metric_mean]
        )

        # Logging all of the metrics as line plots
        for i, varname in enumerate(metric_names):  # adjust var names as needed
            wandb.log({f"{full_log_name}_{varname}_lineplot": wandb.plot.line_series(
                xs=list(range(metric_np.shape[0])),
                ys=[metric_np[:, i].tolist()],
                keys=[wandb.run.name],
                title=f"{full_log_name} {varname}",
                xname="Time Step"
            )})

        # Check if metrics are watched, log exact values
        # TODO: Add logging for only specific variables
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

    def aggregate_and_plot_metrics(self, metrics_dict, prefix):
        """
        Aggregate and create error map plots for all metrics in metrics_dict

        metrics_dict: dictionary with metric_names and list of tensors
            with step-evals.
        prefix: string, prefix to use for logging
        """
        log_dict = {}
        for metric_name, metric_val_list in metrics_dict.items():
            if not metric_val_list:
                print(
                    f"No values for metric {metric_name}, skipping aggregation.")
                continue  # Skip metrics with no values
            metric_tensor = self.all_gather_cat(
                torch.cat(metric_val_list, dim=0)
            )

            if self.trainer.is_global_zero:
                metric_tensor_averaged = torch.mean(metric_tensor, dim=0)

                # Take square root after averaging to change squared metrics
                if "mse" in metric_name:
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name.replace("mse", "rmse")
                elif metric_name.endswith("_squared"):
                    metric_tensor_averaged = torch.sqrt(metric_tensor_averaged)
                    metric_name = metric_name[: -len("_squared")]

                # Note: we here assume rescaling for all metrics is linear
                metric_rescaled = metric_tensor_averaged * \
                    self.data_std.view(1, -1) * self.scalefact
                log_dict.update(
                    self.create_metric_log_dict(
                        metric_rescaled, prefix, metric_name
                    )
                )

        if self.trainer.is_global_zero and not self.trainer.sanity_checking:
            wandb.log(log_dict)  # Log all
            plt.close("all")  # Close all figs

    def on_test_epoch_end(self):
        """
        Compute test metrics and make plots at the end of test epoch.
        Will gather stored tensors and perform plotting and logging on rank 0.
        """
        # Create error maps for all test metrics
        self.aggregate_and_plot_metrics(self.test_metrics, prefix="test")

        # TODO: Fix spatial loss maps
        # Plot spatial loss maps
        # spatial_loss_tensor = self.all_gather_cat(
        #     torch.cat(self.spatial_loss_maps, dim=0)
        # )  # (N_test, N_log, num_grid_nodes)
        # if self.trainer.is_global_zero:
        #     mean_spatial_loss = torch.mean(
        #         spatial_loss_tensor, dim=0
        #     )  # (N_log, num_grid_nodes)

        #     loss_map_figs = [
        #         vis.plot_spatial_error(
        #             loss_map,
        #             self.config_loader,
        #             title=f"Test loss, t={t_i} ({self.step_length * t_i} h)",
        #             grid_limits=self.grid_limits,
        #         )
        #         for t_i, loss_map in zip(
        #             self.args.val_steps_to_log, mean_spatial_loss
        #         )
        #     ]

        #     # log all to same wandb key, sequentially
        #     for fig in loss_map_figs:
        #         wandb.log({"test_loss": wandb.Image(fig)})

        #     # also make without title and save as pdf
        #     pdf_loss_map_figs = [
        #         vis.plot_spatial_error(
        #             loss_map, self.config_loader, grid_limits=self.grid_limits
        #         )
        #         for loss_map in mean_spatial_loss
        #     ]
        #     pdf_loss_maps_dir = os.path.join(
        #         wandb.run.dir, "spatial_loss_maps")
        #     os.makedirs(pdf_loss_maps_dir, exist_ok=True)
        #     for t_i, fig in zip(self.args.val_steps_to_log, pdf_loss_map_figs):
        #         fig.savefig(os.path.join(
        #             pdf_loss_maps_dir, f"loss_t{t_i}.pdf"))
        #     # save mean spatial loss as .pt file also
        #     torch.save(
        #         mean_spatial_loss.cpu(),
        #         os.path.join(wandb.run.dir, "mean_spatial_loss.pt"),
        #     )

        # self.spatial_loss_maps.clear()

    def all_gather_cat(self, tensor_to_gather):
        """
        Gather tensors across all ranks, and concatenate across dim. 0
        (instead of stacking in new dim. 0)

        tensor_to_gather: (d1, d2, ...), distributed over K ranks

        returns: (K*d1, d2, ...)
        """
        return self.all_gather(tensor_to_gather.cpu()).flatten(0, 1)

    def on_load_checkpoint(self, checkpoint):
        """
        Perform any changes to state dict before loading checkpoint
        """
        if not self.restore_opt:
            opt = self.configure_optimizers()
            checkpoint["optimizer_states"] = [opt["optimizer"].state_dict()]

    def save_spectrum(self, rad_truth, rad_forecast, time, dir):
        print("Save spectrum")
        print(f"rad_truth shape: {rad_truth.shape}")
        print(f"rad_forecast shape: {rad_forecast.shape}")
        print(f"Saving spectrum at time step {time}")
        print(f"Saving spectrum to dir: {dir}")
        os.makedirs(dir, exist_ok=True)

        # Plot mean spectra across batch
        plt.figure(figsize=(4, 3))

        if rad_truth.dim() == 1:
            rad_truth = rad_truth.unsqueeze(0)
        if rad_forecast.dim() == 1:
            rad_forecast = rad_forecast.unsqueeze(0)

        x = np.arange(1, rad_truth.shape[1]-1)
        plt.loglog(x, rad_truth.mean(0).numpy()[1:-1], label="Truth")
        plt.loglog(x, rad_forecast.mean(0).numpy()[1:-1], label="Forecast")
        plt.xlabel("Wavenumber")
        plt.ylabel("Power")
        plt.legend()
        plt.title("Spectral Power")
        plt.tight_layout()
        fname = f'{dir}/spectrum_{time}.png'
        plt.savefig(fname, dpi=300)
        plt.close()
        return fname

    # NOTE: Only used for debugging
    # def on_after_backward(self):
    #     """
    #     Checks so that there are no gradients that are None, this will cause an issue with the ddp training strategy in pytorch lightning
    #     """
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name, "has no gradient!")
    #             print(param.grad)
