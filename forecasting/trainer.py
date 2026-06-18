# Standard library
import json
import random
import time
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
from pytorch_lightning.callbacks import LearningRateMonitor

# First-party
from forecasting.models.edm import EDM
from forecasting.models.latent_edm import LatentEDM
from data.SQG.QG_dataset import SQGForecastDataset
import utils

MODELS = {
    "EDM": EDM,
    "LatentEDM": LatentEDM,
}


def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def main(input_args=None):
    """
    Main function for training and evaluating models
    """
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="neural_lam/data_config.yaml",
        help="Path to data config file (default: neural_lam/data_config.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EDM",
        help="Model architecture to train/evaluate (default: FGN)",
    )
    parser.add_argument(
        "--subset_ds",
        action="store_true",
        help="Use only a small subset of the dataset, for debugging"
        "(default: false)",
    )
    parser.add_argument(
        "--seed",
        type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=4,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="upper epoch limit (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size (default: 32)"
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        action="store_true",
        help="If optimizer state should be restored with model "
        "(default: false)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=32,
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )
    parser.add_argument(
        "--num_sanity_steps",
        type=int,
        default=2,
        help="Number of sanity checking validation steps to run before starting"
        " training (default: 2)",
    )

    # Model architecture
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Dimensionality of all hidden representations (default: 32)",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="heun",
        help="The sampler to use when generating trajectories with a diffusion model"
        "(heun/edm) (default: heun)",
    )

    # Training options
    parser.add_argument(
        "--init_states",
        type=int,
        default=2,
        help="Number of initial states "
        "(default: 2)",
    )
    parser.add_argument(
        "--ar_steps",
        type=int,
        default=1,
        help="Number of steps to unroll prediction for in loss "
        "(default: 1)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        help="Loss function to use, see metric.py (default: mse)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--min_lr", type=float, default=1e-5, help="minimum learning rate for cosine annealing (default: 0.00001)"
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--pred_residual",
        action="store_true",
        help="If the model should predict residuals instead of absolute values",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for training. (default: 0.01)",
    )

    # EDM options
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=0.002,
        help="Sigma min for training. (default: 0.002)",
    )
    parser.add_argument(
        "--sigma_max",
        type=float,
        default=88,
        help="Sigma min for training. (default: 88)",
    )

    # SI options
    parser.add_argument(
        "--sigma_coef",
        type=float,
        default=1,
        help="Sigma coefficient for stochatic interpolants (default: 1)",
    )

    # Backbone / UNET options
    parser.add_argument(
        "--resample_filter",
        type=list_of_ints,
        default="1,3,3,1",
        help="Resample filter for backbone unet (default: 1,1 or 1,3,3,1)",
    )
    parser.add_argument(
        "--channel_mult",
        type=list_of_ints,
        default="2,2,2,2",
        help="Channel multiplier for backbone unet (depth and width of UNET) (default: 2,2,2,2)",
    )

    parser.add_argument(
        "--encoder_type",
        type=str,
        default="residual",
        help="Type of encoder to use in backbone unet (standard/residual/skip)"
        "(default: 'standard')",
    )

    parser.add_argument(
        "--attn_resolutions",
        type=list_of_ints,
        default="1",
        help="Resolutions to apply attention to in edm model (default: '1')",
    )

    parser.add_argument(
        "--noise_embedding",
        type=str,
        default="fourier",
        help="Type of encoder to use in edm model (positional/fourier/linear)"
        "(default: 'fourier')",
    )
    parser.add_argument(
        "--channel_mult_emb",
        type=int,
        default=4,
        help="Channel multiplier for noise embedding MLP (default: 4)",
    )
    parser.add_argument(
        "--channel_mult_noise",
        type=int,
        default=1,
        help="Channel multiplier for noise level MLP (default: 1)",
    )

    # CRPS Options
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=32,
        help="Dimension of the noise vector z, 32 in FGN (default: 32)",
    )
    parser.add_argument(
        "--crps_alpha",
        type=float,
        default=0.95,
        help="Alpha parameter for the Almost Fair CRPS (default: 0.95)",
    )
    parser.add_argument(
        "--crps_beta",
        type=float,
        default=1.0,
        help="Beta parameter for the Generalized CRPS (default: 1.0)",
    )
    parser.add_argument(
        "--spatial_noise_dim",
        type=int,
        default=0,
        help="Number of spatial noise channels to concatenate to the input "
        "of the model (default: 0)",
    )
    parser.add_argument(
        "--spectral_loss_weight",
        type=float,
        default=0,
        help="Weighting for spectral loss term of loss, not computed if = 0. (default: 0)",
    )
    parser.add_argument(
        "--n_ens_train",
        type=int,
        default=2,
        help="Number of ensemble members to use during training for CRPS "
        "computation (default: 2)",
    )

    # CVAE Options
    parser.add_argument(
        "--latent_samples_plot",
        type=int,
        default=4,
        help="Number of latent samples to plot (default: 4)",
    )
    parser.add_argument(
        "--learn_prior",
        action="store_true",
        help="If the model should learn a prior distribution (default: False)",
    )
    parser.add_argument(
        "--kl_beta",
        type=float,
        default=1.0,
        help="Beta weighting in front of kl-term in ELBO (default: 1)",
    )
    parser.add_argument(
        "--crps_weight",
        type=float,
        default=0,
        help="Weighting for CRPS term of loss, not computed if = 0. CRPS is "
        "computed based on trajectories sampled using prior distribution. "
        "(default: 0)",
    )
    parser.add_argument(
        "--likelihood_weight",
        type=float,
        default=1.0,
        help="Weighting for likelihood term of loss. (default: 1)",
    )

    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--ar_steps_eval",
        type=int,
        default=50,
        help="Number of steps to unroll prediction for in evaluation "
        "(default: 50)",
    )
    parser.add_argument(
        "--n_example_pred",
        type=int,
        default=1,
        help="Number of example predictions to plot during val/test "
        "(default: 1)",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of ensemble members during evaluation (default: 5)",
    )
    parser.add_argument(
        "--sampler_steps",
        type=int,
        default=20,
        help="Number of sampling steps during inference (default: 20)",
    )

    # System options
    parser.add_argument(
        "--step_length",
        type=int,
        default=3,
        help="Step length in hours (default: 3 hours)",
    )
    parser.add_argument("--nx", type=int, default=64,
                        help="Spatial resolution")
    parser.add_argument("--lat_channels", type=int, default=8,
                        help="Number of latent channels for LatentEDM (default: 8)")
    # Data path arguments (override hardcoded paths)
    parser.add_argument("--train_data_path", type=str, default=None,
                        help="Path to training data. If None, uses default pixel-space path.")
    parser.add_argument("--val_data_path", type=str, default=None,
                        help="Path to validation data. If None, uses default pixel-space path.")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Path to test data. If None, uses default pixel-space path.")

    # Logger Settings
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="SQG",
        help="Wandb run project (default: 'SQG')",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="",
        help="Wandb run name (default: '')",
    )
    parser.add_argument(
        "--val_steps_to_log",
        type=list_of_ints,
        default="1,25,50",
        help="Steps to log val loss for (default: [1, 25, 50])",
    )
    parser.add_argument(
        "--metrics_watch",
        nargs="+",
        default=["val_rmse"],
        help="List of metrics to watch, including any prefix (e.g. val_rmse)",
    )

    args = parser.parse_args(input_args)
    # TODO: Can be added to enable only logging certain variables at certain lead times
    # args.var_leads_metrics_watch = {
    #     int(k): v for k, v in json.loads(args.var_leads_metrics_watch).items()
    # }

    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)

    # Set seed
    seed.seed_everything(args.seed)

    dataset = SQGForecastDataset

    # Resolve data paths — use explicit overrides if given, else defaults
    train_data_path = args.train_data_path if args.train_data_path else args.data_path
    val_data_path = args.val_data_path if args.val_data_path else args.data_path
    test_data_path = args.test_data_path if args.test_data_path else args.data_path

    # Load data
    train_loader = torch.utils.data.DataLoader(
        dataset(
            data_path=train_data_path,
            subsample_step=args.step_length // 3,
            nx=64,
            pred_length=args.ar_steps,
            init_states=args.init_states,
            split="train",
            standardize=True,
            subset_ds=args.subset_ds,
        ),
        args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset(
            data_path=val_data_path,
            subsample_step=args.step_length // 3,
            nx=64,
            pred_length=args.ar_steps_eval,
            init_states=args.init_states,
            split="val",
            standardize=True,
            subset_ds=args.subset_ds,
        ),
        args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
    )

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "cuda"
        # Allows using Tensor Cores on A100s
        torch.set_float32_matmul_precision("high")
    else:
        device_name = "cpu"

    args.device = device_name

    # Load model parameters Use new args for model
    model_class = MODELS[args.model]
    model = model_class(args)

    prefix = "subset-" if args.subset_ds else ""
    if args.eval:
        prefix = prefix + f"eval-{args.eval}-"

    prefix = f"{args.wandb_run_name}-{prefix}" if args.wandb_run_name else prefix
    run_name = (
        f"{prefix}{args.model}-{args.hidden_dim}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )

    # Callbacks for saving model checkpoint
    callbacks = []
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            filename="min_val_loss",
            monitor="val_mean_loss",
            mode="min",
            save_last=True,
        )
    )

    callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            filename="last_epoch",
            monitor="epoch",
            save_on_train_epoch_end=True,
            save_top_k=1,
            every_n_epochs=1,
            save_last=True,  # Optionally also save the last epoch
        )
    )

    # Saving the evaluation results
    wandb_project = args.wandb_project if args.eval is None else f"{args.wandb_project}_eval"

    if args.wandb_entity is not None:
        logger = pl.loggers.WandbLogger(
            entity=args.wandb_entity, project=wandb_project, name=run_name, config=args
        )
    else:
        logger = pl.loggers.WandbLogger(
            project=wandb_project, name=run_name, config=args
        )

    # Training strategy
    # If doing pure autoencoder training (kl_beta = 0), the prior network is not
    # used at all in producing the loss. This is desired, but DDP complains.
    strategy = "ddp" if args.kl_beta > 0 else "ddp_find_unused_parameters_true"

    # Training strategy
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy=strategy,
        accelerator=device_name,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        num_sanity_val_steps=args.num_sanity_steps,
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0:
        utils.init_wandb_metrics(
            logger, args.val_steps_to_log
        )  # Do after wandb.init

    # if args.load is not None:
    #     ckpt = torch.load(args.load, map_location="cpu")
    #     state_dict = ckpt.get("state_dict", ckpt)

    #     model_state = model.state_dict()
    #     # keep only keys that exist in the current model and have matching shapes
    #     compatible_state = {
    #         k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape
    #     }

    #     incompatible_keys = [k for k in state_dict.keys(
    #     ) if k in model_state and model_state[k].shape != state_dict[k].shape]
    #     missing_keys = [
    #         k for k in model_state.keys() if k not in compatible_state]

    #     # load compatible params; missing keys remain initialized by model's __init__
    #     model.load_state_dict(compatible_state, strict=False)

    #     print(
    #         f"Checkpoint partial load: loaded {len(compatible_state)} keys, "
    #         f"{len(missing_keys)} missing, {len(incompatible_keys)} incompatible shape keys (skipped)."
        # )

    #     # prevent Lightning from trying to restore the checkpoint again
    #     ckpt_path_to_pass = None
    # else:
    #     ckpt_path_to_pass = None  # or set from args as desired

    if args.eval:
        if args.eval == "val":
            eval_loader = val_loader
        else:  # Test
            eval_loader = torch.utils.data.DataLoader(
                dataset(
                    data_path=test_data_path,
                    subsample_step=args.step_length // 3,
                    nx=64,
                    pred_length=args.ar_steps_eval,
                    init_states=args.init_states,
                    split="test",
                    standardize=True,
                    subset_ds=args.subset_ds,
                ),
                args.batch_size,
                shuffle=False,
                num_workers=args.n_workers,
                pin_memory=True,
                persistent_workers=True,
            )
        print(f"Running evaluation on {args.eval}")
        trainer.test(model=model, dataloaders=eval_loader, ckpt_path=args.load)
        print(f"Evaluated model trained for {trainer.current_epoch} epochs.")
    else:
        print("Starting training")
        # Train model
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.load,
        )


if __name__ == "__main__":
    main()
