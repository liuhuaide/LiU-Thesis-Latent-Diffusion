# Latent Diffusion for Surface Quasi-Geostrophic Forecasting

Master's thesis code (Huaide Liu, Linköping University).
A latent diffusion pipeline for probabilistic forecasting of two-layer SQG
potential-vorticity fields: a frozen DCAE autoencoder compresses the
(2, 64, 64) state into a (c, 16, 16) latent, where an EDM diffusion model
performs autoregressive forecasting.

## Pipeline

| Stage | Script | Output |
|-------|--------|--------|
| 0. Data generation   | `data/SQG/sqg_nature_run.py`      | (101, 2, 64, 64) trajectories |
| 0. Normalisation     | `data/compute_data_stats.py`      | mean/std + diff stats |
| 1. Autoencoder       | `train/train_sqg_ae.py`           | frozen DCAE (×2/×4/×8/×16) |
| 1.5 AE evaluation    | `eval/eval_sqg_ae.py`, `analysis/plot_ae_reconstruction.py` | RMSE, LSD, spectra |
| 2. Pre-encoding      | `train/encode_dataset.py`         | latent datasets |
| 3. EDM training      | `forecasting/trainer.py`          | Pixel EDM + Latent EDM |
| 3. RQ3 encoder E2    | `train/train_e2_ldm.py`           | learned conditioning encoder |
| 4. End-to-end eval   | `eval/eval_latent_pipeline.py`    | pixel-space RMSE/CRPS/SSR |
| 4. Speed benchmark   | `eval/benchmark_inference.py`     | ms/step, speedup |

## Quickstart

```bash
export PYTHONPATH=$(pwd)
bash scripts/generate_full_pipeline.sh        # data → AE → encode → EDM → eval
```

## Research questions
- **RQ1** — how compression affects *forecasting* (not just reconstruction): Pixel vs Latent EDM (×2, ×4).
- **RQ2** — efficiency–accuracy trade-off + error decomposition (AE reconstruction bias vs diffusion error; `--aetarget`).
- **RQ3** — conditioning mechanism: channel concatenation vs a learned latent encoder E2 (`train/train_e2_ldm.py`).

## Naming conventions (logs / result dirs)
`x2/x4/x8/x16` = compression · `hd32/hd128` = SongUNet base dim ·
`aetarget` = eval vs D(E(x)) · `res/nores` = residual mode ·
`ens20` = ensemble size · `heun` = sampler · `final` = thesis-final run.

## Notes
- `train/train_sqg_ae_gradnorm.py` and AE loss variants (`v2/v3/v4`, gradnorm) are RQ1
  exploration that did **not** improve over the baseline DCAE; kept for the record.
- The final thesis uses **diffusion (EDM) only**; flow-matching / FGN / CVAE explorations
  were removed from this repo.
