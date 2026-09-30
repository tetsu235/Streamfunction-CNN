"""Entry point for training your streamfunction CNN on HDF5 flow series.

Keep main.py beside these existing files:
    streamfunction_cnn.py
    streamfunction_loss.py
    train_streamfunction.py
    hdf5_series_loader.py

Requires Python 3.10+, torch, h5py, and numpy in the same environment.

1. Set the file_path in CONFIG, or pass --file on the command line.
2. Run: python main.py --inspect
   This lists dataset keys/shapes without reading the velocity arrays.
3. Set velocity_key, dx, dy, dt, nu, and any other settings below.
4. Run: python main.py

The existing functions perform the actual loops:
    train_hdf5_series -> train_multiple_series -> train_model
    epoch -> series -> batch of complete temporal triplets

One entire series must fit in CPU RAM. Only its current triplet batch is
sent to the model's device. Model and optimizer are created just once.
The loss assumes unforced, incompressible 2D flow on a uniform fixed grid.
Velocities must use consistent physical or nondimensional units, channels
[u, v], H=y, W=x, and coordinates increasing with array indices.

To reload a saved model for inference:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

Checkpoint reference:
https://docs.pytorch.org/tutorials/beginner/saving_loading_models
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import h5py
import torch
from torch import nn

from training.Series_loader import inspect_hdf5, train_hdf5_series
from models.CNN2D import ConvBlockConfig, StreamfunctionCNN


BASE_DIR = Path(__file__).resolve().parent.parent

# Edit these settings for your actual dataset and experiment.
CONFIG: dict[str, Any] = {
    "file_path": str(BASE_DIR / "datasets" / "ns_incom_inhom_2d_512-0.h5"),
    "velocity_key": "velocity",  # Replace with the actual HDF5 dataset key.
    "input_key": None,  # None: use target velocities as same-resolution inputs.
    "time_key": None,   # Optional shared [T] or per-series [S,T] timestamps.
    "series_indices": None,  # None: all series; e.g. [0, 1, 2] holds out series 3.

    # REQUIRED physical values. Do not infer them from the array shape alone.
    "dx": 0.001953125,  # Output/target grid spacing in x, NOT coarse input spacing.
    "dy": 0.001953125,  # Output/target grid spacing in y.
    "dt": 0.005,  # Time between SAVED snapshots, NOT the solver's internal step.
    "nu": 0.01,  # Kinematic viscosity; use the same units as the other fields.

    "epochs": 20,           # Each epoch visits every selected series once.
    "batch_size": 1,        # Number of triplets; 1 triplet = 3 frames for the CNN.
    "learning_rate": 1e-3,
    "shuffle_series": False,
    "shuffle_triplets": True,  # Reorders whole triplets, preserving time order.
    "seed": 42,
    "device": "auto",       # "auto" selects CUDA if available, otherwise CPU.
    "log_every": 1,
    "cost_params": {
        "lambda_u": 1.0,
        "lambda_v": 1.0,
        "lambda_momentum": 1.0,
    },  # Starting weights only; their useful values depend on the loss scales.
    "blocks": [(32, 5), (64, 3), (64, 3), (32, 3)],  # (channels, kernel size)
    "output_dir": str(BASE_DIR / "outputs"),
}


def read_shapes(config: dict[str, Any]) -> tuple[tuple, tuple]:
    """Read only HDF5 metadata; return input shape and velocity-target shape."""
    with h5py.File(config["file_path"], "r") as hdf:
        shapes = []
        input_key = config["input_key"]
        if input_key is None:
            input_key = config["velocity_key"]
        for key in (input_key, config["velocity_key"]):
            if key not in hdf:
                raise KeyError(f"Dataset {key!r} was not found. Run with --inspect.")
            dataset = hdf[key]
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 5:
                raise ValueError(f"{key!r} must have shape [series,time,H,W,channels].")
            if min(dataset.shape) < 1 or dataset.shape[1] < 3:
                raise ValueError(f"{key!r} needs nonempty axes and at least 3 times.")
            shapes.append(tuple(dataset.shape))

    input_shape, target_shape = shapes
    if input_shape[:2] != target_shape[:2]:
        raise ValueError("Input and target series/time axes must match.")
    if target_shape[-1] != 2 or min(target_shape[2:4]) < 5:
        raise ValueError("Velocity targets require 2 channels [u,v] and a grid >= 5x5.")
    return input_shape, target_shape


def build_model(model_config: dict[str, Any]) -> StreamfunctionCNN:
    """Construct the CNN from the plain settings also stored in its checkpoint."""
    output_size = model_config["upsample_size"]
    upsample = None if output_size is None else partial(
        nn.Upsample, size=tuple(output_size), mode="bilinear", align_corners=False,
    )
    return StreamfunctionCNN(
        in_channels=model_config["in_channels"],
        blocks=[
            ConvBlockConfig(out_channels=channels, kernel_size=kernel)
            for channels, kernel in model_config["blocks"]
        ],
        upsample=upsample,
        output_conv_kwargs={"kernel_size": 1, "padding": 0, "bias": True},
    )


def main(config: dict[str, Any], *, inspect_only: bool = False) -> list[float] | None:
    """Prepare one experiment, start the existing training loop, and save results."""
    config = dict(config)
    data_path = Path(config["file_path"]).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"HDF5 file not found: {data_path}\n"
            "Set CONFIG['file_path'] or pass --file path/to/your_data.h5."
        )
    config["file_path"] = str(data_path)

    # Inspection works before you know the dataset key or physical parameters.
    if inspect_only:
        inspect_hdf5(data_path)
        return None

    missing = [name for name in ("dx", "dy", "dt", "nu") if config[name] is None]
    if missing:
        raise ValueError(f"Set these values in CONFIG from your dataset: {', '.join(missing)}.")

    # 1. Inspect shapes without loading a complete series or all velocity data.
    input_shape, target_shape = read_shapes(config)

    # A separate coarse input_key enables coarse-to-fine reconstruction.
    # Resize to the exact target shape before the CNN's convolution stack.
    # The input and target grids must cover the same physical region and times.
    # This does not generate coarse inputs; they must already exist in the HDF5.
    model_config = {
        "in_channels": input_shape[-1],
        "blocks": config["blocks"],
        "upsample_size": target_shape[2:4] if input_shape[2:4] != target_shape[2:4] else None,
    }

    # 2. Seed initialization, choose the device, and create ONE model.
    torch.manual_seed(config["seed"])
    requested_device = config["device"]
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    model = build_model(model_config).to(device=device, dtype=torch.float32)

    # 3. Create ONE optimizer AFTER moving the model to its final device.
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # A separate folder for each run keeps previous training results intact.
    output_root = Path(config["output_dir"]).expanduser().resolve()
    config["output_dir"] = str(output_root)
    run_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"Device: {device}")
    print(f"HDF5 inputs: {input_shape}; velocity targets: {target_shape}")
    print(f"Batch: {config['batch_size']} triplet(s) = {3 * config['batch_size']} frames")
    print(f"Results: {run_dir}")
    print("Starting training: epoch -> series -> triplet batch", flush=True)

    # 4. START TRAINING. Do not add another epoch or series loop around this.
    # This wrapper loads one series, uses the real streamfunction_loss cost,
    # updates weights for each batch, releases that series, and loads the next.
    # dx/dy/dt/nu are passed separately; window_length=3 and reduction='mean'
    # are already set by train_multiple_series. No DataLoader is needed here.
    history = train_hdf5_series(
        model=model,
        optimizer=optimizer,
        file_path=data_path,
        velocity_key=config["velocity_key"],
        input_key=config["input_key"],
        time_key=config["time_key"],
        series_indices=config["series_indices"],
        dx=config["dx"], dy=config["dy"], dt=config["dt"], nu=config["nu"],
        epochs=config["epochs"],
        device=device,
        batch_size=config["batch_size"],
        shuffle_series=config["shuffle_series"],
        shuffle_triplets=config["shuffle_triplets"],
        seed=config["seed"],
        cost_params=config["cost_params"],
        log_every=config["log_every"],
    )

    # 5. Save AFTER all requested epochs finish. These are final weights, not
    # a best-validation model; this entry point does not run validation.
    checkpoint_path = run_dir / "checkpoint_final.pt"
    torch.save({
        "epochs_completed": len(history),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model_config,
        "training_config": config,
        "input_shape": input_shape,
        "target_shape": target_shape,
        "loss_history": history,
    }, checkpoint_path)

    history_path = run_dir / "loss_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "mean_training_cost"])
        writer.writerows(enumerate(history, start=1))

    print(f"Saved {checkpoint_path}")
    print(f"Saved {history_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, help="Override CONFIG['file_path'].")
    parser.add_argument("--inspect", action="store_true", help="List HDF5 datasets and exit.")
    args = parser.parse_args()
    settings = dict(CONFIG)
    if args.file is not None:
        settings["file_path"] = str(args.file)
    main(settings, inspect_only=args.inspect)
