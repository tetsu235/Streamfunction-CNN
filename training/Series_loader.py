"""Train on one HDF5 flow series at a time using your existing training loop.

Keep this file beside streamfunction_cnn.py, streamfunction_loss.py, and
train_streamfunction.py. Requires Python 3.10+, h5py, and PyTorch.
Install h5py into the same environment as PyTorch: python -m pip install h5py

Expected velocity layout: [series, time, height, width, 2], channels [u, v].
One complete series is loaded into CPU RAM. Only a triplet batch is moved to
the model's device by train_model. This is NOT frame-by-frame disk streaming.
For [4, 1000, 512, 512, 2], one float32 velocity series needs about 1.95 GiB,
in addition to input data (if separate), batches, activations, and parameters.

Loop order: epochs -> series -> batches of consecutive temporal triplets.
Each epoch visits all selected series; each series gets one pass per epoch.
The existing model and optimizer are reused throughout. Triplets never cross
series boundaries, and their three frames always remain in temporal order.

Data assumptions: H is y, W is x; coordinates increase with array indices;
frames are uniformly spaced by dt; velocity channels contain physical u/v.
dx/dy are OUTPUT-grid spacings. dt is the interval between SAVED frames.
This wrapper uses the same dx, dy, dt, and nu for all selected series.
It performs no normalization, spatial resampling, or velocity sign changes.
Optional input data must match the velocity series and timestamps exactly.

Example, after constructing model and optimizer as in your existing code:

    from hdf5_series_loader import inspect_hdf5, train_hdf5_series

    inspect_hdf5("flow_data.h5")  # Find the real velocity dataset key first.
    history = train_hdf5_series(
        model=model, optimizer=optimizer, file_path="flow_data.h5",
        velocity_key="velocity",  # Replace with the key printed above.
        dx=dx, dy=dy, dt=dt, nu=nu,  # Use your dataset's physical values.
        epochs=20, device=device, batch_size=1,
        shuffle_series=False, shuffle_triplets=True,
        cost_params={"lambda_u": 1.0, "lambda_v": 1.0,
                     "lambda_momentum": 1.0},
    )

input_key=None uses the same velocity tensor for inputs and targets without
duplicating the full series. This is same-resolution training. For coarse-to-
fine reconstruction, supply input_key="your_coarse_velocity_key" and configure
the CNN to produce the velocity target's grid size. This loader does not make
coarse inputs automatically. Loss weights above are starting choices only.

References:
https://docs.h5py.org/en/stable/high/dataset.html#reading-writing-data
https://docs.h5py.org/en/stable/high/dataset.html#h5py.Dataset.astype
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import h5py
import torch
from torch import nn

from training.multi_series_training import FlowSeries, train_multiple_series

from torch.nn import functional as F


def inspect_hdf5(file_path: str | Path) -> None:
    """Print dataset keys, shapes, and dtypes without loading their arrays."""
    with h5py.File(file_path, "r") as hdf:
        def show_dataset(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset):
                print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")

        hdf.visititems(show_dataset)


def _frame_dataset(hdf: h5py.File, key: str) -> h5py.Dataset:
    """Validate metadata only; indexing hdf by a key returns a disk handle."""
    if key not in hdf:
        raise KeyError(f"Dataset {key!r} not found. Run inspect_hdf5(file_path).")
    dataset = hdf[key]
    if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 5:
        raise ValueError(f"{key!r} must have shape [series, time, H, W, channels].")
    if min(dataset.shape) < 1 or dataset.shape[1] < 3:
        raise ValueError(f"{key!r} needs nonempty axes and at least three times.")
    if dataset.dtype.kind not in "iuf":
        raise TypeError(f"{key!r} must contain real numeric data.")
    return dataset

def pooled_spatial_shape(
    height: int,
    width: int,
    factor: int | tuple[int, int],
) -> tuple[int, int]:
    """Validate pooling and return (H_out, W_out) without loading any data.

    factor is an integer or (factor_y, factor_x). Require exact divisibility
    so pooling never silently discards cells along the domain boundary.
    """
    if isinstance(factor, Integral):
        factors = (factor, factor)
    elif isinstance(factor, tuple) and len(factor) == 2:
        factors = factor
    else:
        raise ValueError("factor must be a positive integer or a pair (factor_y, factor_x).")
    if any(isinstance(k, bool) or not isinstance(k, Integral) or k < 1 for k in factors):
        raise ValueError("Pooling factors must be positive integers.")
    factor_y, factor_x = (int(k) for k in factors)
    if height < 1 or width < 1:
        raise ValueError("Spatial dimensions must be positive.")
    if height % factor_y or width % factor_x:
        raise ValueError(
            f"Grid {height}x{width} must be divisible by pooling factors "
            f"{factor_y}x{factor_x}; choose factors that preserve the whole domain."
        )
    return height // factor_y, width // factor_x


def average_pool_series(
    series: torch.Tensor,
    factor: int | tuple[int, int] = 4,
) -> torch.Tensor:
    """Downsample one series [T,C,H,W] by averaging spatial blocks.

    Example: factor=4 maps [1000,2,512,512] to [1000,2,128,128].
    Time occupies PyTorch's batch axis, so snapshots and channels are never
    averaged together. The input tensor is not modified. Its dtype/device are
    preserved, and factor=1 returns the same tensor without allocating a copy.
    """
    if not isinstance(series, torch.Tensor) or series.ndim != 4:
        raise ValueError("series must be a tensor with shape [T,C,H,W].")
    if min(series.shape) < 1 or not series.is_floating_point():
        raise ValueError("series must have nonempty axes and a real floating dtype.")
    height, width = series.shape[-2:]
    pooled_height, pooled_width = pooled_spatial_shape(height, width, factor)
    kernel = (height // pooled_height, width // pooled_width)
    if kernel == (1, 1):
        return series
    return F.avg_pool2d(series, kernel_size=kernel, stride=kernel, padding=0)

def train_hdf5_series(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    file_path: str | Path,
    *,
    velocity_key: str,
    dx: float,
    dy: float,
    dt: float,
    nu: float,
    epochs: int,
    device: torch.device | str,
    input_key: str | None = None,
    downsample_factor: int | tuple[int, int] = 1,
    time_key: str | None = None,
    series_indices: Sequence[int] | None = None,
    batch_size: int = 1,
    shuffle_series: bool = False,
    shuffle_triplets: bool = True,
    seed: int | None = None,
    cost_params: Mapping[str, Any] | None = None,
    log_every: int = 1,
) -> list[float]:
    """Load, train on, and release each series before loading the next one.

    Pass your already-created model and optimizer; neither is reinitialized.
    batch_size counts triplets: batch_size=1 sends 3 frames to the CNN.
    series_indices=None selects all series. Pass train-only indices to hold
    other series out for validation. shuffle_series=False preserves this order.

    Optional input_key names [S,T,H_in,W_in,C_in] in the same HDF5 file.
    Optional time_key names shared times [T] or per-series times [S,T]. The
    existing loop checks their spacing against dt. Without it, you must ensure
    uniform timing yourself. Supplying times does not sort or resample data.

    cost_params contains loss weights/options; dx/dy/dt/nu are separate above.
    Uses streamfunction_loss.cost_function via train_multiple_series.
    Returns one mean training cost per global epoch.
    """
    for name, value in (("dx", dx), ("dy", dy), ("dt", dt), ("nu", nu)):
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not isfinite(value) or value < 0
                or (name != "nu" and value == 0)):
            raise ValueError("dx/dy/dt must be positive and nu nonnegative; all must be finite.")

    # 1. Open the file. Dataset handles and shape metadata do not load all data.
    #    The context manager also closes the file if training raises an error.
    with h5py.File(file_path, "r") as hdf:
        velocities = _frame_dataset(hdf, velocity_key)
        n_series, n_times, height, width, channels = velocities.shape
        if channels != 2 or min(height, width) < 5:
            raise ValueError("Velocity targets need channels [u, v] and a grid >= 5x5.")

        input_data = velocities if input_key is None else _frame_dataset(hdf, input_key)
        if input_data.shape[:2] != (n_series, n_times):
            raise ValueError("Inputs and velocities must have matching series/time axes.")

        timestamps = None
        if time_key is not None:
            timestamps = hdf[time_key]
            if (not isinstance(timestamps, h5py.Dataset)
                    or timestamps.shape not in ((n_times,), (n_series, n_times))
                    or timestamps.dtype.kind not in "iuf"):
                raise ValueError("time_key must name real timestamps [T] or [S,T].")

        indices = list(range(n_series)) if series_indices is None else list(series_indices)
        if not indices or any(
            isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < n_series
            for i in indices
        ):
            raise ValueError(f"series_indices must contain indices from 0 to {n_series - 1}.")
        indices = [int(i) for i in indices]
        if len(set(indices)) != len(indices):
            raise ValueError("series_indices must not contain duplicates.")

        # 2. Define the callback that the existing outer loop calls as needed.
        #    It captures only file handles and settings, not previously loaded data.
        def load_series(series_index: int) -> FlowSeries:
            # Read ONLY this series, converting to float32 during the disk read.
            # Avoid velocities[:] or velocities[()], which read every series.
            array = velocities.astype("float32")[series_index]

            # [T,H,W,2] -> [T,2,H,W]. These operations share the array storage;
            # no extra full-series copy is needed for the axis change.
            targets = torch.from_numpy(array).permute(0, 3, 1, 2)

            if input_data.id == velocities.id:
                inputs = targets
            else:
                input_array = input_data.astype("float32")[series_index]
                inputs = torch.from_numpy(input_array).permute(0, 3, 1, 2)

            inputs = average_pool_series(inputs, factor=downsample_factor)

            times = None
            if timestamps is not None:
                time_values = (
                    timestamps.astype("float64")[:]
                    if timestamps.ndim == 1
                    else timestamps.astype("float64")[series_index]
                )
                times = torch.from_numpy(time_values)

            return FlowSeries(
                inputs=inputs, targets=targets, times=times,
                dx=dx, dy=dy, dt=dt, nu=nu,
            )

        # 3. Reuse your epoch -> series -> triplet-batch loop. It deletes its
        #    loader, dataset, and FlowSeries references before the next callback.
        #    Use one process to avoid worker copies/prefetch of full series.
        return train_multiple_series(
            model=model,
            series_sources=indices,
            optimizer=optimizer,
            epochs=epochs,
            device=device,
            load_series=load_series,
            batch_size=batch_size,
            cost_params=cost_params,
            shuffle_series=shuffle_series,
            shuffle_triplets=shuffle_triplets,
            num_workers=0,
            seed=seed,
            log_every=log_every,
        )
