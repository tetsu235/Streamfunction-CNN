"""An explained training loop for the existing streamfunction_cnn.py.

Keep this file beside streamfunction_cnn.py. Requires Python 3.10+ and PyTorch.
Run the small synthetic demonstration with: python train_streamfunction.py
For multiple independent flow series, call train_multiple_series below. Keep
streamfunction_loss.py beside this file to use that temporal training wrapper.

OVERALL STRUCTURE
1. Dataset/DataLoader supplies batches: inputs [N, C_in, H_in, W_in] and,
   optionally, target tensors. N is the number of samples in the current batch.
2. StreamfunctionCNN applies optional upsampling, convolution blocks, then a
   one-channel output convolution: psi_pred [N, 1, H_out, W_out]. Its output is
   a signed streamfunction field, not the two velocity components.
3. cost_fn(psi_pred, target, inputs, **cost_params) calculates ONE differentiable
   scalar. The implemented physics objective is in streamfunction_loss.py.
4. loss.backward() computes derivatives of this scalar with respect to weights
   and biases. optimizer.step() uses those derivatives to update the network.
5. Repeat for all batches (one epoch), record the mean cost, and repeat epochs.
6. After training, evaluate psi with model.eval(). Converting psi to velocities
   belongs in your later differentiation component.

The model architecture, cost definition, and training procedure are independent:
  streamfunction_cnn.py   -- what the network computes
  cost_function          -- optional placeholder for a custom objective
  streamfunction_loss.py -- implemented finite-difference physics objective
  train_model            -- how the model parameters are updated
  train_multiple_series  -- epochs -> flow series -> triplet batches
  demo_mse_cost / demo   -- executable synthetic checks, not a physics model

For your future Cartesian-grid derivatives, distinguish spatial derivatives
of psi from parameter gradients: backward() differentiates the chosen loss
with respect to model parameters. It does not automatically interpret pixel
indices as x/y coordinates. Keep any loss computations differentiable in
PyTorch; do not detach psi, convert it to NumPy, or call .item() inside the loss.

Reference for the optimization sequence:
https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from random import Random
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from models.CNN2D import ConvBlockConfig, StreamfunctionCNN


def cost_function(
    psi_pred: Tensor,
    target: Tensor | None = None,
    inputs: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """Implement your real objective here, or pass a different callable.

    psi_pred: predicted streamfunction [N, 1, H_out, W_out].
    target: optional data supplied by the loader. Its physical meaning and
        compatibility with psi_pred must be checked inside YOUR cost function.
        For example, velocity observations cannot be subtracted directly from
        psi_pred; they require an appropriate observation/differentiation step.
    inputs: the input fields, already on the model's device and floating dtype.
    kwargs: cost_params, e.g. dx, dy, nu, rho, lambda_data, lambda_momentum.
        Passing a coefficient does not implement its associated physical term.

    Return a real floating scalar tensor (shape []) connected to psi_pred.
    For the epoch-average calculation below, return a mean of per-sample costs.
    You may ignore target for an objective that does not need labeled outputs.
    This hook deliberately raises until you supply the objective.
    """
    raise NotImplementedError(
        "Implement cost_function or pass your own cost_fn to train_model."
    )


def train_model(
    model: nn.Module,
    train_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    cost_fn: Callable[..., Tensor],
    epochs: int,
    device: torch.device | str,
    cost_params: Mapping[str, Any] | None = None,
    *,
    log_every: int = 1,
) -> list[float]:
    """Train the CNN and return one sample-weighted mean cost per epoch.

    Setup order matters: construct model -> move model to device/dtype ->
    construct optimizer from model.parameters() -> call this function.
    The loop checks placement; it does not move an already-optimized model.

    Accepted batches: (inputs, targets), (inputs,), or an inputs tensor alone.
    The loader must be re-iterable across epochs, as a DataLoader is. Inputs and
    optional tensor targets are moved to the model's device/dtype each batch.
    Floating model parameters must share one dtype/device. If cost_params
    contains tensors, prepare their device/dtype yourself before calling.

    cost_fn is called with keyword names psi_pred, target, and inputs, preserving
    the original loop interface. It must return a differentiable batch mean.
    Only the model's parameters are optimized; freeze any excluded parameters
    with requires_grad_(False). Use a regular optimizer such as Adam or SGD;
    closure-based optimizers such as LBFGS need a different loop.

    log_every=1 prints every epoch; 0 disables printing. The returned mean is an
    average of costs encountered during training, while parameters change. It
    is not validation error or an evaluation of the final model on all samples.
    """
    # Setup: validate configuration before beginning any updates.
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer.")
    if not isinstance(log_every, int) or log_every < 0:
        raise ValueError("log_every must be a nonnegative integer.")
    cost_params = {} if cost_params is None else dict(cost_params)
    if {"psi_pred", "target", "inputs"} & cost_params.keys():
        raise ValueError("cost_params must not replace psi_pred, target, or inputs.")

    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    parameters = list(model.parameters())
    trainable = [p for p in parameters if p.requires_grad]
    if not trainable:
        raise ValueError("The model must have trainable parameters.")
    dtype = parameters[0].dtype
    if not dtype.is_floating_point or any(p.dtype != dtype for p in parameters):
        raise ValueError("Use one real floating dtype for the model parameters.")
    if any(t.device != device for t in [*parameters, *model.buffers()]):
        raise ValueError("Move model to device before creating the optimizer.")
    model_ids = {id(p) for p in parameters}
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if not {id(p) for p in trainable} <= optimizer_ids or not optimizer_ids <= model_ids:
        raise ValueError("Optimizer must reference this model's trainable parameters.")

    loss_history: list[float] = []
    for epoch in range(epochs):
        # 1. Begin an epoch and enable training behavior for dropout/BatchNorm.
        #    model.train() changes layer behavior; it does not compute gradients.
        print(f"Epoch {epoch + 1:4d}/{epochs} | training on device {device} | dtype {dtype}")
        model.train()
        epoch_loss_sum = 0.0
        sample_count = 0
        batch_count = 0
        half_batch = 0.5 * len(train_loader) if log_every else 0
        for batch in train_loader:
            # 2. Unpack a batch. Inputs-only batches support future objectives
            #    that use physics/boundary data without a target psi field.
            if batch_count == half_batch:
                print("Halfway through this epoch...")
            
            if isinstance(batch, Tensor):
                inputs, targets = batch, None
            elif isinstance(batch, (tuple, list)) and len(batch) in (1, 2):
                inputs = batch[0]
                targets = batch[1] if len(batch) == 2 else None
            else:
                raise TypeError("Each batch must be inputs, (inputs,), or (inputs, targets).")
            if not isinstance(inputs, Tensor) or inputs.ndim != 4:
                raise ValueError("inputs must be a tensor [N, C_in, H_in, W_in].")
            batch_size = inputs.shape[0]
            if batch_size == 0:
                raise ValueError("A batch must contain at least one sample.")
            if inputs.is_complex():
                raise TypeError("inputs must contain real values.")
            if targets is not None:
                if not isinstance(targets, Tensor) or targets.ndim == 0:
                    raise ValueError("targets must be a batched tensor, or None.")
                if targets.shape[0] != batch_size or targets.is_complex():
                    raise ValueError("targets must be real and match the input batch size.")

            # 3. Match CPU/GPU placement and floating precision. This also
            #    handles float64 data passed to a float32 CNN, for example.
            inputs = inputs.to(device=device, dtype=dtype)
            if targets is not None:
                targets = targets.to(device=device, dtype=dtype)

            # 4. Remove the previous batch's gradients. PyTorch accumulates
            #    gradients by default, but here each batch gets its own update.
            optimizer.zero_grad(set_to_none=True)

            # 5. Forward pass: evaluate psi_pred = model(inputs). Autograd records
            #    operations so the loss can be differentiated through the CNN.
            psi_pred = model(inputs)
            if psi_pred.ndim != 4 or psi_pred.shape[:2] != (batch_size, 1):
                raise ValueError("The model must return psi with shape [N, 1, H, W].")

            # 6. Evaluate the replaceable objective. All physics belongs in this
            #    callable. The loop never assumes targets are streamfunctions.
            loss = cost_fn(
                psi_pred=psi_pred,
                target=targets,
                inputs=inputs,
                **cost_params,
            )
            if not isinstance(loss, Tensor) or loss.ndim != 0 or not loss.is_floating_point():
                raise ValueError("cost_fn must return one real floating scalar tensor.")
            if not torch.isfinite(loss).item():
                raise FloatingPointError("The cost is NaN or infinite; no update was made for this batch.")
            if not loss.requires_grad:
                raise ValueError("The cost has no gradient graph; do not detach it or psi_pred.")

            # 7. Backward pass: calculate d(loss)/d(parameter) and fill .grad.
            #    This computes derivatives; it does not yet update the weights.
            loss.backward()
            gradients = [p.grad for p in trainable if p.grad is not None]
            if not gradients:
                raise ValueError("The cost is not connected to any trainable model parameter.")
            if any(not torch.isfinite(g).all().item() for g in gradients):
                raise FloatingPointError("A gradient is NaN or infinite; no update was made for this batch.")

            # 8. Apply one optimizer update using the gradients just calculated.
            optimizer.step()

            # 9. Store a number for reporting, without retaining the graph.
            #    Weight the batch mean by its actual N, including a short final
            #    batch, so each sample contributes equally to the epoch mean.
            epoch_loss_sum += loss.detach().item() * batch_size
            sample_count += batch_size

        # 10. Finish this epoch, save its average, and optionally print progress.
        #     Empty or exhausted loaders must not silently report zero loss.
        if sample_count == 0:
            raise ValueError("train_loader yielded no samples; use a nonempty, re-iterable loader.")
        mean_loss = epoch_loss_sum / sample_count
        loss_history.append(mean_loss)
        if log_every and ((epoch + 1) % log_every == 0 or epoch + 1 == epochs):
            print(f"Epoch {epoch + 1:4d}/{epochs} | mean training cost: {mean_loss:.6g}")

    return loss_history


@dataclass
class FlowSeries:
    """One independent, chronologically ordered flow trajectory.

    inputs: [N_time, C_in, H_in, W_in], with N_time >= 3.
    targets: optional velocities [N_time, 2, H_out, W_out], channels [u, v].
    dx, dy: OUTPUT-grid spacing. dt: interval between SAVED frames, constant
        within this trajectory. nu: kinematic viscosity, constant within it.
    times: optional timestamps [N_time]. When supplied, their increments are
        checked against dt. Without timestamps, uniform spacing and aligned
        inputs/targets remain the caller's responsibility.

    Different series may have different lengths, dx/dy/dt/nu, and grid sizes
    if the same CNN supports those sizes. Each must still meet the fixed-grid
    and physical assumptions of streamfunction_loss.py. Keep tensors on CPU
    for loading with workers; train_model moves only each batch to the device.
    """

    inputs: Tensor
    dx: float
    dy: float
    dt: float
    nu: float
    targets: Tensor | None = None
    times: Tensor | None = None


def train_multiple_series(
    model: nn.Module,
    series_sources: Sequence[Any],
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device | str,
    *,
    load_series: Callable[[Any], FlowSeries] | None = None,
    batch_size: int = 4,
    cost_fn: Callable[..., Tensor] | None = None,
    cost_params: Mapping[str, Any] | None = None,
    shuffle_series: bool = True,
    shuffle_triplets: bool = True,
    num_workers: int = 0,
    seed: int | None = None,
    log_every: int = 1,
) -> list[float]:
    """Train ONE model across ALL series in every global epoch.

    Loop order: epoch -> series -> batches of complete temporal triplets.
    The same model and optimizer (including Adam's moment estimates) are used
    throughout. Neither is reconstructed or reset when the series changes.
    Every interior time in every series is a triplet center once per epoch.
    The returned list contains one triplet-weighted mean cost per global epoch.
    Longer series therefore contribute more centers; series are not weighted
    equally. With different output grid sizes, each center's spatial MEAN has
    equal reporting weight. These are online training costs, not validation.

    series_sources can be a list of FlowSeries objects already in memory.
    For larger data, pass a list of paths/IDs and load_series(source), returning
    one FlowSeries. The callback is called once per series per epoch. This
    wrapper releases its references before loading the next series; callers
    must avoid retaining/caching the loaded tensors if memory is limited.
    One complete series still needs to fit in RAM with this callback approach;
    it is not a frame-by-frame disk reader. num_workers=0 limits extra copies.

    batch_size counts TRIPLETS. For B triplets, collate_time_windows produces
    [3*B,C,H,W] for the CNN, and the loss reconstructs [B,3,1,H_out,W_out].
    Complete triplets may be shuffled; their internal time order never changes.
    No triplet or batch combines different series, so per-series dt is valid.

    Defaults to streamfunction_loss.cost_function. cost_params holds shared
    loss weights/options; dx, dy, dt, nu come ONLY from each FlowSeries.
    window_length=3 and reduction='mean' are fixed for correct reporting.
    Custom costs must accept those arguments and return a batch mean.
    With no targets, set lambda_u=lambda_v=0 for the default physics loss.
    seed controls series/triplet ordering, not model initialization or dropout.
    Split training/validation trajectories BEFORE constructing windows.

    Example, with tensors and physical metadata already prepared:

        runs = [
            FlowSeries(inputs=x_a, targets=uv_a,
                       dx=dx_a, dy=dy_a, dt=dt_a, nu=nu_a),
            FlowSeries(inputs=x_b, targets=uv_b,
                       dx=dx_b, dy=dy_b, dt=dt_b, nu=nu_b),
        ]
        history = train_multiple_series(
            model, runs, optimizer, epochs=20, device=device,
            batch_size=4, seed=42,
            cost_params=dict(lambda_u=1.0, lambda_v=1.0,
                             lambda_momentum=1.0),
        )

    Example callback for your own .pt files, each saved as a dictionary whose
    keys match FlowSeries (inputs, targets, dx, dy, dt, nu, optional times):

        def read_series(path):
            data = torch.load(path, map_location='cpu', weights_only=True)
            return FlowSeries(**data)

        history = train_multiple_series(
            model, ['run_a.pt', 'run_b.pt'], optimizer,
            epochs=20, device=device, load_series=read_series,
        )

    On Windows, put the training call under if __name__ == '__main__': when
    num_workers > 0. Create and move the model, then create the optimizer, once
    before calling this function, just as with train_model.
    """
    from losses.StreamfunctionLoss import (
        TemporalTripletDataset,
        collate_time_windows,
        cost_function_def as physics_cost,
    )

    # Validate outer-loop options before making any optimizer updates.
    for name, value, minimum in (
        ("epochs", epochs, 1), ("batch_size", batch_size, 1),
        ("num_workers", num_workers, 0), ("log_every", log_every, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if isinstance(series_sources, (str, bytes)) or not isinstance(series_sources, Sequence):
        raise TypeError("series_sources must be a nonempty sequence of FlowSeries objects or source IDs.")
    if not len(series_sources):
        raise ValueError("series_sources must contain at least one series.")
    if load_series is not None and not callable(load_series):
        raise TypeError("load_series must be callable or None.")
    if not isinstance(shuffle_series, bool) or not isinstance(shuffle_triplets, bool):
        raise TypeError("shuffle_series and shuffle_triplets must be bools.")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63
    ):
        raise ValueError("seed must be None or an integer in [0, 2**63).")

    shared_params = dict(cost_params or {})
    reserved = {"dx", "dy", "dt", "nu", "psi_pred", "target", "inputs"}
    if reserved & shared_params.keys():
        raise ValueError("Put dx/dy/dt/nu on each FlowSeries; do not override batch arguments.")
    if shared_params.get("window_length", 3) != 3:
        raise ValueError("TemporalTripletDataset requires window_length=3.")
    if shared_params.get("reduction", "mean") != "mean":
        raise ValueError("Use reduction='mean' for triplet-weighted cost reporting.")
    shared_params.update(window_length=3, reduction="mean")
    objective = physics_cost if cost_fn is None else cost_fn
    if not callable(objective):
        raise TypeError("cost_fn must be callable or None.")

    # Make RNGs once; do not reseed at each series/epoch and repeat the order.
    series_rng = Random(seed)
    loader_rng = None if seed is None else torch.Generator().manual_seed(seed)
    history: list[float] = []

    for epoch in range(epochs):
        order = list(range(len(series_sources)))
        if shuffle_series:
            series_rng.shuffle(order)
        epoch_cost_sum = 0.0
        epoch_triplet_count = 0

        # This is the new outer loop around the existing triplet batch loop.
        for series_index in order:
            source = series_sources[series_index]
            series = source if load_series is None else load_series(source)
            if not isinstance(series, FlowSeries):
                raise TypeError(f"Series {series_index}: expected a FlowSeries object.")
            dataset = TemporalTripletDataset(series.inputs, series.targets)
            params = dict(shared_params)
            for name in ("dx", "dy", "dt", "nu"):
                value = getattr(series, name)
                if (
                    isinstance(value, bool) or not isinstance(value, Real)
                    or not isfinite(value) or value < 0
                    or (name != "nu" and value == 0)
                ):
                    raise ValueError(
                        f"Series {series_index}: dx/dy/dt must be finite and positive; "
                        "nu must be finite and nonnegative."
                    )
                params[name] = float(value)

            if series.times is not None:
                times = torch.as_tensor(series.times, device="cpu")
                if times.is_complex() or times.ndim != 1 or len(times) != len(series.inputs):
                    raise ValueError(f"Series {series_index}: times must be real [N_time].")
                times = times.to(dtype=torch.float64)
                steps = times[1:] - times[:-1]
                expected_steps = torch.full_like(steps, params["dt"])
                if not torch.isfinite(times).all().item() or not torch.allclose(
                    steps, expected_steps, rtol=1e-4, atol=1e-8 * params["dt"],
                ):
                    raise ValueError(f"Series {series_index}: timestamps must increase uniformly by dt.")
                del times, steps, expected_steps

            if num_workers > 0 and any(
                tensor is not None and tensor.device.type != "cpu"
                for tensor in (series.inputs, series.targets)
            ):
                raise ValueError("Use CPU source tensors when num_workers > 0.")

            # A fresh dataset per series prevents windows crossing boundaries.
            loader = DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle_triplets,
                collate_fn=collate_time_windows, drop_last=False,
                num_workers=num_workers, generator=loader_rng,
                persistent_workers=False,
            )
            series_mean = train_model(
                model=model, train_loader=loader, optimizer=optimizer,
                cost_fn=objective, epochs=1, device=device,
                cost_params=params, log_every=0,
            )[0]
            triplet_count = len(dataset)  # N_time - 2; includes a short last batch.
            epoch_cost_sum += series_mean * triplet_count
            epoch_triplet_count += triplet_count

            # With callback loading, release this trajectory BEFORE loading the
            # next one. No trajectories, outputs, or graphs go into history.
            del loader, dataset, series, source

        mean_cost = epoch_cost_sum / epoch_triplet_count
        history.append(mean_cost)
        if log_every and ((epoch + 1) % log_every == 0 or epoch + 1 == epochs):
            print(
                f"Epoch {epoch + 1:4d}/{epochs} | series: {len(series_sources)} | "
                f"triplets: {epoch_triplet_count} | mean training cost: {mean_cost:.6g}"
            )

    return history


