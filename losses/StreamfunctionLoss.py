"""Finite-difference cost for streamfunction_cnn.py and train_streamfunction.py.

Physics and notation
--------------------
u = psi_y, v = -psi_x, omega = -(psi_xx + psi_yy).
R = omega_t + u*omega_x + v*omega_y - nu*laplacian(omega).
loss = reduce(lambda_u*(u-u_target)**2 + lambda_v*(v-v_target)**2
              + lambda_momentum*R**2).

The advection velocities come from PREDICTED psi. The targets are observed
physical x/y velocities, in channel order [u, v]. Two signs in the handwritten
schematic are corrected: the v mismatch is (-psi_x - v_target), and
laplacian(omega) = -(psi_xxxx + 2*psi_xxyy + psi_yyyy).

Assumptions: 2D incompressible, unforced flow, constant kinematic viscosity nu,
a fixed uniform Cartesian grid, and uniformly spaced saved frames. W is x;
H is y; both physical coordinates increase with their tensor indices. dx/dy
are the OUTPUT grid spacings; dt is the saved-frame interval, not necessarily
the CFD solver's time step. Use consistent physical units or a consistent
nondimensionalization. Undo independent field normalization before this loss.

Discretization
--------------
* Time: centered difference of omega at t[k-1], t[k+1], divided by 2*dt.
* First/second spatial derivatives: centered three-point formulas.
* Third/fourth spatial derivatives: direct centered five-point formulas.
* Mixed derivatives: tensor products of the corresponding 1D formulas.
All are second-order accurate; a fourth derivative is not fourth-order accuracy.
No automatic differentiation with respect to coordinates is needed. Ordinary
PyTorch operations keep the graph for backward() to differentiate the loss
with respect to the CNN parameters.
Reference: https://docs.pytorch.org/docs/stable/notes/autograd.html

All terms use the SAME valid region: time indices 1:-1, rows/columns 2:-2.
No padding, wrapping, boundary condition, initial-condition term, or solid-body
mask is inferred. Do not use stencils crossing an obstacle. Boundary points
and the two endpoint times have no directly evaluated loss term, although
their psi values can receive gradients through neighboring interior stencils.
Adding any spatially constant C(t) to psi leaves this derivative-only loss
unchanged; an absolute streamfunction level is not identified.

Inputs and integration
----------------------
The loss accepts [B,T,1,H,W], with T >= 3. It also accepts the CNN's ordinary
[B*T,1,H,W] output when window_length=T is EXPLICITLY supplied. Each group of
T frames must be from one trajectory in chronological order, with spacing dt.
Never use arbitrary shuffled frames as the time axis.

The helpers below integrate with the EXISTING training loop without changing
the CNN or train_model. Supply real dataset spacings/viscosity in cost_params:

    from torch.utils.data import DataLoader
    from streamfunction_loss import (
        TemporalTripletDataset, collate_time_windows, cost_function,
    )
    from train_streamfunction import train_model

    # input_frames:     [N_time, C_in, H_in, W_in], one ordered trajectory
    # velocity_targets: [N_time, 2, H_out, W_out], at matching times/coordinates
    dataset = TemporalTripletDataset(input_frames, velocity_targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=True,
                        collate_fn=collate_time_windows)
    # Shuffling here reorders COMPLETE triplets, never their three frames.
    # Prepare model/device/dtype, then optimizer, as in the existing loop.
    history = train_model(
        model=model, train_loader=loader, optimizer=optimizer,
        cost_fn=cost_function, epochs=epochs, device=device,
        cost_params=dict(dx=dx, dy=dy, dt=dt, nu=nu, window_length=3,
                         lambda_u=1.0, lambda_v=1.0, lambda_momentum=1.0,
                         reduction="mean"),
    )

Every original interior time is a triplet center exactly once per epoch with
drop_last=False and ordinary sampling. Adjacent triplets share context frames.
For multiple trajectories, make a dataset for EACH, then use ConcatDataset;
do not concatenate raw time series across trajectory boundaries. Split train/
validation trajectories (or time ranges) BEFORE making overlapping windows.

"mean" matches the existing loop's batch-mean contract. "sum" reproduces the
handwritten sum over the retained points but changes gradient scale with batch
size and should not be used with that loop's mean-cost reporting unchanged.
With fixed triplet/grid sizes, its frame-count-weighted reporting is also a
correct triplet-weighted mean. Loss weights are user choices, not calibrated
defaults. Velocity labels must be on the output grid; no interpolation or
comparison with coarse inputs is performed implicitly.
"""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset, default_collate


# Coefficients are listed as (integer offset, weight); divide by h**order.
# For example, f''' = (-f[-2] + 2*f[-1] - 2*f[1] + f[2]) / (2*h**3).
_STENCILS = {
    0: ((0, 1.0),),
    1: ((-1, -0.5), (1, 0.5)),
    2: ((-1, 1.0), (0, -2.0), (1, 1.0)),
    3: ((-2, -0.5), (-1, 1.0), (1, -1.0), (2, 0.5)),
    4: ((-2, 1.0), (-1, -4.0), (0, 6.0), (1, -4.0), (2, 1.0)),
}


def _scalar(name: str, value: float, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar, not a tensor.")
    value = float(value)
    if not isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        condition = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {condition}.")
    return value


def _as_sequence(
    values: Tensor,
    channels: int,
    window_length: int | None,
    name: str,
) -> Tensor:
    """Keep separate batch and time axes; never infer time from batch order."""
    if not isinstance(values, Tensor) or not values.is_floating_point():
        raise TypeError(f"{name} must be a real floating tensor.")
    if window_length is not None and (
        isinstance(window_length, bool)
        or not isinstance(window_length, int)
        or window_length < 3
    ):
        raise ValueError("window_length must be an integer >= 3.")
    if values.ndim == 4:
        if window_length is None:
            raise ValueError(
                "For flattened CNN output, explicitly supply window_length "
                "and arrange each group as one chronological time window."
            )
        if values.shape[0] % window_length:
            raise ValueError(f"{name}'s frame count must be divisible by window_length.")
        values = values.reshape(-1, window_length, *values.shape[1:])
    elif values.ndim != 5:
        raise ValueError(f"{name} must have shape [B,T,C,H,W] or [B*T,C,H,W].")
    if window_length is not None and values.shape[1] != window_length:
        raise ValueError(f"{name}'s time dimension does not match window_length.")
    if values.shape[0] < 1 or values.shape[1] < 3 or values.shape[2] != channels:
        raise ValueError(f"{name} requires B >= 1, T >= 3, and C = {channels}.")
    if min(values.shape[-2:]) < 5:
        raise ValueError(f"{name} requires at least 5 rows and 5 columns.")
    return values


def centered_derivative(
    field: Tensor, *, order_x: int, order_y: int, dx: float, dy: float,
) -> Tensor:
    """Direct centered spatial derivative on field[..., 2:-2, 2:-2].

    Arbitrary leading dimensions are preserved. The returned last dimensions
    are H-4, W-4, for every derivative, so all residual terms align exactly.
    Mixed derivatives use products of 1D weights, e.g. DxxDyy for psi_xxyy.
    The weights implement Taylor-derived second-order-accurate formulas.
    """
    if not isinstance(field, Tensor) or not field.is_floating_point():
        raise TypeError("field must be a real floating tensor.")
    if field.ndim < 2 or min(field.shape[-2:]) < 5:
        raise ValueError("field must have at least 5 rows and 5 columns.")
    for order in (order_x, order_y):
        if isinstance(order, bool) or not isinstance(order, int) or order not in _STENCILS:
            raise ValueError("Each derivative order must be an integer from 0 to 4.")
    if order_x + order_y > 4:
        raise ValueError("Only derivatives of total order <= 4 are implemented.")
    dx, dy = _scalar("dx", dx), _scalar("dy", dy)
    height, width = field.shape[-2:]
    result = None
    for ox, wx in _STENCILS[order_x]:
        for oy, wy in _STENCILS[order_y]:
            shifted = field[..., 2 + oy:height - 2 + oy, 2 + ox:width - 2 + ox]
            term = (wx * wy) * shifted
            result = term if result is None else result + term
    return result / (dx**order_x * dy**order_y)


# Calculate u, v, omega, and each term of the vorticity-transport residual from predicted psi.
def physics_fields(
    psi_pred: Tensor, *, dx: float, dy: float, dt: float, nu: float,
    window_length: int | None = None,
) -> dict[str, Tensor]:
    """Compute velocities and each term of the vorticity-transport residual.

    Every returned tensor has shape [B,T-2,1,H-4,W-4]. These tensors retain
    their autograd graph. "viscous" means -nu*laplacian(omega), the term on the
    LEFT side of the residual, rather than the diffusion term on its RHS.
    """
    dx, dy = _scalar("dx", dx), _scalar("dy", dy)
    dt, nu = _scalar("dt", dt), _scalar("nu", nu, allow_zero=True)
    psi = _as_sequence(psi_pred, 1, window_length, "psi_pred")

    def derivative(field: Tensor, nx: int, ny: int) -> Tensor:
        return centered_derivative(field, order_x=nx, order_y=ny, dx=dx, dy=dy)

    # 1. Vorticity at ALL times: the centered time difference needs neighbors.
    omega = -(derivative(psi, 2, 0) + derivative(psi, 0, 2))
    time_term = (omega[:, 2:] - omega[:, :-2]) / (2.0 * dt)

    # 2. All remaining operators are evaluated at the interior/center times.
    current = psi[:, 1:-1]
    u = derivative(current, 0, 1)
    v = -derivative(current, 1, 0)

    # 3. Expand advection in third derivatives of predicted psi, as specified.
    omega_x = -(derivative(current, 3, 0) + derivative(current, 1, 2))
    omega_y = -(derivative(current, 2, 1) + derivative(current, 0, 3))
    advection = u * omega_x + v * omega_y

    # 4. The mixed fourth derivative has a PLUS TWO inside the parentheses.
    laplacian_omega = -(
        derivative(current, 4, 0)
        + 2.0 * derivative(current, 2, 2)
        + derivative(current, 0, 4)
    )
    viscous = -nu * laplacian_omega

    # 5. This is the curl-of-momentum (vorticity-transport) residual.
    residual = time_term + advection + viscous
    return {
        "u": u, "v": v, "omega": omega[:, 1:-1],
        "time": time_term, "advection": advection, "viscous": viscous,
        "momentum_residual": residual,
    }


def loss_components(
    psi_pred: Tensor,
    target: Tensor | None = None,
    inputs: Tensor | None = None,
    *,
    dx: float,
    dy: float,
    dt: float,
    nu: float,
    lambda_u: float = 1.0,
    lambda_v: float = 1.0,
    lambda_momentum: float = 1.0,
    window_length: int | None = None,
    reduction: str = "mean",
) -> dict[str, Tensor]:
    """Return scalar total and unweighted u, v, momentum losses.

    target: velocity observations [B,T,2,H,W] or [B*T,2,H,W], matching the
        prediction's grid, time order, device, and floating dtype exactly.
        Required if either velocity weight is nonzero. Zero-weight components
        are reported as zero. With no labels, set both lambda_u=lambda_v=0.
    inputs: accepted for compatibility with train_model; not used as targets.
    reduction: "mean" over B, interior times, and retained spatial points, or
        "sum" over those same points. All three terms use the same region.

    No gradients are detached. These scalar components may also be inspected
    for logging; detach only the logging copies outside this function.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum'.")
    weights = (
        _scalar("lambda_u", lambda_u, allow_zero=True),
        _scalar("lambda_v", lambda_v, allow_zero=True),
        _scalar("lambda_momentum", lambda_momentum, allow_zero=True),
    )
    if not any(weights):
        raise ValueError("At least one loss weight must be positive.")
    psi = _as_sequence(psi_pred, 1, window_length, "psi_pred")
    fields = physics_fields(psi, dx=dx, dy=dy, dt=dt, nu=nu)

    # Validate labels before arithmetic to prevent unintended broadcasting.
    if target is not None:
        target = _as_sequence(target, 2, psi.shape[1], "target")
        expected_shape = (psi.shape[0], psi.shape[1], 2, *psi.shape[-2:])
        if tuple(target.shape) != expected_shape:
            raise ValueError(f"target must match the prediction grid: {expected_shape}.")
        if target.device != psi.device or target.dtype != psi.dtype:
            raise ValueError("target and psi_pred must share device and dtype.")
        target = target[:, 1:-1, :, 2:-2, 2:-2]
    elif weights[0] or weights[1]:
        raise ValueError("Velocity targets are required when a data weight is nonzero.")

    def squared_error(error: Tensor) -> Tensor:
        return error.square().mean() if reduction == "mean" else error.square().sum()

    # 6. Compare reconstructed u and v with co-located reference velocities.
    zero = psi.new_zeros(())
    loss_u = squared_error(fields["u"] - target[:, :, 0:1]) if weights[0] else zero
    loss_v = squared_error(fields["v"] - target[:, :, 1:2]) if weights[1] else zero

    # 7. Penalize the pointwise PDE residual, then combine the three costs.
    loss_momentum = squared_error(fields["momentum_residual"]) if weights[2] else zero
    total = weights[0] * loss_u + weights[1] * loss_v + weights[2] * loss_momentum
    return {"total": total, "u": loss_u, "v": loss_v, "momentum": loss_momentum}


def cost_function_def(
    psi_pred: Tensor,
    target: Tensor | None = None,
    inputs: Tensor | None = None,
    **kwargs: Any,
) -> Tensor:
    """Scalar loss hook: pass this as cost_fn to the existing train_model.

    Required kwargs: dx, dy, dt, nu. See loss_components for optional weights,
    reduction, and temporal layout. Flattened CNN output needs window_length.
    """
    return loss_components(psi_pred, target, inputs, **kwargs)["total"]


class TemporalTripletDataset(Dataset):
    """Sliding [t[k-1], t[k], t[k+1]] windows from ONE uniform time series.

    inputs has shape [N_time,C_in,H_in,W_in]. Optional targets have shape
    [N_time,2,H_out,W_out]. The caller must ensure equal time spacing, matching
    timestamps, and a fixed output grid; timestamps are not encoded in tensors.
    The first/last original times are context, so there are N_time-2 samples.
    """

    def __init__(self, inputs: Tensor, targets: Tensor | None = None) -> None:
        if not isinstance(inputs, Tensor) or inputs.ndim != 4 or inputs.is_complex():
            raise ValueError("inputs must be real [N_time,C_in,H_in,W_in].")
        if inputs.shape[0] < 3 or min(inputs.shape[1:]) < 1:
            raise ValueError("inputs need >= 3 times and nonempty channels/grid.")
        if targets is not None:
            if not isinstance(targets, Tensor) or targets.ndim != 4 or targets.is_complex():
                raise ValueError("targets must be real [N_time,2,H_out,W_out].")
            if targets.shape[:2] != (inputs.shape[0], 2) or min(targets.shape[-2:]) < 5:
                raise ValueError("targets need matching times, 2 channels, and an output grid >= 5x5.")
        self.inputs, self.targets = inputs, targets

    def __len__(self) -> int:
        return self.inputs.shape[0] - 2

    def __getitem__(self, index: int) -> tuple[Tensor, ...]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        window = self.inputs[index:index + 3]
        if self.targets is None:
            return (window,)
        return window, self.targets[index:index + 3]


def collate_time_windows(batch: list[tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
    """Flatten [B,3,C,H,W] into [B*3,C,H,W] for the unchanged CNN/loop.

    Use ONLY with TemporalTripletDataset (or an equivalent triplet dataset),
    and pass window_length=3 to the loss. Order is window-major, time-minor.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    stacked = default_collate(batch)
    if len(stacked) not in (1, 2):
        raise ValueError("Expected triplet inputs, with optional velocity targets.")
    if any(tensor.ndim != 5 or tensor.shape[1] != 3 for tensor in stacked):
        raise ValueError("Each collated tensor must have shape [B,3,C,H,W].")
    return tuple(tensor.flatten(0, 1) for tensor in stacked)
