"""An explained training loop for the existing streamfunction_cnn.py.

Keep this file beside streamfunction_cnn.py. Requires Python 3.10+ and PyTorch.
Run the small synthetic demonstration with: python train_streamfunction.py

OVERALL STRUCTURE
1. Dataset/DataLoader supplies batches: inputs [N, C_in, H_in, W_in] and,
   optionally, target tensors. N is the number of samples in the current batch.
2. StreamfunctionCNN applies optional upsampling, convolution blocks, then a
   one-channel output convolution: psi_pred [N, 1, H_out, W_out]. Its output is
   a signed streamfunction field, not the two velocity components.
3. cost_fn(psi_pred, target, inputs, **cost_params) calculates ONE differentiable
   scalar. The actual physical objective is yours to implement.
4. loss.backward() computes derivatives of this scalar with respect to weights
   and biases. optimizer.step() uses those derivatives to update the network.
5. Repeat for all batches (one epoch), record the mean cost, and repeat epochs.
6. After training, evaluate psi with model.eval(). Converting psi to velocities
   belongs in your later differentiation component.

The model architecture, cost definition, and training procedure are independent:
  streamfunction_cnn.py   -- what the network computes
  cost_function          -- what a good prediction means (unimplemented)
  train_model            -- how the model parameters are updated
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

from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset
from losses.StreamfunctionLoss import cost_function_def, TemporalTripletDataset

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

    cost_fn = cost_function_def(psi_pred=psi_pred, target=target, inputs=inputs, **kwargs)

    return cost_fn

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
        model.train()
        epoch_loss_sum = 0.0
        sample_count = 0

        for batch in train_loader:
            # 2. Unpack a batch. Inputs-only batches support future objectives
            #    that use physics/boundary data without a target psi field.
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


def demo_mse_cost(
    psi_pred: Tensor,
    target: Tensor | None = None,
    inputs: Tensor | None = None,
    *,
    scale: float = 1.0,
) -> Tensor:
    """Synthetic demo only: MSE against same-shape artificial psi targets.

    This is separate from your unimplemented physical objective. Exact shape
    checking prevents broadcasting psi against two velocity channels by mistake.
    """
    if target is None or target.shape != psi_pred.shape:
        raise ValueError("Demo targets must have exactly the same shape as psi_pred.")
    return scale * (psi_pred - target).square().mean()


def demo() -> list[float]:
    """Run a small executable example; the artificial data have no flow physics."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Same architecture as the example in streamfunction_cnn.py:
    # channels 2 -> 32 -> 64 -> 64 -> 32 -> 1; kernels 5, 3, 3, 3, then 1.
    # stride=1, padding='same', and no pooling preserve the grid resolution.
    model = StreamfunctionCNN(
        in_channels=2,
        blocks=[
            ConvBlockConfig(out_channels=channels, kernel_size=kernel)
            for channels, kernel in [(32, 5), (64, 3), (64, 3), (32, 3)]
        ],
        upsample=None,
        output_conv_kwargs={"kernel_size": 1, "padding": 0, "bias": True},
    ).to(device=device, dtype=torch.float32)

    # Construct the optimizer AFTER finalizing the model's device and dtype.
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    inputs = torch.randn(17, 2, 12, 16)
    artificial_psi = 0.5 * inputs[:, :1] - 0.25 * inputs[:, 1:2]
    train_loader = DataLoader(
        TemporalTripletDataset(inputs, artificial_psi), batch_size=4, shuffle=True,
    )
    print("Synthetic demonstration only; the physics cost is still unimplemented.")
    return train_model(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        cost_fn=demo_mse_cost,  # Later: cost_fn=cost_function (after implementing it).
        epochs=5,             # Example only; choose epochs for your actual data.
        device=device,
        cost_params={"scale": 1.0},
        log_every=1,
    )


if __name__ == "__main__":
    demo()
