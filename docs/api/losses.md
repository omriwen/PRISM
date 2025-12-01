# prism.models.losses

Module: losses.py
Purpose: Loss functions for progressive PRISM training
Dependencies: torch
Main Classes:
    - LossAggregator: **CRITICAL** - Aggregated loss for progressive training
        Combines loss from old measurements (accumulated mask) with new measurement
        Returns (loss_old, loss_new) for independent convergence checking

Architecture Details:
    LossAggregator (Progressive Loss):
    - Computes loss on both old (accumulated) and new measurements
    - Normalized by zero-loss to make threshold-based stopping robust
    - Supports L1 or L2 loss
    - Returns separate old/new losses for dual convergence criteria
    - Used with TelescopeAggregator to generate measurements through cumulative mask

Usage Pattern:
    from prism.models.losses import LossAggregator
    from prism.core.aggregator import TelescopeAggregator

    # Initialization
    criterion = LossAggregator(loss_type="l1")
    telescope_agg = TelescopeAggregator(n=1024, r=50)

    # Progressive training loop
    for sample_center in sample_centers:
        # Generate measurement through telescope
        measurement = telescope_agg.measure(image, reconstruction, sample_center)

        # Forward pass (decoder-only model)
        output = model()

        # Compute dual loss (old mask + new mask)
        loss_old, loss_new = criterion(output, measurement, telescope_agg, sample_center)

        # Check convergence on both losses
        if loss_old < threshold and loss_new < threshold:
            break  # Both losses converged

        # Optimize
        loss = loss_old + loss_new
        loss.backward()
        optimizer.step()

        # Add new mask to accumulator
        telescope_agg.add_measurement(sample_center)

## Classes

### AMPLossWrapper

```python
AMPLossWrapper(strategy: prism.models.losses.ProgressiveLossStrategy, force_float32: bool = True)
```

Automatic Mixed Precision (AMP) wrapper for loss strategies.

This wrapper ensures loss computations are AMP-compatible by:
1. Casting inputs to float32 for SSIM computation (numerical stability)
2. Running the underlying loss in the appropriate precision
3. Returning results in the original dtype

Using mixed precision training (float16) can provide 1.5-2x speedup on
compatible GPUs while maintaining accuracy.

Args:
    strategy: The underlying loss strategy to wrap
    force_float32: Force float32 for numerically sensitive operations (default: True)

Example:
    >>> from torch.cuda.amp import autocast, GradScaler
    >>> scaler = GradScaler()
    >>>
    >>> # Wrap SSIM for AMP compatibility
    >>> ssim_loss = AMPLossWrapper(SSIMLossStrategy())
    >>>
    >>> with autocast():
    ...     pred = model(input)  # float16
    ...     loss = ssim_loss(pred, target)  # Handled correctly
    >>>
    >>> scaler.scale(loss).backward()

Notes:
    - SSIM and MS-SSIM require float32 for numerical stability
    - L1/L2 losses work fine with float16
    - Automatically detects and handles dtype conversion

#### Methods

##### `__call__` (forward)

Compute loss with AMP-compatible dtype handling.

Args:
    pred: Predicted tensor [B, C, H, W] (may be float16)
    target: Target tensor [B, C, H, W] (may be float16)

Returns:
    Loss value (cast back to original dtype)

##### `__init__`

Initialize AMP loss wrapper.

Args:
    strategy: The loss strategy to wrap
    force_float32: Force float32 precision (recommended for SSIM)

### CompositeLossStrategy

```python
CompositeLossStrategy(losses: Dict[str, Tuple[prism.models.losses.ProgressiveLossStrategy, float]])
```

Weighted combination of multiple loss strategies.

Enables flexible composition of different loss functions with custom weights.
Useful for combining pixel-wise losses (L1/L2) with perceptual losses (SSIM).

Args:
    losses: Dict of {name: (strategy, weight)} pairs

Example:
    >>> # Combine L1 (70%) and SSIM (30%)
    >>> composite = CompositeLossStrategy({
    ...     'l1': (L1LossStrategy(), 0.7),
    ...     'ssim': (SSIMLossStrategy(), 0.3)
    ... })
    >>> pred = torch.rand(1, 1, 128, 128)
    >>> target = torch.rand(1, 1, 128, 128)
    >>> loss = composite(pred, target)
    >>>
    >>> # Triple combination
    >>> composite3 = CompositeLossStrategy({
    ...     'l1': (L1LossStrategy(), 0.5),
    ...     'l2': (L2LossStrategy(), 0.3),
    ...     'ssim': (SSIMLossStrategy(), 0.2)
    ... })

Notes:
    - Weights must sum to 1.0 (within tolerance of ±0.01)
    - All strategies are evaluated on the same pred/target pair
    - Loss name is constructed from weighted components (e.g., "0.7*l1+0.3*ssim")
    - Useful for balancing different aspects of reconstruction quality

#### Methods

##### `__call__` (forward)

Compute weighted sum of all component losses.

Args:
    pred: Predicted tensor [B, C, H, W]
    target: Target tensor [B, C, H, W]

Returns:
    Weighted combination of all losses

##### `__init__`

Initialize composite loss strategy.

Args:
    losses: Dict of {name: (strategy, weight)} pairs

Raises:
    ValueError: If weights don't sum to approximately 1.0

Example:
    >>> losses = {
    ...     'pixel': (L1LossStrategy(), 0.6),
    ...     'structural': (SSIMLossStrategy(), 0.4)
    ... }
    >>> composite = CompositeLossStrategy(losses)

### FastSSIMLossStrategy

```python
FastSSIMLossStrategy(window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0, max_size: int = 256, downsample_mode: str = 'bilinear')
```

Optimized SSIM loss with automatic downsampling for large images.

This strategy provides significant speedup (5-10x) for large images by
downsampling before computing SSIM. Since SSIM measures structural similarity,
downsampling preserves the essential quality metrics while dramatically
reducing computation time.

Args:
    window_size: Gaussian window size (default: 11)
    sigma: Gaussian sigma (default: 1.5)
    data_range: Expected range of values (default: 1.0)
    max_size: Maximum image size before downsampling (default: 256)
        Images larger than this will be downsampled to this size.
    downsample_mode: Interpolation mode for downsampling (default: 'bilinear')

Example:
    >>> # Standard SSIM for 1024x1024: ~3ms
    >>> # Fast SSIM for 1024x1024: ~0.3ms (10x speedup)
    >>> loss_fn = FastSSIMLossStrategy(max_size=256)
    >>> pred = torch.rand(1, 1, 1024, 1024)
    >>> target = torch.rand(1, 1, 1024, 1024)
    >>> loss = loss_fn(pred, target)

Notes:
    - Preserves SSIM quality metrics (structural similarity is scale-invariant)
    - 5-10x speedup for images > 512x512
    - Recommended for training where exact SSIM isn't critical
    - For evaluation, use standard SSIMLossStrategy for exact values

#### Methods

##### `__call__` (forward)

Compute SSIM loss with automatic downsampling for large images.

Args:
    pred: Predicted tensor [B, C, H, W]
    target: Target tensor [B, C, H, W]

Returns:
    DSSIM loss scalar in range [0, 0.5]

##### `__init__`

Initialize fast SSIM loss strategy.

Args:
    window_size: Gaussian window size (must be odd)
    sigma: Gaussian standard deviation
    data_range: Expected range of input values
    max_size: Maximum dimension before downsampling (default: 256)
    downsample_mode: Interpolation mode ('bilinear', 'bicubic', 'area')

### L1LossStrategy

```python
L1LossStrategy(/, *args, **kwargs)
```

L1 (Mean Absolute Error) loss strategy.

Computes the mean absolute difference between predictions and targets.
Also known as MAE. Robust to outliers compared to L2.

Formula: L1 = mean(|pred - target|)

Example:
    >>> loss_fn = L1LossStrategy()
    >>> pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    >>> loss = loss_fn(pred, target)
    >>> print(f"L1 Loss: {loss:.4f}")
    L1 Loss: 0.5000

Notes:
    - Suitable for general-purpose reconstruction tasks
    - Less sensitive to outliers than L2
    - Gradient magnitude is constant (not proportional to error)

#### Methods

##### `__call__` (forward)

Compute L1 loss.

### L2LossStrategy

```python
L2LossStrategy(/, *args, **kwargs)
```

L2 (Mean Squared Error) loss strategy.

Computes the mean squared difference between predictions and targets.
Also known as MSE. Penalizes large errors more than L1.

Formula: L2 = mean((pred - target)²)

Example:
    >>> loss_fn = L2LossStrategy()
    >>> pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> target = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
    >>> loss = loss_fn(pred, target)
    >>> print(f"L2 Loss: {loss:.4f}")
    L2 Loss: 0.2500

Notes:
    - Standard choice for many reconstruction tasks
    - More sensitive to outliers than L1 (quadratic penalty)
    - Gradient magnitude proportional to error (larger errors → larger gradients)

#### Methods

##### `__call__` (forward)

Compute L2 loss.

### LossAggregator

```python
LossAggregator(loss_type: Literal['l1', 'l2', 'ssim', 'ms-ssim', 'composite'] = 'l1', new_weight: Optional[float] = None, f_weight: Optional[float] = None, loss_weights: Optional[Dict[str, float]] = None, **strategy_kwargs: Any) -> None
```

Aggregated loss for progressive PRISM training.

This is a critical component of PRISM. It computes loss on both:
1. Old measurements (accumulated mask from previous samples)
2. New measurement (current sample only)

The dual loss allows independent convergence checking:
- loss_old: Ensures model still fits all previous measurements
- loss_new: Ensures model fits the new measurement

For L1/L2 losses: Both losses are normalized by "zero loss" (loss of zeros vs target)
to make threshold-based stopping robust to varying measurement intensities.

For SSIM losses: Uses DSSIM = (1 - SSIM) / 2 formulation, operating in measurement space
(same as L1/L2, but using structural similarity instead of pixel-wise difference).

Args:
    loss_type: Type of loss function ("l1", "l2", "ssim", or "ms-ssim") (default: "l1")
    new_weight: Weight for new loss (unused, kept for compatibility)
    f_weight: Frequency-domain weight (unused, kept for compatibility)

Attributes:
    running_loss1: Running average of loss_old (for monitoring)
    running_loss2: Running average of loss_new (for monitoring)

Returns:
    Tuple[Tensor, Tensor]: (loss_old, loss_new)
        - loss_old: Normalized loss on accumulated measurements
        - loss_new: Normalized loss on new measurement

Example:
    >>> # L1 loss in measurement space
    >>> criterion = LossAggregator(loss_type="l1")
    >>> telescope_agg = TelescopeAggregator(n=1024, r=50)
    >>>
    >>> # First measurement (only new loss is meaningful)
    >>> measurement = telescope_agg.measure(image, reconstruction, center=[0, 0])
    >>> output = model()
    >>> loss_old, loss_new = criterion(output, measurement, telescope_agg, [0, 0])
    >>> print(f"Old: {loss_old:.4f}, New: {loss_new:.4f}")
    >>>
    >>> # Add to accumulator
    >>> telescope_agg.add_measurement([0, 0])
    >>>
    >>> # Second measurement (both losses are meaningful)
    >>> measurement = telescope_agg.measure(image, reconstruction, center=[10, 10])
    >>> output = model()
    >>> loss_old, loss_new = criterion(output, measurement, telescope_agg, [10, 10])
    >>> # loss_old checks fit to first measurement
    >>> # loss_new checks fit to second measurement
    >>>
    >>> # SSIM loss in measurement space (same as L1, different metric)
    >>> criterion_ssim = LossAggregator(loss_type="ssim")
    >>> measurement = telescope_agg.measure(image, reconstruction, center=[10, 10])
    >>> output = model()
    >>> loss_old, loss_new = criterion_ssim(output, measurement, telescope_agg, [10, 10])

Notes:
    - All losses operate in measurement space using TelescopeAggregator
    - L1/L2: Pixel-wise differences, normalized by zero-loss for consistent thresholds
    - SSIM: Structural similarity in measurement space, uses DSSIM = (1 - SSIM) / 2, range [0, 0.5]
    - Returns tuple of two losses for dual convergence checking

#### Methods

##### `__call__` (forward)

No documentation available.

##### `__init__`

Initialize aggregated loss.

Args:
    loss_type: One of ["l1", "l2", "ssim", "ms-ssim", "composite"]
    new_weight: Weight for new loss (unused, kept for compatibility)
    f_weight: Frequency-domain weight (unused, kept for compatibility)
    loss_weights: For composite losses, dict of {loss_name: weight}
                 Example: {'l1': 0.7, 'ssim': 0.3}
    **strategy_kwargs: Additional arguments for loss strategies
                      (e.g., window_size, sigma for SSIM)

Raises:
    ValueError: If loss_type is not one of the supported types

Example:
    >>> # Simple L1 loss
    >>> loss = LossAggregator(loss_type='l1')
    >>>
    >>> # SSIM with custom window
    >>> loss = LossAggregator(loss_type='ssim', window_size=7, sigma=1.0)
    >>>
    >>> # Composite: 70% L1 + 30% SSIM
    >>> loss = LossAggregator(
    ...     loss_type='composite',
    ...     loss_weights={'l1': 0.7, 'ssim': 0.3}
    ... )

##### `add_module`

Add a child module to the current module.

The module can be accessed as an attribute using the given name.

Args:
    name (str): name of the child module. The child module can be
        accessed from this module using the given name
    module (Module): child module to be added to the module.

##### `apply`

Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

Typical use includes initializing the parameters of a model
(see also :ref:`nn-init-doc`).

Args:
    fn (:class:`Module` -> None): function to be applied to each submodule

Returns:
    Module: self

Example::

    >>> @torch.no_grad()
    >>> def init_weights(m):
    >>>     print(m)
    >>>     if type(m) == nn.Linear:
    >>>         m.weight.fill_(1.0)
    >>>         print(m.weight)
    >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    >>> net.apply(init_weights)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Linear(in_features=2, out_features=2, bias=True)
    Parameter containing:
    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )

##### `bfloat16`

Casts all floating point parameters and buffers to ``bfloat16`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `buffers`

Return an iterator over module buffers.

Args:
    recurse (bool): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module.

Yields:
    torch.Tensor: module buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for buf in model.buffers():
    >>>     print(type(buf), buf.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `children`

Return an iterator over immediate children modules.

Yields:
    Module: a child module

##### `compile`

Compile this Module's forward using :func:`torch.compile`.

This Module's `__call__` method is compiled and all arguments are passed as-is
to :func:`torch.compile`.

See :func:`torch.compile` for details on the arguments for this function.

##### `cpu`

Move all model parameters and buffers to the CPU.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `cuda`

Move all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on GPU while being optimized.

.. note::
    This method modifies the module in-place.

Args:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `double`

Casts all floating point parameters and buffers to ``double`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `eval`

Set the module in evaluation mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e. whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

See :ref:`locally-disable-grad-doc` for a comparison between
`.eval()` and several similar mechanisms that may be confused with it.

Returns:
    Module: self

##### `extra_repr`

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

##### `float`

Casts all floating point parameters and buffers to ``float`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `forward`

Compute aggregated loss on old and new measurements.

Args:
    inputs: Model output (reconstructed image) [B, C, H, W]
    target: Target measurements [2, C, H, W]
            target[0] = old accumulated measurements
            target[1] = new measurement
    telescope: TelescopeAggregator instance for generating measurements from reconstruction
    center: Center coordinates for new measurement

Returns:
    Tuple[Tensor, Tensor]: (loss_old, loss_new)
        - L1/L2: Normalized loss on accumulated/new measurements
        - SSIM: DSSIM = (1 - SSIM) / 2, range [0, 0.5]

Notes:
    - All losses operate in measurement space
    - Telescope transforms reconstruction into measurements (diffraction patterns)
    - L1/L2: Pixel-wise differences, normalized by zero-loss
    - SSIM: Structural similarity metric on measurements
    - If telescope is None, inputs are duplicated (for simple testing)

##### `get_buffer`

Return the buffer given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the buffer
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.Tensor: The buffer referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not a
        buffer

##### `get_extra_state`

Return any extra state to include in the module's state_dict.

Implement this and a corresponding :func:`set_extra_state` for your module
if you need to store extra state. This function is called when building the
module's `state_dict()`.

Note that extra state should be picklable to ensure working serialization
of the state_dict. We only provide backwards compatibility guarantees
for serializing Tensors; other objects may break backwards compatibility if
their serialized pickled form changes.

Returns:
    object: Any extra state to store in the module's state_dict

##### `get_parameter`

Return the parameter given by ``target`` if it exists, otherwise throw an error.

See the docstring for ``get_submodule`` for a more detailed
explanation of this method's functionality as well as how to
correctly specify ``target``.

Args:
    target: The fully-qualified string name of the Parameter
        to look for. (See ``get_submodule`` for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Parameter: The Parameter referenced by ``target``

Raises:
    AttributeError: If the target string references an invalid
        path or resolves to something that is not an
        ``nn.Parameter``

##### `get_submodule`

Return the submodule given by ``target`` if it exists, otherwise throw an error.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
            )
            (linear): Linear(in_features=100, out_features=200, bias=True)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` which has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To check whether or not we have the ``linear`` submodule, we
would call ``get_submodule("net_b.linear")``. To check whether
we have the ``conv`` submodule, we would call
``get_submodule("net_b.net_c.conv")``.

The runtime of ``get_submodule`` is bounded by the degree
of module nesting in ``target``. A query against
``named_modules`` achieves the same result, but it is O(N) in
the number of transitive modules. So, for a simple check to see
if some submodule exists, ``get_submodule`` should always be
used.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)

Returns:
    torch.nn.Module: The submodule referenced by ``target``

Raises:
    AttributeError: If at any point along the path resulting from
        the target string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `half`

Casts all floating point parameters and buffers to ``half`` datatype.

.. note::
    This method modifies the module in-place.

Returns:
    Module: self

##### `ipu`

Move all model parameters and buffers to the IPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on IPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `load_state_dict`

Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

If :attr:`strict` is ``True``, then
the keys of :attr:`state_dict` must exactly match the keys returned
by this module's :meth:`~torch.nn.Module.state_dict` function.

.. warning::
    If :attr:`assign` is ``True`` the optimizer must be created after
    the call to :attr:`load_state_dict` unless
    :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

Args:
    state_dict (dict): a dict containing parameters and
        persistent buffers.
    strict (bool, optional): whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    assign (bool, optional): When set to ``False``, the properties of the tensors
        in the current module are preserved whereas setting it to ``True`` preserves
        properties of the Tensors in the state dict. The only
        exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`
        for which the value from the module is preserved. Default: ``False``

Returns:
    ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
        * ``missing_keys`` is a list of str containing any keys that are expected
            by this module but missing from the provided ``state_dict``.
        * ``unexpected_keys`` is a list of str containing the keys that are not
            expected by this module but present in the provided ``state_dict``.

Note:
    If a parameter or buffer is registered as ``None`` and its corresponding key
    exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
    ``RuntimeError``.

##### `modules`

Return an iterator over all modules in the network.

Yields:
    Module: a module in the network

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.modules()):
    ...     print(idx, '->', m)

    0 -> Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    )
    1 -> Linear(in_features=2, out_features=2, bias=True)

##### `mtia`

Move all model parameters and buffers to the MTIA.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on MTIA while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `named_buffers`

Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

Args:
    prefix (str): prefix to prepend to all buffer names.
    recurse (bool, optional): if True, then yields buffers of this module
        and all submodules. Otherwise, yields only buffers that
        are direct members of this module. Defaults to True.
    remove_duplicate (bool, optional): whether to remove the duplicated buffers in the result. Defaults to True.

Yields:
    (str, torch.Tensor): Tuple containing the name and buffer

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, buf in self.named_buffers():
    >>>     if name in ['running_var']:
    >>>         print(buf.size())

##### `named_children`

Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

Yields:
    (str, Module): Tuple containing a name and child module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, module in model.named_children():
    >>>     if name in ['conv4', 'conv5']:
    >>>         print(module)

##### `named_modules`

Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

Args:
    memo: a memo to store the set of modules already added to the result
    prefix: a prefix that will be added to the name of the module
    remove_duplicate: whether to remove the duplicated module instances in the result
        or not

Yields:
    (str, Module): Tuple of name and module

Note:
    Duplicate modules are returned only once. In the following
    example, ``l`` will be returned only once.

Example::

    >>> l = nn.Linear(2, 2)
    >>> net = nn.Sequential(l, l)
    >>> for idx, m in enumerate(net.named_modules()):
    ...     print(idx, '->', m)

    0 -> ('', Sequential(
      (0): Linear(in_features=2, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=2, bias=True)
    ))
    1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

##### `named_parameters`

Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

Args:
    prefix (str): prefix to prepend to all parameter names.
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.
    remove_duplicate (bool, optional): whether to remove the duplicated
        parameters in the result. Defaults to True.

Yields:
    (str, Parameter): Tuple containing the name and parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for name, param in self.named_parameters():
    >>>     if name in ['bias']:
    >>>         print(param.size())

##### `parameters`

Return an iterator over module parameters.

This is typically passed to an optimizer.

Args:
    recurse (bool): if True, then yields parameters of this module
        and all submodules. Otherwise, yields only parameters that
        are direct members of this module.

Yields:
    Parameter: module parameter

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> for param in model.parameters():
    >>>     print(type(param), param.size())
    <class 'torch.Tensor'> (20L,)
    <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

##### `register_backward_hook`

Register a backward hook on the module.

This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
the behavior of this function will change in future versions.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_buffer`

Add a buffer to the module.

This is typically used to register a buffer that should not be
considered a model parameter. For example, BatchNorm's ``running_mean``
is not a parameter, but is part of the module's state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting :attr:`persistent` to ``False``. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module's
:attr:`state_dict`.

Buffers can be accessed as attributes using given names.

Args:
    name (str): name of the buffer. The buffer can be accessed
        from this module using the given name
    tensor (Tensor or None): buffer to be registered. If ``None``, then operations
        that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
        the buffer is **not** included in the module's :attr:`state_dict`.
    persistent (bool): whether the buffer is part of this module's
        :attr:`state_dict`.

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> self.register_buffer('running_mean', torch.zeros(num_features))

##### `register_forward_hook`

Register a forward hook on the module.

The hook will be called every time after :func:`forward` has computed an output.

If ``with_kwargs`` is ``False`` or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
output. It can modify the input inplace but it will not have effect on
forward since this is called after :func:`forward` is called. The hook
should have the following signature::

    hook(module, args, output) -> None or modified output

If ``with_kwargs`` is ``True``, the forward hook will be passed the
``kwargs`` given to the forward function and be expected to return the
output possibly modified. The hook should have the following signature::

    hook(module, args, kwargs, output) -> None or modified output

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If ``True``, the provided ``hook`` will be fired
        before all existing ``forward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``forward`` hooks registered with
        :func:`register_module_forward_hook` will fire before all hooks
        registered by this method.
        Default: ``False``
    with_kwargs (bool): If ``True``, the ``hook`` will be passed the
        kwargs given to the forward function.
        Default: ``False``
    always_call (bool): If ``True`` the ``hook`` will be run regardless of
        whether an exception is raised while calling the Module.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_forward_pre_hook`

Register a forward pre-hook on the module.

The hook will be called every time before :func:`forward` is invoked.


If ``with_kwargs`` is false or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the ``forward``. The hook can modify the
input. User can either return a tuple or a single modified value in the
hook. We will wrap the value into a tuple if a single value is returned
(unless that value is already a tuple). The hook should have the
following signature::

    hook(module, args) -> None or modified input

If ``with_kwargs`` is true, the forward pre-hook will be passed the
kwargs given to the forward function. And if the hook modifies the
input, both the args and kwargs should be returned. The hook should have
the following signature::

    hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

Args:
    hook (Callable): The user defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``forward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``forward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``forward_pre`` hooks registered with
        :func:`register_module_forward_pre_hook` will fire before all
        hooks registered by this method.
        Default: ``False``
    with_kwargs (bool): If true, the ``hook`` will be passed the kwargs
        given to the forward function.
        Default: ``False``

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_hook`

Register a backward hook on the module.

The hook will be called every time the gradients with respect to a module are computed, and its firing rules are as follows:

    1. Ordinarily, the hook fires when the gradients are computed with respect to the module inputs.
    2. If none of the module inputs require gradients, the hook will fire when the gradients are computed
       with respect to module outputs.
    3. If none of the module outputs require gradients, then the hooks will not fire.

The hook should have the following signature::

    hook(module, grad_input, grad_output) -> tuple(Tensor) or None

The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of :attr:`grad_input` in
subsequent computations. :attr:`grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs or outputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward`` hooks on
        this :class:`torch.nn.Module`. Note that global
        ``backward`` hooks registered with
        :func:`register_module_full_backward_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_full_backward_pre_hook`

Register a backward pre-hook on the module.

The hook will be called every time the gradients for the module are computed.
The hook should have the following signature::

    hook(module, grad_output) -> tuple[Tensor] or None

The :attr:`grad_output` is a tuple. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the output that will be used in place of :attr:`grad_output` in
subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
all non-Tensor arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

.. warning ::
    Modifying inputs inplace is not allowed when using backward hooks and
    will raise an error.

Args:
    hook (Callable): The user-defined hook to be registered.
    prepend (bool): If true, the provided ``hook`` will be fired before
        all existing ``backward_pre`` hooks on this
        :class:`torch.nn.Module`. Otherwise, the provided
        ``hook`` will be fired after all existing ``backward_pre`` hooks
        on this :class:`torch.nn.Module`. Note that global
        ``backward_pre`` hooks registered with
        :func:`register_module_full_backward_pre_hook` will fire before
        all hooks registered by this method.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_post_hook`

Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, incompatible_keys) -> None

The ``module`` argument is the current module that this hook is registered
on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
is a ``list`` of ``str`` containing the missing keys and
``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

The given incompatible_keys can be modified inplace if needed.

Note that the checks performed when calling :func:`load_state_dict` with
``strict=True`` are affected by modifications the hook makes to
``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
set of keys will result in an error being thrown when ``strict=True``, and
clearing out both missing and unexpected keys will avoid an error.

Returns:
    :class:`torch.utils.hooks.RemovableHandle`:
        a handle that can be used to remove the added hook by calling
        ``handle.remove()``

##### `register_load_state_dict_pre_hook`

Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

Arguments:
    hook (Callable): Callable hook that will be invoked before
        loading the state dict.

##### `register_module`

Alias for :func:`add_module`.

##### `register_parameter`

Add a parameter to the module.

The parameter can be accessed as an attribute using given name.

Args:
    name (str): name of the parameter. The parameter can be accessed
        from this module using the given name
    param (Parameter or None): parameter to be added to the module. If
        ``None``, then operations that run on parameters, such as :attr:`cuda`,
        are ignored. If ``None``, the parameter is **not** included in the
        module's :attr:`state_dict`.

##### `register_state_dict_post_hook`

Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, state_dict, prefix, local_metadata) -> None

The registered hooks can modify the ``state_dict`` inplace.

##### `register_state_dict_pre_hook`

Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

It should have the following signature::
    hook(module, prefix, keep_vars) -> None

The registered hooks can be used to perform pre-processing before the ``state_dict``
call is made.

##### `requires_grad_`

Change if autograd should record operations on parameters in this module.

This method sets the parameters' :attr:`requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).

See :ref:`locally-disable-grad-doc` for a comparison between
`.requires_grad_()` and several similar mechanisms that may be confused with it.

Args:
    requires_grad (bool): whether autograd should record operations on
                          parameters in this module. Default: ``True``.

Returns:
    Module: self

##### `set_extra_state`

Set extra state contained in the loaded `state_dict`.

This function is called from :func:`load_state_dict` to handle any extra state
found within the `state_dict`. Implement this function and a corresponding
:func:`get_extra_state` for your module if you need to store extra state within its
`state_dict`.

Args:
    state (dict): Extra state from the `state_dict`

##### `set_submodule`

Set the submodule given by ``target`` if it exists, otherwise throw an error.

.. note::
    If ``strict`` is set to ``False`` (default), the method will replace an existing submodule
    or create a new submodule if the parent module exists. If ``strict`` is set to ``True``,
    the method will only attempt to replace an existing submodule and throw an error if
    the submodule does not exist.

For example, let's say you have an ``nn.Module`` ``A`` that
looks like this:

.. code-block:: text

    A(
        (net_b): Module(
            (net_c): Module(
                (conv): Conv2d(3, 3, 3)
            )
            (linear): Linear(3, 3)
        )
    )

(The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
submodule ``net_b``, which itself has two submodules ``net_c``
and ``linear``. ``net_c`` then has a submodule ``conv``.)

To override the ``Conv2d`` with a new submodule ``Linear``, you
could call ``set_submodule("net_b.net_c.conv", nn.Linear(1, 1))``
where ``strict`` could be ``True`` or ``False``

To add a new submodule ``Conv2d`` to the existing ``net_b`` module,
you would call ``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))``.

In the above if you set ``strict=True`` and call
``set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)``, an AttributeError
will be raised because ``net_b`` does not have a submodule named ``conv``.

Args:
    target: The fully-qualified string name of the submodule
        to look for. (See above example for how to specify a
        fully-qualified string.)
    module: The module to set the submodule to.
    strict: If ``False``, the method will replace an existing submodule
        or create a new submodule if the parent module exists. If ``True``,
        the method will only attempt to replace an existing submodule and throw an error
        if the submodule doesn't already exist.

Raises:
    ValueError: If the ``target`` string is empty or if ``module`` is not an instance of ``nn.Module``.
    AttributeError: If at any point along the path resulting from
        the ``target`` string the (sub)path resolves to a non-existent
        attribute name or an object that is not an instance of ``nn.Module``.

##### `share_memory`

See :meth:`torch.Tensor.share_memory_`.

##### `state_dict`

Return a dictionary containing references to the whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.
Parameters and buffers set to ``None`` are not included.

.. note::
    The returned object is a shallow copy. It contains references
    to the module's parameters and buffers.

.. warning::
    Currently ``state_dict()`` also accepts positional arguments for
    ``destination``, ``prefix`` and ``keep_vars`` in order. However,
    this is being deprecated and keyword arguments will be enforced in
    future releases.

.. warning::
    Please avoid the use of argument ``destination`` as it is not
    designed for end-users.

Args:
    destination (dict, optional): If provided, the state of module will
        be updated into the dict and the same object is returned.
        Otherwise, an ``OrderedDict`` will be created and returned.
        Default: ``None``.
    prefix (str, optional): a prefix added to parameter and buffer
        names to compose the keys in state_dict. Default: ``''``.
    keep_vars (bool, optional): by default the :class:`~torch.Tensor` s
        returned in the state dict are detached from autograd. If it's
        set to ``True``, detaching will not be performed.
        Default: ``False``.

Returns:
    dict:
        a dictionary containing a whole state of the module

Example::

    >>> # xdoctest: +SKIP("undefined vars")
    >>> module.state_dict().keys()
    ['bias', 'weight']

##### `to`

Move and/or cast the parameters and buffers.

This can be called as

.. function:: to(device=None, dtype=None, non_blocking=False)
   :noindex:

.. function:: to(dtype, non_blocking=False)
   :noindex:

.. function:: to(tensor, non_blocking=False)
   :noindex:

.. function:: to(memory_format=torch.channels_last)
   :noindex:

Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
floating point or complex :attr:`dtype`\ s. In addition, this method will
only cast the floating point or complex parameters and buffers to :attr:`dtype`
(if given). The integral parameters and buffers will be moved
:attr:`device`, if that is given, but with dtypes unchanged. When
:attr:`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

.. note::
    This method modifies the module in-place.

Args:
    device (:class:`torch.device`): the desired device of the parameters
        and buffers in this module
    dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
        the parameters and buffers in this module
    tensor (torch.Tensor): Tensor whose dtype and device are the desired
        dtype and device for all parameters and buffers in this module
    memory_format (:class:`torch.memory_format`): the desired memory
        format for 4D parameters and buffers in this module (keyword
        only argument)

Returns:
    Module: self

Examples::

    >>> # xdoctest: +IGNORE_WANT("non-deterministic")
    >>> linear = nn.Linear(2, 2)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]])
    >>> linear.to(torch.double)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1913, -0.3420],
            [-0.5113, -0.2325]], dtype=torch.float64)
    >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
    >>> gpu1 = torch.device("cuda:1")
    >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
    >>> cpu = torch.device("cpu")
    >>> linear.to(cpu)
    Linear(in_features=2, out_features=2, bias=True)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.1914, -0.3420],
            [-0.5112, -0.2324]], dtype=torch.float16)

    >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
    >>> linear.weight
    Parameter containing:
    tensor([[ 0.3741+0.j,  0.2382+0.j],
            [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
    >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
    tensor([[0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j],
            [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

##### `to_empty`

Move the parameters and buffers to the specified device without copying storage.

Args:
    device (:class:`torch.device`): The desired device of the parameters
        and buffers in this module.
    recurse (bool): Whether parameters and buffers of submodules should
        be recursively moved to the specified device.

Returns:
    Module: self

##### `train`

Set the module in training mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e., whether they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
etc.

Args:
    mode (bool): whether to set training mode (``True``) or evaluation
                 mode (``False``). Default: ``True``.

Returns:
    Module: self

##### `type`

Casts all parameters and buffers to :attr:`dst_type`.

.. note::
    This method modifies the module in-place.

Args:
    dst_type (type or string): the desired type

Returns:
    Module: self

##### `xpu`

Move all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.

.. note::
    This method modifies the module in-place.

Arguments:
    device (int, optional): if specified, all parameters will be
        copied to that device

Returns:
    Module: self

##### `zero_grad`

Reset gradients of all model parameters.

See similar function under :class:`torch.optim.Optimizer` for more context.

Args:
    set_to_none (bool): instead of setting to zero, set the grads to None.
        See :meth:`torch.optim.Optimizer.zero_grad` for details.

### MSSSIMLossStrategy

```python
MSSSIMLossStrategy(window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0, weights: Optional[List[float]] = None)
```

Multi-Scale SSIM (MS-SSIM) loss strategy.

Computes SSIM at multiple scales for better perceptual quality assessment.
More robust to scaling and viewing distance than single-scale SSIM.

Args:
    window_size: Gaussian window size (default: 11)
    sigma: Gaussian sigma (default: 1.5)
    data_range: Expected range of values (default: 1.0)
    weights: Scale weights (default: [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

Example:
    >>> loss_fn = MSSSIMLossStrategy()
    >>> pred = torch.rand(1, 1, 256, 256)
    >>> target = torch.rand(1, 1, 256, 256)
    >>> loss = loss_fn(pred, target)
    >>> print(f"MS-SSIM Loss: {loss:.4f}")

Notes:
    - Requires minimum image size of 176×176 for 5 scales
    - Automatically adjusts to fewer scales if image is too small
    - Default weights from Wang et al. (2003) sum to 1.0
    - More computationally expensive than single-scale SSIM

#### Methods

##### `__call__` (forward)

Compute MS-SSIM loss with input validation.

Args:
    pred: Predicted tensor [B, C, H, W]
    target: Target tensor [B, C, H, W]

Returns:
    MS-DSSIM loss scalar in range [0, 0.5]

Raises:
    ValueError: If input is too small for the number of scales

##### `__init__`

Initialize MS-SSIM loss strategy.

Args:
    window_size: Gaussian window size
    sigma: Gaussian standard deviation
    data_range: Expected range of input values
    weights: Per-scale weights (default: standard 5-scale weights)

### ProgressiveLossStrategy

```python
ProgressiveLossStrategy(/, *args, **kwargs)
```

Abstract base class for progressive loss functions.

All loss strategies implement a common interface for computing
loss between predictions and targets. This enables flexible composition
and extension of loss functions without modifying core training logic.

Subclasses must implement:
    - __call__: Compute loss between prediction and target
    - name: Property returning loss function name for logging

Example:
    >>> class CustomLoss(ProgressiveLossStrategy):
    ...     def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
    ...         return torch.abs(pred - target).mean()
    ...
    ...     @property
    ...     def name(self) -> str:
    ...         return "custom"

#### Methods

##### `__call__` (forward)

Compute loss between prediction and target.

Args:
    pred: Predicted tensor [B, C, H, W]
    target: Target tensor [B, C, H, W]

Returns:
    Scalar loss value (differentiable)

Notes:
    - Implementation should return a scalar tensor for backpropagation
    - Loss should be differentiable w.r.t. pred
    - Both inputs should be on the same device

### SSIMLossStrategy

```python
SSIMLossStrategy(window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0)
```

SSIM (Structural Similarity) loss strategy.

Computes structural similarity between predictions and targets,
considering luminance, contrast, and structure. Returns DSSIM as loss.

Formula: Loss = (1 - SSIM) / 2, where SSIM ∈ [-1, 1]
Range: [0, 0.5] where 0 = perfect match, 0.5 = maximum dissimilarity

Args:
    window_size: Gaussian window size (default: 11)
    sigma: Gaussian sigma (default: 1.5)
    data_range: Expected range of values (default: 1.0 for [0,1] images)

Example:
    >>> loss_fn = SSIMLossStrategy(window_size=11, sigma=1.5)
    >>> pred = torch.rand(1, 1, 128, 128)
    >>> target = torch.rand(1, 1, 128, 128)
    >>> loss = loss_fn(pred, target)
    >>> print(f"SSIM Loss: {loss:.4f}")

Notes:
    - Better perceptual quality than L1/L2 for images
    - Considers structural information, not just pixel differences
    - Window is cached per device for efficiency
    - Requires images larger than window_size in each dimension

#### Methods

##### `__call__` (forward)

Compute SSIM loss with device-aware window caching.

Args:
    pred: Predicted tensor [B, C, H, W]
    target: Target tensor [B, C, H, W]

Returns:
    DSSIM loss scalar in range [0, 0.5]

##### `__init__`

Initialize SSIM loss strategy.

Args:
    window_size: Gaussian window size (must be odd)
    sigma: Gaussian standard deviation
    data_range: Expected range of input values

## Functions

### get_retry_loss_type

```python
get_retry_loss_type(original_loss: str, retry_num: int) -> str
```

Get alternative loss function for retry attempt.

Cycles through LOSS_RETRY_ORDER to provide different loss landscapes
when a sample fails to converge with the original loss function.

Parameters
----------
original_loss : str
    Original loss type used ("l1", "l2", "ssim", "ms-ssim")
retry_num : int
    Current retry attempt number (1-based)

Returns
-------
str
    Alternative loss type to use for this retry

Examples
--------
>>> get_retry_loss_type("l1", 1)
'ssim'
>>> get_retry_loss_type("l1", 2)
'l2'
>>> get_retry_loss_type("ssim", 1)
'l2'

Notes
-----
The function cycles through LOSS_RETRY_ORDER starting from the
original loss type. This ensures each retry tries a different
loss landscape that might help the sample converge.
