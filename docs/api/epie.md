# prism.core.algorithms.epie

Module: prism.core.algorithms.epie
Purpose: Extended Ptychographic Iterative Engine (ePIE) implementation for Fourier ptychography
Dependencies: torch, prism.core.telescope, prism.utils.{image,transforms,metrics}, matplotlib, numpy, pickle

Main Classes:
    - ePIE: Traditional phase retrieval algorithm extending Telescope class

Description:
    Implements the ePIE algorithm for iterative phase retrieval from multiple
    aperture measurements. Unlike deep learning approaches, ePIE directly
    estimates the complex-valued object and probe through physics-based updates.

ePIE Class (extends Telescope):
    Core Properties:
    - obj (og): Complex-valued object estimate (Parameter)
    - probe (Pg): Complex-valued probe/aperture function (Parameter, optional)
    - ground_truth (image): True object for error calculation
    - cum_mask: Accumulated k-space coverage
    - measurements (Im): Cached intensity measurements
    - fix_probe: If True, probe is fixed as known aperture mask

    Algorithm Steps:
    1. Forward model:
        Phi = Og * Pr  (exit wave = object × probe in k-space)
        phi = IFFT(Phi)  (propagate to detector)
        Ir = |phi|²  (predicted intensity)

    2. Measurement constraint:
        Psi = FFT(sqrt(Im) * exp(i*angle(phi)))  (replace magnitude, keep phase)

    3. Updates:
        dOg = lr_obj * Pr* * (Psi - Phi) / max(|Pr|²)
        Og += dOg  (or sum over samples if parallel_update=True)

        dPg = lr_probe * Og* * (Psi - Phi) / max(|Og|²)  (if not fix_probe)

    Properties (computed on-the-fly):
    - Og: Object in k-space (FFT domain)
    - Pg: Probe at center (k-space)
    - Pr: Shifted probes at current sample positions
    - Phi: Exit wave (Og * Pr)
    - phi: Exit wave in real space
    - Psi: Updated exit wave after measurement constraint
    - Im: Measured intensity (cached, computed once per sample)
    - Ir: Reconstructed intensity
    - dOg: Object update gradient

    Key Methods:
    - update_obj(): Apply object update (parallel or serial)
    - update_probe(): Apply probe update (if not fix_probe)
    - update_step(): Combined object and probe update
    - update_cntr(center, center_rec, idx): Set current sample position
    - errors(): Compute RMSE, SSIM, PSNR vs ground truth
    - init_plot(): Create figure for visualization
    - update_plot(): Refresh visualization with current state

Implementation Notes:
    - Object constrained to support mask (zero outside)
    - Real vs complex: Can enforce real-valued if complex_data=False
    - Measurement caching: Im computed once per sample position
    - Single sample mode: Use only center of line for faster updates
    - Parallel update: Sum gradients from all line samples
    - Serial update: Apply gradients sequentially (slower but more stable)

    IMPORTANT: Fixed probe mode (fix_probe=True) is recommended as probe
    is typically known from aperture design. Non-fixed probe requires
    proper Fourier shift implementation.

## Classes

### ePIE

```python
ePIE(n: int = 256, r: float = 10, is_sum: bool = True, sum_pattern: Optional[torch.Tensor] = None, cropping: bool = True, obj_size: Optional[int] = None, snr: float = 100, telescope: Optional[prism.core.telescope.Telescope] = None, req_grad: bool = False, fix_probe: bool = True, lr_obj: float = 1.0, lr_probe: float = 1.0, ground_truth: Optional[torch.Tensor] = None, complex_data: bool = False, parallel_update: bool = True, single_sample: bool = False, blur_image: bool = False)
```

Extended Ptychographic Iterative Engine (ePIE) algorithm.

Physics-based iterative phase retrieval algorithm that alternately updates
object and probe estimates from multiple aperture measurements.

#### Methods

##### `__call__` (forward)

No documentation available.

##### `__init__`

Initialize ePIE algorithm.

Args:
    n: Image size in pixels
    r: Aperture radius in pixels
    is_sum: Whether to sum line samples
    sum_pattern: Custom summation pattern
    cropping: Whether to crop output to object size
    obj_size: Size of object region
    snr: Signal-to-noise ratio in dB
    telescope: Existing Telescope instance to copy parameters from
    req_grad: Whether to compute gradients (for gradient-based ePIE)
    fix_probe: Keep probe fixed as known aperture
    lr_obj: Learning rate for object updates
    lr_probe: Learning rate for probe updates
    ground_truth: Ground truth image for metrics
    complex_data: Allow complex-valued reconstructions
    parallel_update: Sum gradients from all samples vs serial
    single_sample: Use only center sample of line
    blur_image: Apply blur to measurements

##### `add_mask`

Update cumulative k-space coverage mask.

Args:
    centers: Sample center positions
    r: Aperture radius

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

##### `compute_psf`

Compute telescope point spread function.

Args:
    **kwargs: Optional PSF computation parameters (unused for basic telescope)

Returns:
    2D PSF tensor, normalized to peak value of 1

Notes:
    Computes the PSF by:
    1. Creating aperture mask at [0, 0]
    2. Taking FFT (Fraunhofer diffraction to focal plane)
    3. Computing intensity (|·|²)
    4. Normalizing to peak = 1

Example:
    >>> telescope = Telescope(n=512, r=50)
    >>> psf = telescope.compute_psf()
    >>> print(psf.shape, psf.max())
    torch.Size([512, 512]) tensor(1.)

##### `compute_synthetic_aperture`

Compute synthetic aperture by averaging all diffraction patterns in k-space.

This pre-computes all measurements that will be used during training
and averages them in Fourier space to create a synthetic aperture preview.
This is physically equivalent to having all apertures open simultaneously.

Parameters
----------
tensor : Tensor
    Input object image [B, C, H, W]
all_centers : List[List[float]] | Tensor
    ALL aperture centers that will be used during reconstruction
    Shape: [N, 2] where N is number of positions
r : float, optional
    Aperture radius. If None, uses self.r
return_complex : bool, optional
    If True, return complex field; if False, return intensity (default)
batch_size : int, optional
    Process centers in batches to manage memory (default: 100)

Returns
-------
Tensor
    Synthetic aperture reconstruction [1, 1, H, W]
    Real-valued intensity if return_complex=False
    Complex field if return_complex=True

Notes
-----
For large numbers of positions (>1000), this may use significant memory.
The batch_size parameter controls memory usage vs computation time.

Examples
--------
>>> telescope = Telescope(n=512, r=25)
>>> centers = torch.randn(100, 2) * 50  # 100 random positions
>>> obj = torch.ones(1, 1, 512, 512)
>>> synthetic = telescope.compute_synthetic_aperture(obj, centers)
>>> print(synthetic.shape)  # torch.Size([1, 1, 512, 512])

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

##### `errors`

Compute reconstruction error metrics.

Returns:
    Tuple of (RMSE, SSIM, PSNR) comparing reconstruction to ground truth

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

Forward model: simulate telescope measurement.

Applies probe masks in k-space, transforms to real space,
takes magnitude, and optionally adds noise.

Args:
    tensor: Input image
    centers: Sample center positions
    r: Aperture radius
    is_sum: Whether to sum line samples
    sum_pattern: Custom summation pattern

Returns:
    Simulated measurement intensity

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

##### `get_info`

Get instrument information summary.

Returns:
    Dictionary with instrument parameters and characteristics

##### `get_instrument_type`

Return instrument type identifier.

Returns:
    Lowercase instrument class name

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

##### `init_plot`

Initialize visualization figure.

Returns:
    Figure and axes array for plotting

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

##### `mask`

Generate aperture mask using strategy pattern.

Creates a binary mask representing the aperture shape in k-space.
The aperture type (circular, hexagonal, obscured) is determined by
self.aperture set during initialization.

Args:
    center (list or None): Center position [y, x]. Defaults to [0, 0]
    r (float or None): Radius parameter (DEPRECATED, ignored when using
                      non-circular apertures). For backward compatibility
                      with circular apertures only.

Returns:
    Tensor: Boolean mask of shape (n, n), True inside aperture

Example:
    >>> telescope = Telescope(n=256, r=10)
    >>> mask = telescope.mask([0, 0])  # Uses circular aperture
    >>>
    >>> telescope_hex = Telescope(n=256, r=20, aperture='hexagonal')
    >>> mask = telescope_hex.mask([0, 0])  # Uses hexagonal aperture

Notes:
    - The 'r' parameter is deprecated and only affects CircularAperture
    - For other aperture types, size is set during initialization
    - Aperture shape affects PSF and diffraction patterns

##### `mask_batch`

Generate multiple aperture masks efficiently (vectorized).

Creates binary masks for multiple aperture centers at once, using
vectorized operations for better performance than repeated mask() calls.
Uses the aperture strategy pattern to support different aperture shapes.

Args:
    centers: Aperture centers as Tensor or [[y0, x0], [y1, x1], ...]
            Can be from patterns.create_patterns() or list of lists.
    r (float or None): Radius parameter (DEPRECATED, only for backward
                      compatibility with circular apertures). Defaults to self.r

Returns:
    Tensor: Boolean masks of shape (N, n, n) where N = len(centers)

Example:
    >>> telescope = Telescope(n=256, r=10)
    >>> masks = telescope.mask_batch([[0, 0], [10, 10], [20, 20]])
    >>> masks.shape  # (3, 256, 256)
    >>>
    >>> telescope_hex = Telescope(n=256, r=20, aperture='hexagonal')
    >>> masks = telescope_hex.mask_batch([[0, 0], [10, 10]])
    >>> # Hexagonal masks at two positions

Performance:
    ~3-5x faster than list comprehension for N > 10 centers (circular)
    Other aperture types may vary in performance

Notes:
    - CircularAperture has optimized vectorized implementation
    - Other aperture types may fall back to looping (still correct)
    - The 'r' parameter only affects CircularAperture for backward compatibility

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

##### `propagate_extended_source`

Propagate extended source through telescope (partially coherent).

Models spatially extended incoherent sources by decomposing them into
independent coherent point sources. Each point is propagated coherently,
and intensities are summed.

Parameters
----------
source_intensity : Tensor
    Extended source intensity distribution (real, non-negative).
    Shape: (H, W)
propagator : ExtendedSourcePropagator, optional
    Custom extended source propagator. If None, creates one.
n_source_points : int, optional
    Number of source points for decomposition. Default: 500.
    More points = better accuracy but slower computation.
sampling_method : {"grid", "monte_carlo", "adaptive"}, optional
    Source sampling strategy. Default: "adaptive"
fft_cache : FFTCache, optional
    Shared FFT cache for performance optimization
**kwargs
    Additional arguments passed to propagator

Returns
-------
Tensor
    Output intensity distribution

Raises
------
ValueError
    If no aperture is defined for the telescope

Examples
--------
>>> from prism.core.propagators import create_stellar_disk
>>> telescope = Telescope(n=256, r=20)
>>> # Create stellar disk source (need grid for create_stellar_disk)
>>> source = torch.rand(256, 256)
>>> output = telescope.propagate_extended_source(source, n_source_points=200)

Notes
-----
- Uses ExtendedSourcePropagator for partially coherent illumination
- Essential for simulating stellar disks, resolved objects
- More accurate than OTF for non-isoplanatic scenarios
- Computational complexity: O(n_source_points × FFT cost)

##### `propagate_incoherent`

Propagate intensity through telescope using incoherent illumination.

Uses the Optical Transfer Function (OTF) to propagate intensity
distributions, suitable for fully incoherent, spatially extended sources.

Parameters
----------
intensity : Tensor
    Input intensity distribution (real, non-negative).
    Shape: (H, W) or (B, C, H, W)
propagator : OTFPropagator, optional
    Custom OTF propagator. If None, creates one from self.aperture
fft_cache : FFTCache, optional
    Shared FFT cache for performance optimization

Returns
-------
Tensor
    Output intensity distribution

Raises
------
ValueError
    If no aperture is defined for the telescope

Examples
--------
>>> telescope = Telescope(n=256, r=20)
>>> intensity_in = torch.rand(256, 256)
>>> intensity_out = telescope.propagate_incoherent(intensity_in)

Notes
-----
- Uses OTF (autocorrelation of pupil) for propagation
- Energy-conserving within numerical precision
- Suitable for extended astronomical sources (planets, nebulae)

##### `propagate_to_kspace`

Propagate tensor to k-space (Fraunhofer diffraction).

Args:
    tensor (Tensor): Input object image
    crop_size (int or None): Size to crop to before FFT

Returns:
    Tensor: Complex-valued k-space representation

Notes:
    - If cropping enabled, crops/pads to crop_size before FFT
    - Uses self.propagator if available, otherwise falls back to FFT
    - FFT computes Fourier transform (Fraunhofer diffraction)
    - Result is squeezed to remove singleton dimensions

##### `propagate_to_spatial`

Propagate k-space to image plane and compute intensity.

Args:
    tensor (Tensor): Complex k-space field (possibly masked)
    crop_size (int or None): Size to crop to after IFFT

Returns:
    Tensor: Real-valued intensity measurement

Notes:
    - Uses self.propagator if available, otherwise falls back to FFT
    - Takes FFT of k-space (second Fraunhofer transform to detector)
    - Takes absolute value (intensity measurement)
    - Flips dimensions (coordinate convention)
    - Optionally crops to object size
    - Optionally normalizes by max_mean

##### `propagate_with_illumination`

Propagate field/intensity to spatial domain with specified illumination mode.

This is a unified interface for all illumination types, choosing the
appropriate propagation method automatically.

Parameters
----------
tensor : Tensor
    Input field or intensity:
    - For "coherent": Complex field
    - For "incoherent" or "partially_coherent": Real intensity
illumination : {"coherent", "incoherent", "partially_coherent"}, optional
    Illumination mode. Default: "coherent"
crop_size : int, optional
    Size to crop output. Default: self.obj_size
**kwargs
    Additional propagator arguments:
    - For "incoherent": propagator, fft_cache
    - For "partially_coherent": propagator, n_source_points, sampling_method, fft_cache

Returns
-------
Tensor
    Output intensity distribution

Raises
------
ValueError
    If illumination mode is unknown

Examples
--------
>>> telescope = Telescope(n=256, r=20)
>>>
>>> # Coherent (default, existing behavior)
>>> complex_field = torch.randn(256, 256, dtype=torch.cfloat)
>>> output = telescope.propagate_with_illumination(complex_field)
>>>
>>> # Incoherent
>>> intensity = torch.rand(256, 256)
>>> output = telescope.propagate_with_illumination(
...     intensity, illumination="incoherent"
... )
>>>
>>> # Partially coherent
>>> output = telescope.propagate_with_illumination(
...     intensity, illumination="partially_coherent", n_source_points=300
... )

Notes
-----
- "coherent": Uses existing propagate_to_spatial (FFT-based)
- "incoherent": Uses OTFPropagator (fully incoherent)
- "partially_coherent": Uses ExtendedSourcePropagator

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

##### `set_max_mean`

Set normalization factor based on measurement.

Args:
    tensor (Tensor): Object to measure
    center (Tensor): Center position for normalization measurement

Notes:
    - Takes measurement at specified center
    - Stores maximum value for future normalization
    - Used to standardize measurement scales

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

##### `update_cntr`

Update current sample position.

Args:
    center: Measurement center positions
    center_rec: Reconstruction center positions
    idx: Sample index for measurement caching

##### `update_obj`

Apply object update step.

Updates object estimate using gradient dOg. Supports three modes:
- single_sample: Use only center sample of line
- parallel_update: Sum gradients from all samples
- serial: Apply gradients sequentially

##### `update_plot`

Update visualization with current reconstruction state.

##### `update_probe`

Apply probe update step (if not fix_probe).

Updates probe estimate using gradient similar to object update.

##### `update_step`

Perform one ePIE iteration.

Alternately updates object and probe estimates.

##### `validate_field`

Validate and prepare input field.

Args:
    field: Input field tensor

Returns:
    Validated field tensor

Raises:
    ValueError: If field shape doesn't match grid

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

## Functions

### replace_mag

```python
replace_mag(tensor: torch.Tensor, mag: torch.Tensor) -> torch.Tensor
```

Replace the magnitude of a complex tensor with the given magnitude.

Args:
    tensor: Complex tensor whose magnitude will be replaced
    mag: New magnitude tensor

Returns:
    Complex tensor with replaced magnitude but original phase
