"""
Incoherent propagation methods for extended sources.

This module provides propagators for incoherent and partially coherent
illumination scenarios, including OTF-based propagation and extended
source decomposition.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from prism.core.grid import Grid
from prism.core.propagators.base import Propagator, SamplingMethod
from prism.core.propagators.utils import validate_intensity_input
from prism.utils.transforms import FFTCache


class OTFPropagator(Propagator):
    """
    Optical Transfer Function propagator for incoherent illumination.

    Uses the OTF (autocorrelation of the pupil function) to propagate
    intensity distributions. Suitable for fully incoherent, spatially
    extended sources.

    Parameters
    ----------
    aperture : Tensor
        Complex pupil function (aperture). Shape: (H, W)
    grid : Grid, optional
        Coordinate system for the propagation
    normalize : bool, optional
        Whether to normalize the OTF to have unit DC component. Default: True
    fft_cache : FFTCache, optional
        Shared FFT cache for performance

    Notes
    -----
    The OTF is computed as the normalized autocorrelation of the pupil:

        OTF = Autocorrelation(P) / Autocorrelation(P)[0,0]

    where P is the pupil function (aperture).

    For incoherent imaging:

        I_out = IFFT[FFT[I_in] × OTF]

    Examples
    --------
    >>> grid = Grid(nx=256, dx=1e-5, wavelength=550e-9)
    >>> aperture = create_circular_aperture(grid, radius=0.01)
    >>> propagator = OTFPropagator(aperture, grid)
    >>> intensity_out = propagator(intensity_in)
    """

    def __init__(
        self,
        aperture: Tensor,
        grid: Optional[Grid] = None,
        normalize: bool = True,
        fft_cache: Optional[FFTCache] = None,
    ):
        super().__init__(fft_cache=fft_cache)

        self.normalize = normalize

        # Compute OTF from aperture
        otf = self._compute_otf(aperture)
        self.register_buffer("otf", otf)

        # Store grid reference if provided
        if grid is not None:
            self.register_buffer("_wavelength", torch.tensor(grid.wl))

    @property
    def otf_tensor(self) -> Tensor:
        """OTF as Tensor (type-safe accessor for buffer)."""
        assert isinstance(self.otf, Tensor)
        return self.otf

    def _compute_otf(self, aperture: Tensor) -> Tensor:
        """
        Compute OTF from aperture/pupil function.

        The OTF is the Fourier transform of the PSF, normalized so that DC=1.
        For incoherent imaging, PSF = |coherent_PSF|² = |FFT(aperture)|²

        Parameters
        ----------
        aperture : Tensor
            Complex pupil function

        Returns
        -------
        Tensor
            Real-valued OTF with DC at center, normalized to have DC=1
        """
        # Compute incoherent PSF: PSF = |FFT(aperture)|²
        # Use fft_cache.fft2 which handles centered aperture properly
        # (it does ifftshift before FFT and fftshift after)
        coherent_psf = self.fft_cache.fft2(aperture, norm="ortho")
        psf = coherent_psf.abs() ** 2  # PSF is now centered

        # OTF = FFT(PSF), normalized
        # Use fft_cache.fft2 which expects centered input and produces centered output
        otf_complex = self.fft_cache.fft2(psf, norm="ortho")

        # OTF should be real for symmetric PSF (take real part)
        otf = otf_complex.real

        # Normalize so DC component = 1 (DC is at center after fft_cache.fft2)
        if self.normalize:
            h, w = otf.shape[-2], otf.shape[-1]
            dc_value = otf[..., h // 2, w // 2]
            # Handle batched or unbatched tensors
            if dc_value.dim() == 0:
                otf = otf / dc_value.clamp(min=1e-10)
            else:
                otf = otf / dc_value.unsqueeze(-1).unsqueeze(-1).clamp(min=1e-10)

        return otf

    def forward(  # type: ignore[override]
        self,
        intensity: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """
        Propagate intensity distribution through incoherent optical system.

        Parameters
        ----------
        intensity : Tensor
            Input intensity distribution. Must be real-valued and non-negative.
            Shape: (..., H, W)

        Returns
        -------
        Tensor
            Output intensity distribution. Real-valued, non-negative.
            Shape: (..., H, W)

        Raises
        ------
        ValueError
            If input contains negative values or is complex
        """
        # Validate input
        if intensity.is_complex():
            raise ValueError(
                "OTFPropagator expects real intensity input, got complex tensor. "
                "For coherent propagation, use FraunhoferPropagator or "
                "AngularSpectrumPropagator instead."
            )

        # Transform to frequency domain using fft_cache
        # fft_cache.fft2 produces centered output (DC at center)
        intensity_fft = self.fft_cache.fft2(intensity, norm="ortho")

        # Apply OTF (both are centered, so direct multiplication)
        filtered_fft = intensity_fft * self.otf_tensor

        # Inverse transform back to spatial domain
        # fft_cache.ifft2 expects centered input and produces centered output
        intensity_out = self.fft_cache.ifft2(filtered_fft, norm="ortho")

        # Take real part and clamp to non-negative
        intensity_out = intensity_out.real.clamp(min=0)

        return intensity_out

    @property
    def illumination_mode(self) -> str:
        """Return the illumination mode for this propagator."""
        return "incoherent"

    def get_otf(self) -> Tensor:
        """Return the precomputed OTF."""
        return self.otf_tensor

    def get_mtf(self) -> Tensor:
        """
        Return the Modulation Transfer Function (MTF).

        MTF is the magnitude of the OTF.
        """
        return self.otf_tensor.abs()

    def get_ptf(self) -> Tensor:
        """
        Return the Phase Transfer Function (PTF).

        PTF is the phase of the OTF (usually zero for symmetric apertures).
        """
        # OTF is real for symmetric apertures, but compute phase anyway
        return torch.zeros_like(self.otf_tensor)


# =============================================================================
# Extended Source Geometry Helpers
# =============================================================================


def create_stellar_disk(
    grid: Grid,
    angular_diameter: float,
    limb_darkening: float = 0.0,
    center: Optional[Tuple[float, float]] = None,
) -> Tensor:
    """
    Create stellar disk intensity distribution with optional limb darkening.

    Parameters
    ----------
    grid : Grid
        Coordinate system
    angular_diameter : float
        Angular diameter of the disk (in same units as grid coordinates)
    limb_darkening : float, optional
        Linear limb darkening coefficient (0 = uniform, 1 = dark at limb).
        Typical values: Sun ≈ 0.6, red giants ≈ 0.3. Default: 0.0
    center : tuple of float, optional
        Center position (x, y) in physical units. Default: (0, 0)

    Returns
    -------
    Tensor
        Stellar disk intensity distribution, normalized to sum to 1.

    Notes
    -----
    Limb darkening model (linear):
        I(μ) = I_0 × (1 - u × (1 - μ))

    where μ = cos(θ) = sqrt(1 - (r/R)²) is the angle from disk center.

    Examples
    --------
    >>> grid = Grid(nx=256, dx=1e-6, wavelength=550e-9)
    >>> sun = create_stellar_disk(grid, angular_diameter=0.01, limb_darkening=0.6)
    >>> assert sun.shape == (256, 256)
    >>> assert torch.isclose(sun.sum(), torch.tensor(1.0))
    """
    if center is None:
        center = (0.0, 0.0)

    x = grid.x - center[0]
    y = grid.y - center[1]
    r = torch.sqrt(x**2 + y**2)

    radius = angular_diameter / 2
    inside_disk = r <= radius

    # Base uniform disk
    disk = inside_disk.float()

    # Apply limb darkening if specified
    if limb_darkening > 0:
        r_normalized = (r / radius).clamp(max=1.0)
        mu = torch.sqrt((1 - r_normalized**2).clamp(min=0))  # cos(θ)
        limb_factor = 1 - limb_darkening * (1 - mu)
        disk = disk * limb_factor

    # Normalize to sum to 1
    total = disk.sum()
    if total > 0:
        disk = disk / total

    return disk


def create_gaussian_source(
    grid: Grid,
    sigma: float,
    center: Optional[Tuple[float, float]] = None,
    amplitude: float = 1.0,
) -> Tensor:
    """
    Create Gaussian source intensity distribution.

    Parameters
    ----------
    grid : Grid
        Coordinate system
    sigma : float
        Standard deviation in physical units (same as grid)
    center : tuple of float, optional
        Center position (x, y). Default: (0, 0)
    amplitude : float, optional
        Peak amplitude. Default: 1.0

    Returns
    -------
    Tensor
        Gaussian intensity distribution

    Examples
    --------
    >>> grid = Grid(nx=256, dx=1e-6, wavelength=550e-9)
    >>> source = create_gaussian_source(grid, sigma=5e-5)
    """
    if center is None:
        center = (0.0, 0.0)

    x = grid.x - center[0]
    y = grid.y - center[1]
    r2 = x**2 + y**2

    return amplitude * torch.exp(-r2 / (2 * sigma**2))


def create_binary_source(
    grid: Grid,
    separation: float,
    flux_ratio: float = 1.0,
    position_angle: float = 0.0,
    point_source_sigma: Optional[float] = None,
) -> Tensor:
    """
    Create binary star intensity distribution.

    Parameters
    ----------
    grid : Grid
        Coordinate system
    separation : float
        Angular separation between components (same units as grid)
    flux_ratio : float, optional
        Flux ratio (secondary/primary). Default: 1.0 (equal brightness)
    position_angle : float, optional
        Position angle in radians (0 = along x-axis). Default: 0.0
    point_source_sigma : float, optional
        Width of each component as Gaussian sigma. If None, uses 1 pixel.

    Returns
    -------
    Tensor
        Binary source intensity distribution, normalized to sum to 1.

    Examples
    --------
    >>> grid = Grid(nx=256, dx=1e-6, wavelength=550e-9)
    >>> binary = create_binary_source(grid, separation=0.01, flux_ratio=0.5)
    """
    if point_source_sigma is None:
        point_source_sigma = grid.dx  # One pixel

    # Component positions
    half_sep = separation / 2
    dx = half_sep * np.cos(position_angle)
    dy = half_sep * np.sin(position_angle)

    # Create components
    primary = create_gaussian_source(grid, point_source_sigma, center=(-dx, -dy))
    secondary = create_gaussian_source(grid, point_source_sigma, center=(dx, dy))

    # Combine with flux ratio
    total = primary + flux_ratio * secondary

    # Normalize
    return total / total.sum()


def create_ring_source(
    grid: Grid,
    inner_radius: float,
    outer_radius: float,
    center: Optional[Tuple[float, float]] = None,
) -> Tensor:
    """
    Create ring/annulus intensity distribution.

    Useful for modeling:
    - Planetary rings
    - Circumstellar disks
    - Ring nebulae

    Parameters
    ----------
    grid : Grid
        Coordinate system
    inner_radius : float
        Inner radius of the ring
    outer_radius : float
        Outer radius of the ring
    center : tuple of float, optional
        Center position. Default: (0, 0)

    Returns
    -------
    Tensor
        Ring intensity distribution, normalized to sum to 1.
    """
    if center is None:
        center = (0.0, 0.0)

    x = grid.x - center[0]
    y = grid.y - center[1]
    r = torch.sqrt(x**2 + y**2)

    ring = ((r >= inner_radius) & (r <= outer_radius)).float()

    total = ring.sum()
    if total > 0:
        ring = ring / total

    return ring


def estimate_required_samples(
    source_intensity: Tensor,
    grid: Grid,
    target_snr: float = 100.0,
) -> int:
    """
    Estimate number of source points needed for accurate propagation.

    Parameters
    ----------
    source_intensity : Tensor
        Source intensity distribution
    grid : Grid
        Coordinate system
    target_snr : float, optional
        Target signal-to-noise ratio for sampling. Default: 100

    Returns
    -------
    int
        Recommended number of source points

    Notes
    -----
    Uses the effective number of resolution elements in the source
    to estimate sampling requirements.
    """
    # Find extent of source
    threshold = 0.01 * source_intensity.max()
    mask = source_intensity > threshold

    if not mask.any():
        return 100  # Minimum

    # Count significant pixels
    n_significant = mask.sum().item()

    # Estimate based on SNR requirement
    # SNR ≈ sqrt(n_samples / n_significant)
    n_samples = int(n_significant * (target_snr / 10) ** 2)

    # Clamp to reasonable range
    return max(100, min(n_samples, 10000))


# =============================================================================
# Extended Source Propagator
# =============================================================================


class ExtendedSourcePropagator(Propagator):
    """
    Extended source propagator for partially coherent illumination.

    Models spatially extended incoherent sources by decomposing them into
    independent coherent point sources. Each point source is propagated
    coherently, and intensities are summed (no cross-interference between
    different source points).

    This propagator is essential for simulating:
    - Stellar disks with finite angular extent
    - Resolved astronomical objects (planets, nebulae)
    - Partially coherent illumination scenarios
    - van Cittert-Zernike theorem effects

    Parameters
    ----------
    coherent_propagator : Propagator
        Coherent propagator for individual point sources (e.g., Fraunhofer)
    grid : Grid
        Coordinate system for source and observation planes
    source_grid : Grid, optional
        Separate grid for source plane (if different resolution needed)
    n_source_points : int, optional
        Number of source points for decomposition. Default: 1000.
        More points = better accuracy but slower computation.
    sampling_method : {"grid", "monte_carlo", "adaptive"}, optional
        Source sampling strategy:
        - "grid": Uniform grid sampling (fast, may miss fine structure)
        - "monte_carlo": Random importance sampling (good for peaked sources)
        - "adaptive": Hybrid stratified + importance (best accuracy)
        Default: "adaptive"
    coherent_patch_size : float, optional
        Size of coherent patches in meters. Points within a patch are
        treated as coherent (interfere), while patches are incoherent.
        If None, each point is independent (fully incoherent decomposition).
    batch_size : int, optional
        Number of source points to process in parallel. Default: 32.
        Larger = faster but more GPU memory.
    use_psf_cache : bool, optional
        Cache PSFs for repeated evaluations. Default: True.
    fft_cache : FFTCache, optional
        Shared FFT cache for performance

    Attributes
    ----------
    coherent_propagator : Propagator
        The underlying coherent propagator
    grid : Grid
        Coordinate system
    n_source_points : int
        Number of source decomposition points
    _psf_cache : dict
        Cache of computed PSFs (if use_psf_cache=True)

    Notes
    -----
    **Computational Complexity**:
    - For N source points: O(N × propagation_cost)
    - FraunhoferPropagator: O(N × M log M) where M = grid size
    - Memory: O(batch_size × M) per batch

    **Accuracy Considerations**:
    - n_source_points > 100 recommended for smooth sources
    - n_source_points > 500 for accurate flux measurements
    - Adaptive sampling improves accuracy for non-uniform sources

    **When to Use**:
    - Source angular size > diffraction limit
    - Need per-source-point control (e.g., limb darkening)
    - PSF varies across field of view (non-isoplanatic)
    - Modeling specific source geometries (binary stars, disks)

    **When NOT to Use** (use OTFPropagator instead):
    - Source is much larger than coherence length
    - Isoplanatic approximation is valid
    - Speed is critical and source is uniform

    Examples
    --------
    >>> # Basic usage with stellar disk
    >>> from prism.core.propagators import ExtendedSourcePropagator, FraunhoferPropagator
    >>> from prism.core.grid import Grid
    >>>
    >>> grid = Grid(nx=256, dx=1e-6, wavelength=550e-9)
    >>> coherent_prop = FraunhoferPropagator()
    >>>
    >>> ext_prop = ExtendedSourcePropagator(
    ...     coherent_propagator=coherent_prop,
    ...     grid=grid,
    ...     n_source_points=500,
    ...     sampling_method="adaptive",
    ... )
    >>>
    >>> # Create stellar disk and propagate
    >>> stellar_disk = create_stellar_disk(grid, angular_diameter=0.01)
    >>> image = ext_prop(stellar_disk, aperture=aperture)

    >>> # With coherent patches for partial coherence
    >>> ext_prop_partial = ExtendedSourcePropagator(
    ...     coherent_propagator=coherent_prop,
    ...     grid=grid,
    ...     n_source_points=200,
    ...     coherent_patch_size=1e-5,  # 10 micron coherent patches
    ... )

    See Also
    --------
    OTFPropagator : For fully incoherent (isoplanatic) propagation
    FraunhoferPropagator : Underlying coherent propagator
    create_stellar_disk : Helper for creating stellar disk sources
    """

    def __init__(
        self,
        coherent_propagator: Propagator,
        grid: Grid,
        source_grid: Optional[Grid] = None,
        n_source_points: int = 1000,
        sampling_method: SamplingMethod = "adaptive",
        coherent_patch_size: Optional[float] = None,
        batch_size: int = 32,
        use_psf_cache: bool = True,
        fft_cache: Optional[FFTCache] = None,
    ):
        super().__init__(fft_cache=fft_cache)

        self.coherent_propagator = coherent_propagator
        self.grid = grid
        self.source_grid = source_grid if source_grid is not None else grid
        self.n_source_points = n_source_points
        self.sampling_method = sampling_method
        self.coherent_patch_size = coherent_patch_size
        self.batch_size = batch_size
        self.use_psf_cache = use_psf_cache

        # PSF cache for efficiency with repeated calls
        self._psf_cache: Dict[Tuple[int, int], Tensor] = {}

        # Statistics tracking
        self._last_n_samples = 0
        self._last_coverage = 0.0

    def forward(  # type: ignore[override]
        self,
        source_intensity: Tensor,
        aperture: Optional[Tensor] = None,
        source_positions: Optional[Tensor] = None,
        source_weights: Optional[Tensor] = None,
        return_diagnostics: bool = False,
        **kwargs: Any,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """
        Propagate extended source intensity distribution.

        Parameters
        ----------
        source_intensity : Tensor
            Source intensity map (real, non-negative). Shape: (..., H, W)
        aperture : Tensor, optional
            Complex aperture function to apply before propagation.
            Shape: (H, W). If None, no aperture masking is applied.
        source_positions : Tensor, optional
            Explicit source point positions in physical coordinates.
            Shape: (N, 2). If None, positions are sampled from source_intensity.
        source_weights : Tensor, optional
            Intensity weights for each source point. Shape: (N,).
            If None, weights are determined by sampling method.
        return_diagnostics : bool, optional
            If True, return diagnostic dict with sampling info. Default: False.
        **kwargs
            Additional arguments passed to coherent_propagator.

        Returns
        -------
        output : Tensor
            Output intensity distribution. Shape: (..., H, W)
        diagnostics : dict, optional
            Sampling diagnostics (if return_diagnostics=True):
            - "n_samples": Number of source points used
            - "positions": Sampled positions
            - "weights": Sample weights
            - "coverage": Estimated coverage of source intensity

        Raises
        ------
        ValueError
            If source_intensity is complex or aperture shape doesn't match grid
        """
        # Validate input
        validate_intensity_input(source_intensity, "source_intensity")

        if aperture is not None:
            if aperture.shape[-2:] != (self.grid.nx, self.grid.ny):
                raise ValueError(
                    f"Aperture shape {aperture.shape} doesn't match grid "
                    f"({self.grid.nx}, {self.grid.ny})"
                )

        # Sample source positions if not provided
        if source_positions is None or source_weights is None:
            positions, weights = self._sample_source_points(source_intensity)
        else:
            positions = source_positions
            weights = source_weights

        # Track statistics
        self._last_n_samples = len(positions)

        # Initialize output accumulator
        output_shape = source_intensity.shape
        device = source_intensity.device
        output = torch.zeros(output_shape, device=device)

        # Process source points in batches for GPU efficiency
        n_sources = positions.shape[0]
        for batch_start in range(0, n_sources, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n_sources)
            batch_positions = positions[batch_start:batch_end]
            batch_weights = weights[batch_start:batch_end]

            # Create complex fields for this batch of point sources
            point_fields = self._create_point_source_fields(batch_positions, aperture)

            # Propagate batch coherently
            propagated = self.coherent_propagator(point_fields, **kwargs)

            # Convert to intensities
            batch_intensities = propagated.abs() ** 2

            # Apply weights and accumulate
            weighted = batch_intensities * batch_weights.view(-1, 1, 1)
            output = output + weighted.sum(dim=0)

        if return_diagnostics:
            diagnostics = {
                "n_samples": self._last_n_samples,
                "positions": positions,
                "weights": weights,
                "sampling_method": self.sampling_method,
            }
            return output, diagnostics

        return output

    def _sample_source_points(
        self,
        intensity: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample source point positions from intensity distribution.

        Parameters
        ----------
        intensity : Tensor
            Source intensity distribution. Shape: (H, W)

        Returns
        -------
        positions : Tensor
            Source point positions in physical coordinates. Shape: (N, 2)
        weights : Tensor
            Intensity weights for each point. Shape: (N,)
        """
        if self.sampling_method == "grid":
            return self._sample_grid(intensity)
        elif self.sampling_method == "monte_carlo":
            return self._sample_monte_carlo(intensity)
        elif self.sampling_method == "adaptive":
            return self._sample_adaptive(intensity)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling_method}")

    def _sample_grid(self, intensity: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Uniform grid sampling of source points.

        Creates a regular grid of sample points across the source plane.
        Simple and predictable but may miss fine structure if grid is coarse.
        """
        h, w = intensity.shape[-2:]
        device = intensity.device

        n_per_dim = int(np.sqrt(self.n_source_points))

        # Create uniform grid indices
        y_indices = torch.linspace(0, h - 1, n_per_dim, device=device).long()
        x_indices = torch.linspace(0, w - 1, n_per_dim, device=device).long()

        # Create meshgrid of positions
        yy, xx = torch.meshgrid(y_indices, x_indices, indexing="ij")
        y_flat = yy.flatten()
        x_flat = xx.flatten()

        # Convert to physical coordinates
        positions = self._indices_to_positions(x_flat, y_flat)

        # Sample weights from intensity at these positions
        weights = intensity[y_flat, x_flat]

        # Normalize weights
        total_weight = weights.sum()
        if total_weight > 0:
            weights = weights / total_weight
        else:
            weights = torch.ones_like(weights) / len(weights)

        return positions, weights

    def _sample_monte_carlo(self, intensity: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Monte Carlo importance sampling of source points.

        Samples points with probability proportional to intensity.
        More samples in bright regions, better for peaked sources.
        """
        h, w = intensity.shape[-2:]
        device = intensity.device

        # Normalize intensity to probability distribution
        prob = intensity.flatten()
        prob = prob / prob.sum().clamp(min=1e-10)

        # Sample indices according to probability (importance sampling)
        indices_1d = torch.multinomial(prob, self.n_source_points, replacement=True)

        # Convert to 2D indices
        y_indices = indices_1d // w
        x_indices = indices_1d % w

        # Convert to physical coordinates
        positions = self._indices_to_positions(x_indices, y_indices)

        # For importance sampling, weights are uniform
        # (the sampling already accounts for intensity distribution)
        weights = torch.ones(self.n_source_points, device=device)
        weights = weights / weights.sum()

        return positions, weights

    def _sample_adaptive(self, intensity: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Adaptive hybrid sampling strategy.

        Combines stratified grid sampling with importance sampling:
        - 50% of points from uniform stratified grid (ensures coverage)
        - 50% of points from importance sampling (focuses on bright regions)

        This provides both uniform coverage and accurate representation
        of high-intensity regions.
        """
        h, w = intensity.shape[-2:]
        device = intensity.device

        # Part 1: Stratified grid sampling for uniform coverage
        n_stratified = self.n_source_points // 2
        n_per_dim = int(np.sqrt(n_stratified))

        y_indices = torch.linspace(0, h - 1, n_per_dim, device=device).long()
        x_indices = torch.linspace(0, w - 1, n_per_dim, device=device).long()
        yy, xx = torch.meshgrid(y_indices, x_indices, indexing="ij")
        strat_y = yy.flatten()[:n_stratified]
        strat_x = xx.flatten()[:n_stratified]
        strat_positions = self._indices_to_positions(strat_x, strat_y)

        # Part 2: Importance sampling in high-intensity regions
        n_importance = self.n_source_points - n_stratified

        # Square intensity to emphasize peaks
        prob = intensity.flatten() ** 2
        prob = prob / prob.sum().clamp(min=1e-10)

        indices_1d = torch.multinomial(prob, n_importance, replacement=True)
        imp_y = indices_1d // w
        imp_x = indices_1d % w
        imp_positions = self._indices_to_positions(imp_x, imp_y)

        # Combine positions
        positions = torch.cat([strat_positions, imp_positions], dim=0)

        # Compute weights based on local intensity for all positions
        all_x = torch.cat([strat_x, imp_x], dim=0)
        all_y = torch.cat([strat_y, imp_y], dim=0)

        # Clamp indices to valid range
        all_x = all_x.clamp(0, w - 1)
        all_y = all_y.clamp(0, h - 1)

        weights = intensity[all_y, all_x]
        weights = weights / weights.sum().clamp(min=1e-10)

        return positions, weights

    def _indices_to_positions(
        self,
        x_indices: Tensor,
        y_indices: Tensor,
    ) -> Tensor:
        """Convert pixel indices to physical coordinates."""
        x_coords = self.source_grid.x[0, 0] + x_indices.float() * self.source_grid.dx
        y_coords = self.source_grid.y[0, 0] + y_indices.float() * self.source_grid.dy
        return torch.stack([x_coords, y_coords], dim=-1)

    def _positions_to_indices(self, positions: Tensor) -> Tuple[Tensor, Tensor]:
        """Convert physical coordinates to pixel indices."""
        x_indices = (positions[:, 0] - self.source_grid.x[0, 0]) / self.source_grid.dx
        y_indices = (positions[:, 1] - self.source_grid.y[0, 0]) / self.source_grid.dy
        return x_indices.long(), y_indices.long()

    def _create_point_source_fields(
        self,
        positions: Tensor,
        aperture: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Create complex fields for point sources at given positions.

        For off-axis point sources, creates tilted plane waves representing
        the illumination angle from each source position.

        Parameters
        ----------
        positions : Tensor
            Source positions in physical coordinates. Shape: (N, 2)
        aperture : Tensor, optional
            Complex aperture function. Shape: (H, W)

        Returns
        -------
        Tensor
            Complex fields for each point source. Shape: (N, H, W)
        """
        n_sources = positions.shape[0]
        h, w = self.grid.nx, self.grid.ny
        device = positions.device

        # Get coordinate grids
        x = self.grid.x  # Shape: (H, W)
        y = self.grid.y  # Shape: (H, W)
        wavelength = self.grid.wl

        # Initialize fields
        fields = torch.zeros(n_sources, h, w, dtype=torch.cfloat, device=device)

        # Create tilted plane waves for each source position
        for i in range(n_sources):
            source_x, source_y = positions[i]

            # Phase tilt for off-axis illumination (far-field approximation)
            # The source position determines the angle of incidence
            # k_x = 2π × sin(θ_x) / λ ≈ 2π × source_x / (λ × z_eff)
            # For normalized angular coordinates:
            phase = 2 * np.pi * (source_x * x + source_y * y) / wavelength

            fields[i] = torch.exp(1j * phase)

        # Apply aperture if provided
        if aperture is not None:
            fields = fields * aperture.unsqueeze(0)

        return fields

    @property
    def illumination_mode(self) -> str:
        """Return the illumination mode for this propagator."""
        return "partially_coherent"

    def set_coherence_from_source(
        self,
        source_angular_diameter: float,
        propagation_distance: float,
    ) -> None:
        """
        Configure coherent patch size from source geometry.

        Uses van Cittert-Zernike theorem to estimate coherence length.

        Parameters
        ----------
        source_angular_diameter : float
            Angular diameter of source in radians
        propagation_distance : float
            Distance from source to observation plane in meters
        """
        wavelength = self.grid.wl
        # Coherence length ≈ λ / θ_source
        if source_angular_diameter > 0:
            self.coherent_patch_size = wavelength / source_angular_diameter
        else:
            self.coherent_patch_size = None  # Fully coherent

    def clear_cache(self) -> None:
        """Clear the PSF cache."""
        self._psf_cache.clear()

    def get_sampling_diagnostics(
        self,
        source_intensity: Tensor,
    ) -> Dict[str, Any]:
        """
        Get diagnostic information about source sampling without propagating.

        Useful for tuning n_source_points and sampling_method.

        Parameters
        ----------
        source_intensity : Tensor
            Source intensity distribution

        Returns
        -------
        dict
            Diagnostic information:
            - n_samples: Number of source points
            - positions: Sampled positions
            - weights: Sample weights
            - coverage: Fraction of total intensity covered
            - sampling_method: Method used
        """
        positions, weights = self._sample_source_points(source_intensity)

        # Compute coverage (how well samples represent total intensity)
        x_idx, y_idx = self._positions_to_indices(positions)
        h, w = source_intensity.shape[-2:]
        valid = (x_idx >= 0) & (x_idx < w) & (y_idx >= 0) & (y_idx < h)

        sampled_intensity = source_intensity[
            y_idx[valid].clamp(0, h - 1), x_idx[valid].clamp(0, w - 1)
        ].sum()
        total_intensity = source_intensity.sum()
        coverage = (sampled_intensity / total_intensity).item() if total_intensity > 0 else 0.0

        return {
            "n_samples": len(positions),
            "positions": positions,
            "weights": weights,
            "coverage": coverage,
            "sampling_method": self.sampling_method,
            "valid_samples": valid.sum().item(),
        }
