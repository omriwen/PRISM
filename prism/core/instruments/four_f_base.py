"""Abstract base class for 4f optical system instruments.

This module provides a unified implementation for all instruments based on the
four-focal-length (4f) optical system architecture. It consolidates common
functionality including forward modeling, k-space propagation, aperture mask
generation, and noise modeling.

The 4f system is the canonical configuration for Fourier optics, consisting of
two lenses separated by the sum of their focal lengths, with pupils placed at
the common back focal plane.

Classes
-------
FourFSystem
    Abstract base class providing unified 4f implementation for microscopes,
    telescopes, cameras, and SPIDS instruments.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional

import torch
from torch import Tensor

from ..optics.aperture_masks import ApertureMaskGenerator
from ..optics.detector_noise import DetectorNoiseModel
from ..optics.four_f_forward import FourFForwardModel
from ..propagators.base import CoherenceMode
from ..propagators.incoherent import OTFPropagator
from .base import Instrument, InstrumentConfig


if TYPE_CHECKING:
    pass


class FourFSystem(Instrument, ABC):
    """Abstract base class for 4f optical system instruments.

    This class provides unified implementation for all instruments based on the
    four-focal-length (4f) optical system. It consolidates common functionality
    that was previously duplicated across Microscope, Telescope, and SPIDS classes.

    Provided Functionality
    ----------------------
    - Forward model: Unified 4f propagation with padding to prevent aliasing
    - K-space propagation: FFT-based object ↔ pupil plane transformations
    - Aperture mask generation: Unified interface for all mask types
    - Input validation: Automatic detection of intensity/amplitude/complex inputs
    - Noise modeling: Optional detector noise (shot, read, dark current)

    Subclass Requirements
    ---------------------
    Subclasses must implement:
        - _create_pupils(): Return illumination and detection pupil functions
        - resolution_limit: Property returning theoretical resolution limit

    Subclasses may override:
        - compute_psf(): Custom PSF computation
        - get_info(): Additional instrument-specific info
        - _create_grid(): Custom grid configuration
        - _select_propagator(): Custom propagator selection

    Parameters
    ----------
    config : InstrumentConfig
        Instrument configuration (wavelength, n_pixels, pixel_size, etc.)
    padding_factor : float, default=2.0
        FFT padding factor to prevent wraparound artifacts. Must be >= 1.0.
        Default 2.0 provides good anti-aliasing. Set to 1.0 for no padding.
    aperture_cutoff_type : str, default='na'
        How to interpret aperture radius specifications:
        - 'na': Use numerical aperture (microscopes)
        - 'physical': Use physical radius in meters (telescopes)
        - 'pixels': Use radius in pixel units (generic)
    medium_index : float, default=1.0
        Refractive index of medium (1.0=air, 1.33=water, 1.515=oil)
    noise_model : DetectorNoiseModel, optional
        Optional detector noise model. If None, no noise is added.

    Attributes
    ----------
    padding_factor : float
        FFT padding factor used by forward model
    medium_index : float
        Refractive index of the medium
    _forward_model : FourFForwardModel
        Core 4f forward model (pad → FFT → pupils → IFFT → crop)
    _aperture_generator : ApertureMaskGenerator
        Unified aperture mask generator
    _noise_model : DetectorNoiseModel or None
        Optional detector noise model
    _default_aperture_radius : float
        Default sub-aperture radius for generate_aperture_mask

    Examples
    --------
    Subclassing FourFSystem (simplified Microscope):

    >>> class SimpleMicroscope(FourFSystem):
    ...     def __init__(self, config, na=1.4):
    ...         self.na = na
    ...         super().__init__(config, padding_factor=2.0,
    ...                         aperture_cutoff_type='na')
    ...
    ...     def _create_pupils(self, illumination_mode=None,
    ...                       illumination_params=None):
    ...         # Create NA-limited circular pupils
    ...         illum_pupil = self._aperture_generator.circular(na=self.na*0.8)
    ...         detect_pupil = self._aperture_generator.circular(na=self.na)
    ...         return illum_pupil, detect_pupil
    ...
    ...     @property
    ...     def resolution_limit(self):
    ...         return 0.61 * self.config.wavelength / self.na

    See Also
    --------
    Instrument : Base class for all optical instruments
    FourFForwardModel : Core 4f forward model implementation
    ApertureMaskGenerator : Unified aperture mask generation
    DetectorNoiseModel : Realistic detector noise simulation
    """

    def __init__(
        self,
        config: InstrumentConfig,
        padding_factor: float = 2.0,
        aperture_cutoff_type: str = "na",
        medium_index: float = 1.0,
        noise_model: Optional[DetectorNoiseModel] = None,
    ) -> None:
        """Initialize FourFSystem with configuration and components.

        Parameters
        ----------
        config : InstrumentConfig
            Instrument configuration parameters
        padding_factor : float, default=2.0
            FFT padding factor (>= 1.0)
        aperture_cutoff_type : str, default='na'
            Aperture specification type ('na', 'physical', 'pixels')
        medium_index : float, default=1.0
            Refractive index of medium
        noise_model : DetectorNoiseModel, optional
            Optional detector noise model

        Raises
        ------
        ValueError
            If padding_factor < 1.0
        """
        # Initialize base instrument
        super().__init__(config)

        # Store configuration
        self.padding_factor = padding_factor
        self.medium_index = medium_index
        self._default_aperture_radius: float = 0.0  # Set by subclass

        # Coherence mode settings
        self._coherence_warnings_enabled: bool = True

        # Create core components (lazy initialization via properties)
        self._forward_model_instance: Optional[FourFForwardModel] = None
        self._aperture_generator_instance: Optional[ApertureMaskGenerator] = None
        self._noise_model = noise_model

        # Validate padding factor
        if padding_factor < 1.0:
            raise ValueError(f"padding_factor must be >= 1.0, got {padding_factor}")

        # Store aperture cutoff type for generator
        self._aperture_cutoff_type = aperture_cutoff_type

    @property
    def _forward_model_lazy(self) -> FourFForwardModel:
        """Get forward model (lazy initialization).

        Returns
        -------
        FourFForwardModel
            The 4f forward model instance
        """
        if self._forward_model_instance is None:
            self._forward_model_instance = FourFForwardModel(
                grid=self.grid,
                padding_factor=self.padding_factor,
                normalize_output=True,
            )
        return self._forward_model_instance

    @property
    def _aperture_generator_lazy(self) -> ApertureMaskGenerator:
        """Get aperture mask generator (lazy initialization).

        Returns
        -------
        ApertureMaskGenerator
            The aperture mask generator instance
        """
        if self._aperture_generator_instance is None:
            self._aperture_generator_instance = ApertureMaskGenerator(
                grid=self.grid,
                cutoff_type=self._aperture_cutoff_type,
                wavelength=self.config.wavelength,
                medium_index=self.medium_index,
            )
        return self._aperture_generator_instance

    @abstractmethod
    def _create_pupils(
        self,
        illumination_mode: Optional[str] = None,
        illumination_params: Optional[dict] = None,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Create illumination and detection pupil functions.

        This abstract method must be implemented by subclasses to define
        the instrument-specific pupil functions. Pupils can vary based on
        illumination mode (brightfield, darkfield, phase, DIC, etc.).

        Parameters
        ----------
        illumination_mode : str, optional
            Type of illumination ('brightfield', 'darkfield', 'phase', 'dic',
            'custom', etc.). If None, use instrument default.
        illumination_params : dict, optional
            Additional parameters for illumination mode (e.g., NA values,
            annular ratios, phase shifts, custom pupils).

        Returns
        -------
        tuple[Tensor or None, Tensor or None]
            (illumination_pupil, detection_pupil) in k-space.
            Either can be None for identity (all-pass, no filtering).
            Pupils should be complex-valued tensors of shape (n_pixels, n_pixels).

        Notes
        -----
        Pupils are applied in the Fourier plane (back focal plane of first lens).
        They should be provided as complex-valued tensors that include both
        amplitude and phase modulation.

        Examples
        --------
        Microscope implementation:

        >>> def _create_pupils(self, illumination_mode=None,
        ...                   illumination_params=None):
        ...     params = illumination_params or {}
        ...     if illumination_mode == 'brightfield':
        ...         illum_na = params.get('illumination_na', 0.8 * self.na)
        ...         illum = self._aperture_generator_lazy.circular(na=illum_na)
        ...         detect = self._aperture_generator_lazy.circular(na=self.na)
        ...     elif illumination_mode == 'darkfield':
        ...         annular_ratio = params.get('annular_ratio', 0.8)
        ...         illum = self._aperture_generator_lazy.annular(
        ...             inner_na=annular_ratio*self.na, outer_na=self.na)
        ...         detect = self._aperture_generator_lazy.circular(na=self.na)
        ...     else:
        ...         illum = self._aperture_generator_lazy.circular(na=0.8*self.na)
        ...         detect = self._aperture_generator_lazy.circular(na=self.na)
        ...     return illum, detect
        """
        ...

    @property
    @abstractmethod
    def resolution_limit(self) -> float:
        """Theoretical resolution limit.

        Returns
        -------
        float
            Resolution limit in meters (for microscopes/cameras) or radians
            (for telescopes). The units depend on the instrument type.

        Notes
        -----
        For microscopes: Abbe limit = 0.61 * λ / NA
        For telescopes: Rayleigh criterion = 1.22 * λ / D
        """
        ...

    def forward(
        self,
        field: Tensor,
        illumination_mode: Optional[str] = None,
        illumination_params: Optional[dict] = None,
        add_noise: bool = False,
        input_mode: str = "auto",
        input_pixel_size: Optional[float] = None,
        coherence_mode: CoherenceMode = CoherenceMode.COHERENT,
        source_intensity: Optional[Tensor] = None,
        n_source_points: int = 100,
        **kwargs: Any,
    ) -> Tensor:
        """Forward propagation through 4f system.

        This is the unified forward method for all 4f-based instruments.
        Instrument-specific behavior (pupil shapes, illumination modes) is
        controlled by the _create_pupils() abstract method.

        Parameters
        ----------
        field : Tensor
            Input field at object plane. Shape: (H, W), (C, H, W), or (B, C, H, W)
        illumination_mode : str, optional
            Type of illumination. Interpretation is instrument-specific.
            Common modes: 'brightfield', 'darkfield', 'phase', 'dic', 'custom'
        illumination_params : dict, optional
            Parameters for illumination mode (NA values, phase shifts, etc.)
        add_noise : bool, default=False
            Whether to add detector noise (requires noise_model to be set)
        input_mode : str, default='auto'
            How to interpret input values:
            - 'intensity': Field is I = |E|², converted via sqrt(I)
            - 'amplitude': Field is |E|, values >= 0
            - 'complex': Field is already complex E = |E|*exp(i*phi)
            - 'auto': Auto-detect from dtype and values
        input_pixel_size : float, optional
            Physical size of input pixels (meters) for FOV validation
        coherence_mode : CoherenceMode, default=CoherenceMode.COHERENT
            Illumination coherence mode:
            - COHERENT: Standard coherent amplitude transfer (laser illumination)
            - INCOHERENT: OTF-based propagation for self-luminous objects
              (fluorescence). Uses only detection pupil; illumination pupil ignored.
            - PARTIALLY_COHERENT: Extended source integration for LED/extended
              illumination. Requires source_intensity parameter.
        source_intensity : Tensor, optional
            Source intensity distribution for PARTIALLY_COHERENT mode.
            Required when coherence_mode=PARTIALLY_COHERENT.
            Shape should match field spatial dimensions.
        n_source_points : int, default=100
            Number of source points for PARTIALLY_COHERENT sampling.
            Higher values give more accurate results but slower computation.
        **kwargs : Any
            Additional instrument-specific parameters (passed to _create_pupils)

        Returns
        -------
        Tensor
            Output intensity at detector. Shape matches input.

        Notes
        -----
        The forward model depends on coherence_mode:

        **COHERENT** (default):
            I(x,y) = |IFFT{ P_det · P_illum · FFT{E_object} }|²

        **INCOHERENT** (fluorescence):
            I_out = IFFT{ OTF · FFT{I_in} }
            where OTF = Autocorr(P_det) and I_in = |field|²
            Note: P_illum is IGNORED (self-luminous emission model)

        **PARTIALLY_COHERENT** (extended source):
            I = Σ_i w_i · |coherent_propagate(E, phase_i)|²
            Integrates over source intensity distribution

        Examples
        --------
        Basic forward pass (coherent, default):

        >>> field = torch.randn(512, 512)
        >>> intensity = instrument.forward(field)

        With illumination mode:

        >>> intensity = instrument.forward(
        ...     field,
        ...     illumination_mode='darkfield',
        ...     illumination_params={'annular_ratio': 0.8}
        ... )

        Fluorescence imaging (incoherent):

        >>> intensity = instrument.forward(
        ...     field,
        ...     coherence_mode=CoherenceMode.INCOHERENT,
        ... )

        LED brightfield (partially coherent):

        >>> source = gaussian_source(512, sigma=10)  # Extended source
        >>> intensity = instrument.forward(
        ...     field,
        ...     coherence_mode=CoherenceMode.PARTIALLY_COHERENT,
        ...     source_intensity=source,
        ...     n_source_points=200,
        ... )

        With noise:

        >>> instrument._noise_model = DetectorNoiseModel(snr_db=40.0)
        >>> noisy_intensity = instrument.forward(field, add_noise=True)
        """
        # Validate input (handles intensity/amplitude/complex conversion)
        field = self.validate_field(
            field,
            input_mode=input_mode,
            input_pixel_size=input_pixel_size,
        )

        # Get pupils for this illumination mode (instrument-specific)
        illum_pupil, detect_pupil = self._create_pupils(illumination_mode, illumination_params)

        # Ensure pupils are on same device as field
        if illum_pupil is not None:
            illum_pupil = illum_pupil.to(field.device)
        if detect_pupil is not None:
            detect_pupil = detect_pupil.to(field.device)

        # Route to appropriate coherence mode handler
        if coherence_mode == CoherenceMode.COHERENT:
            output = self._forward_coherent(field, illum_pupil, detect_pupil)
        elif coherence_mode == CoherenceMode.INCOHERENT:
            output = self._forward_incoherent(field, detect_pupil, illumination_mode)
        elif coherence_mode == CoherenceMode.PARTIALLY_COHERENT:
            output = self._forward_partially_coherent(
                field,
                illum_pupil,
                detect_pupil,
                source_intensity,
                n_source_points,
            )
        else:
            raise ValueError(
                f"Invalid coherence_mode: {coherence_mode}. Must be one of {list(CoherenceMode)}"
            )

        # Add noise if requested and noise model is available
        if add_noise and self._noise_model is not None:
            output = self._noise_model(output, add_noise=True)

        return output

    def _forward_coherent(
        self,
        field: Tensor,
        illum_pupil: Optional[Tensor],
        detect_pupil: Optional[Tensor],
    ) -> Tensor:
        """Forward propagation for coherent illumination.

        Standard amplitude transfer through 4f system.

        Parameters
        ----------
        field : Tensor
            Validated complex input field
        illum_pupil : Tensor or None
            Illumination pupil (k-space filter)
        detect_pupil : Tensor or None
            Detection pupil (k-space filter)

        Returns
        -------
        Tensor
            Output intensity |E|²
        """
        # Forward through 4f model (pad → FFT → pupils → IFFT → crop)
        output = self._forward_model_lazy(
            field,
            illumination_pupil=illum_pupil,
            detection_pupil=detect_pupil,
            return_complex=False,  # Return intensity
        )
        return output

    def _forward_incoherent(
        self,
        field: Tensor,
        detect_pupil: Optional[Tensor],
        illumination_mode: Optional[str],
    ) -> Tensor:
        """Forward propagation for incoherent (self-luminous) objects.

        Uses OTF-based propagation. The illumination pupil is IGNORED
        because incoherent imaging models self-luminous objects where
        only the detection optics matter.

        Parameters
        ----------
        field : Tensor
            Input field (will be converted to intensity)
        detect_pupil : Tensor or None
            Detection pupil used to compute OTF
        illumination_mode : str or None
            Illumination mode (for warning validation)

        Returns
        -------
        Tensor
            Output intensity after OTF convolution
        """
        # Warn for illumination modes that don't make physical sense with incoherent
        # (e.g., darkfield, phase contrast, DIC require coherent interference)
        if self._coherence_warnings_enabled and illumination_mode is not None:
            if illumination_mode.lower() not in ("brightfield",):
                warnings.warn(
                    f"Illumination mode '{illumination_mode}' with INCOHERENT coherence "
                    f"may not produce physically meaningful results. "
                    f"Incoherent imaging (e.g., fluorescence) models self-luminous objects "
                    f"where phase information is lost. Consider using COHERENT mode for "
                    f"'{illumination_mode}' or 'brightfield' for incoherent imaging.",
                    UserWarning,
                    stacklevel=4,
                )

        # Convert complex field to intensity (self-luminous emission model)
        # For fluorescence: input represents emission intensity pattern
        if field.is_complex():
            intensity = torch.abs(field) ** 2
        else:
            # Already intensity (real-valued)
            intensity = field

        # Create default pupil if none provided (all-pass)
        if detect_pupil is None:
            detect_pupil = torch.ones(
                self.config.n_pixels,
                self.config.n_pixels,
                dtype=torch.complex64,
                device=field.device,
            )

        # Create OTF propagator from detection pupil only
        # (illumination pupil is ignored for self-luminous objects)
        otf_propagator = OTFPropagator(
            aperture=detect_pupil,
            grid=self.grid,
            normalize=True,
        )

        # Propagate intensity through OTF
        output = otf_propagator(intensity)

        return output

    def _forward_partially_coherent(
        self,
        field: Tensor,
        illum_pupil: Optional[Tensor],
        detect_pupil: Optional[Tensor],
        source_intensity: Optional[Tensor],
        n_source_points: int,
    ) -> Tensor:
        """Forward propagation for partially coherent (extended source) illumination.

        Integrates over source points to model extended source illumination.
        Each source point contributes a coherent image with a different
        illumination tilt; the total is their incoherent (intensity) sum.

        Parameters
        ----------
        field : Tensor
            Complex input field
        illum_pupil : Tensor or None
            Illumination pupil
        detect_pupil : Tensor or None
            Detection pupil
        source_intensity : Tensor or None
            Source intensity distribution (required)
        n_source_points : int
            Number of source points for integration

        Returns
        -------
        Tensor
            Output intensity integrated over source

        Raises
        ------
        ValueError
            If source_intensity is not provided
        """
        # Validate source_intensity is provided
        if source_intensity is None:
            raise ValueError(
                "source_intensity is required for PARTIALLY_COHERENT mode. "
                "Provide a 2D tensor representing the extended source distribution. "
                "For LED brightfield, use a Gaussian centered at the optical axis."
            )

        device = field.device

        # Ensure source_intensity is on same device
        source_intensity = source_intensity.to(device)

        # Sample source points using importance sampling
        # (more samples where source is brighter)
        positions, weights = self._sample_source_points(source_intensity, n_source_points)

        # Initialize output accumulator
        output = torch.zeros_like(field.real if field.is_complex() else field)

        # Get coordinate grid for phase tilt computation
        x = self.grid.x.to(device)  # Shape: (H, W)
        y = self.grid.y.to(device)  # Shape: (H, W)
        wavelength = self.config.wavelength

        # Integrate over source points
        for i in range(len(positions)):
            source_x, source_y = positions[i]
            weight = weights[i]

            # Skip negligible weights
            if weight < 1e-10:
                continue

            # Create phase tilt for this source position
            # Each source point creates an illumination angle, which appears
            # as a phase tilt in the pupil plane: exp(i * 2π/λ * (θ_x * x + θ_y * y))
            # The source position (in angular coordinates) maps to the tilt angle
            phase_tilt = torch.exp(1j * 2 * torch.pi * (source_x * x + source_y * y) / wavelength)

            # Apply phase tilt to illumination pupil
            if illum_pupil is not None:
                tilted_illum = illum_pupil * phase_tilt
            else:
                tilted_illum = phase_tilt.to(torch.complex64)

            # Run coherent forward pass with tilted illumination
            coherent_output = self._forward_coherent(field, tilted_illum, detect_pupil)

            # Accumulate weighted intensity
            output = output + weight * coherent_output

        return output

    def _sample_source_points(
        self,
        source_intensity: Tensor,
        n_points: int,
    ) -> tuple[Tensor, Tensor]:
        """Sample source points from intensity distribution using importance sampling.

        Parameters
        ----------
        source_intensity : Tensor
            Source intensity distribution, shape (H, W)
        n_points : int
            Number of points to sample

        Returns
        -------
        positions : Tensor
            Source point positions in physical coordinates, shape (N, 2)
        weights : Tensor
            Normalized weights for each point, shape (N,)
        """
        device = source_intensity.device
        h, w = source_intensity.shape[-2:]

        # Normalize to probability distribution
        prob = source_intensity.flatten()
        prob = prob / prob.sum().clamp(min=1e-10)

        # Sample indices using importance sampling
        indices_1d = torch.multinomial(prob, n_points, replacement=True)

        # Convert to 2D indices
        y_indices = indices_1d // w
        x_indices = indices_1d % w

        # Convert to physical coordinates using grid
        # Note: grid.x and grid.y give coordinates, we need to map pixel indices
        dx = self.grid.dx
        dy = self.grid.dy
        # Grid coordinates are centered, so we compute offset from center
        x_center = w // 2
        y_center = h // 2

        x_coords = (x_indices.float() - x_center) * dx
        y_coords = (y_indices.float() - y_center) * dy

        positions = torch.stack([x_coords, y_coords], dim=-1)

        # For importance sampling, weights are uniform
        # (the sampling already accounts for intensity distribution)
        weights = torch.ones(n_points, device=device) / n_points

        return positions, weights

    def propagate_to_kspace(self, field: Tensor) -> Tensor:
        """Propagate field to k-space (Fourier domain).

        Performs FFT from object plane to pupil plane (back focal plane of
        first lens). The output is fftshift'd so DC is at the center.

        Parameters
        ----------
        field : Tensor
            Complex field at object plane. Shape: (H, W) or (B, C, H, W)

        Returns
        -------
        Tensor
            Complex k-space field (centered, DC at N//2)

        Notes
        -----
        This is equivalent to the first two steps of the 4f model:
            1. FFT to Fourier plane
            2. fftshift to center DC

        The coordinate system has DC at [N//2, N//2], matching the
        convention used by generate_aperture_mask().

        Examples
        --------
        >>> field_obj = torch.randn(512, 512, dtype=torch.complex64)
        >>> field_kspace = instrument.propagate_to_kspace(field_obj)
        >>> print(field_kspace.shape)  # (512, 512)
        """
        # Handle different input dimensions
        if field.ndim == 2:
            # Single field: (H, W)
            field_2d = field
        elif field.ndim == 3:
            # Multi-channel: (C, H, W) - process first channel
            field_2d = field[0]
        elif field.ndim == 4:
            # Batched: (B, C, H, W) - process first batch/channel
            field_2d = field[0, 0]
        else:
            field_2d = field

        # FFT to k-space with centering (DC at center)
        field_kspace = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.fftshift(field_2d, dim=(-2, -1))),
            dim=(-2, -1),
        )

        return field_kspace

    def propagate_to_spatial(self, field_kspace: Tensor) -> Tensor:
        """Propagate k-space field back to spatial domain and compute intensity.

        Performs inverse FFT from pupil plane back to image plane, then
        computes intensity. This is the final step of the 4f model.

        Parameters
        ----------
        field_kspace : Tensor
            Complex k-space field (centered, DC at N//2)

        Returns
        -------
        Tensor
            Real-valued intensity at detector plane

        Notes
        -----
        This implements the final steps of the 4f model:
            1. ifftshift to prepare for IFFT
            2. IFFT to image plane
            3. Compute intensity |E|²

        The input is expected to be fftshift'd (DC at center), matching
        the output of propagate_to_kspace().

        Examples
        --------
        >>> field_kspace = instrument.propagate_to_kspace(field_obj)
        >>> # Apply pupil mask
        >>> field_kspace_masked = field_kspace * pupil_mask
        >>> intensity = instrument.propagate_to_spatial(field_kspace_masked)
        """
        # Inverse FFT back to spatial domain
        field_spatial = torch.fft.ifftshift(
            torch.fft.ifft2(torch.fft.ifftshift(field_kspace, dim=(-2, -1))),
            dim=(-2, -1),
        )

        # Compute intensity
        intensity = torch.abs(field_spatial) ** 2

        return intensity

    def generate_aperture_mask(
        self,
        center: Optional[List[float]] = None,
        radius: Optional[float] = None,
    ) -> Tensor:
        """Generate circular aperture mask at specified position.

        Creates a circular aperture in the pupil plane (k-space) at the
        specified center position. This enables SPIDS synthetic aperture
        construction by sampling different regions of k-space.

        Parameters
        ----------
        center : List[float], optional
            Center position [y, x] in pixels from DC (0,0).
            Defaults to [0.0, 0.0] (centered on DC).
        radius : float, optional
            Aperture radius in units specified by aperture_cutoff_type.
            Defaults to _default_aperture_radius if not specified.

        Returns
        -------
        Tensor
            Binary mask of shape (n_pixels, n_pixels), dtype float32

        Notes
        -----
        The aperture specification depends on aperture_cutoff_type:
        - 'na': radius interpreted as numerical aperture
        - 'physical': radius in meters
        - 'pixels': radius in pixel units

        The coordinate system matches propagate_to_kspace() output
        (DC at center).

        Examples
        --------
        >>> # Centered aperture (conventional imaging)
        >>> mask = instrument.generate_aperture_mask([0, 0])
        >>>
        >>> # Off-center sub-aperture for SPIDS
        >>> mask = instrument.generate_aperture_mask([10, 5], radius=15)
        >>>
        >>> # Use with k-space propagation
        >>> field_kspace = instrument.propagate_to_kspace(field)
        >>> masked_kspace = field_kspace * mask
        >>> intensity = instrument.propagate_to_spatial(masked_kspace)
        """
        if center is None:
            center = [0.0, 0.0]

        if radius is None:
            radius = self._default_aperture_radius

        # Use sub_aperture method of aperture generator
        return self._aperture_generator_lazy.sub_aperture(
            center=center,
            radius=radius,
        )

    def compute_psf(self, **kwargs: Any) -> Tensor:
        """Compute point spread function.

        Default implementation creates a delta function input and propagates
        through the system. Subclasses may override for specialized PSF
        computation (e.g., 3D PSF stacks for microscopes).

        Parameters
        ----------
        **kwargs : Any
            Additional parameters passed to forward() (e.g., illumination_mode)

        Returns
        -------
        Tensor
            PSF tensor (2D or 3D depending on instrument), normalized to max=1

        Examples
        --------
        >>> psf = instrument.compute_psf()
        >>> print(psf.shape)  # (512, 512)
        >>>
        >>> # With illumination mode
        >>> psf_df = instrument.compute_psf(illumination_mode='darkfield')
        """
        # Create point source at center
        n = self.config.n_pixels
        delta = torch.zeros(n, n, dtype=torch.complex64, device=self.grid.x.device)
        delta[n // 2, n // 2] = 1.0

        # Forward through system
        psf = self.forward(delta, **kwargs)

        # Normalize
        max_val = psf.max()
        if max_val > 0:
            psf = psf / max_val

        return psf

    def get_info(self) -> dict:
        """Get instrument information summary.

        Returns
        -------
        dict
            Dictionary with instrument parameters and characteristics

        Notes
        -----
        Subclasses should override to add instrument-specific information,
        calling super().get_info() first to get base information.

        Examples
        --------
        >>> info = instrument.get_info()
        >>> print(info['resolution_limit'])
        >>> print(info['wavelength'])
        """
        # Get base info from Instrument
        info = super().get_info()

        # Add 4f-specific info
        info.update(
            {
                "padding_factor": self.padding_factor,
                "medium_index": self.medium_index,
                "aperture_cutoff_type": self._aperture_cutoff_type,
            }
        )

        return info

    def __repr__(self) -> str:
        """String representation of instrument."""
        return (
            f"{self.__class__.__name__}("
            f"wavelength={self.config.wavelength:.2e}, "
            f"n_pixels={self.config.n_pixels}, "
            f"resolution={self.resolution_limit:.2e})"
        )


__all__ = ["FourFSystem"]
