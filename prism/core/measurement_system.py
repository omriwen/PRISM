"""Measurement system for progressive imaging.

This module implements MeasurementSystem, which handles SPIDS-specific progressive
measurement logic that works with ANY instrument (Telescope, Microscope, Camera).

The MeasurementSystem wraps an Instrument instance and adds:
- Cumulative mask tracking for progressive measurements
- Measurement accumulation and caching
- SPIDS training support (dual measurements: old mask + new aperture)
- Support for both scanning aperture and scanning illumination modes

This separation allows:
- Pure optical physics in Instrument classes
- SPIDS-specific measurement logic in MeasurementSystem
- Any instrument to be used with SPIDS algorithm
- Flexible switching between scanning aperture and scanning illumination

Scanning Modes
--------------
APERTURE (default):
    Traditional synthetic aperture mode. A sub-aperture scans different
    k-space positions while illumination remains uniform. The `centers`
    parameter specifies aperture positions in pixel coordinates.

ILLUMINATION:
    FPM-style scanning illumination mode. Tilted illumination shifts the
    object spectrum, with detection at DC. The `centers` parameter specifies
    illumination k-space positions. For Microscope instruments, centers
    are converted from pixel coordinates to k-space coordinates automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from ..utils.measurement_cache import MeasurementCache
from .instruments.base import Instrument


if TYPE_CHECKING:
    from .line_acquisition import IncoherentLineAcquisition
    from .optics.illumination import IlluminationSourceType


class ScanningMode(Enum):
    """Scanning mode for MeasurementSystem.

    Defines how the measurement system samples k-space. Both modes are
    mathematically equivalent for point-like sources by the Fourier shift
    theorem, but they have different physical interpretations and support
    different source geometries.

    Attributes
    ----------
    APERTURE : auto
        Scanning aperture mode (default). Sub-aperture scans k-space positions
        while illumination is uniform. Centers are specified in pixel coordinates.
        This is the traditional synthetic aperture approach.

    ILLUMINATION : auto
        Scanning illumination mode. Tilted illumination shifts object spectrum,
        with detection aperture fixed at DC. Centers are specified in pixel
        coordinates and converted to k-space automatically. This mode supports
        finite-size sources (GAUSSIAN, CIRCULAR) for partial coherence modeling.

    See Also
    --------
    spids.core.instruments.microscope.Microscope : Instrument supporting both modes
    spids.core.optics.illumination : Illumination source models
    """

    APERTURE = auto()
    ILLUMINATION = auto()


class IlluminationScanMethod(Enum):
    """Method for illumination scanning.

    Defines how illumination scanning is performed when using
    ScanningMode.ILLUMINATION.

    Attributes
    ----------
    ANGULAR : auto
        Tilted plane wave illumination (source at infinity).
        Phase tilt exp(i·2π·(kx·x + ky·y)) applied uniformly.
        This is the traditional FPM/LED array approach.

    SPATIAL : auto
        Spatially-shifted source (finite distance).
        Source at position (x₀, y₀) creates position-dependent
        illumination angles. Phase varies spatially based on
        path length from source to each object point.
    """

    ANGULAR = auto()
    SPATIAL = auto()


@dataclass
class MeasurementSystemConfig:
    """Configuration for MeasurementSystem.

    Attributes
    ----------
    enable_caching : bool
        Whether to enable measurement caching (15-25% speedup). Default True.
    cache_max_size : int
        Maximum number of cached measurements. Default 1000.
    add_noise_by_default : bool
        Whether to add noise by default in measurements. Default True.
    scanning_mode : ScanningMode
        Scanning mode for k-space sampling. Default APERTURE.
        - APERTURE: Traditional scanning aperture (sub-aperture moves in k-space)
        - ILLUMINATION: Scanning illumination (tilted illumination, detect at DC)
    illumination_source_type : str
        Type of illumination source for ILLUMINATION mode. Default "POINT".
        - "POINT": Tilted plane wave (equivalent to scanning aperture)
        - "GAUSSIAN": Gaussian intensity profile (partial coherence)
        - "CIRCULAR": Circular top-hat profile
        Only used when scanning_mode is ILLUMINATION.
    illumination_radius : float, optional
        k-space width parameter for finite illumination sources (1/m).
        Required for GAUSSIAN (sigma) and CIRCULAR (radius) sources.
        Only used when scanning_mode is ILLUMINATION.
    illumination_scan_method : IlluminationScanMethod
        Method for illumination scanning. Default ANGULAR.
        - ANGULAR: Tilted plane wave illumination (source at infinity)
        - SPATIAL: Spatially-shifted source (finite distance)
        Only used when scanning_mode is ILLUMINATION.
    illumination_source_distance : float, optional
        Source-to-object distance for SPATIAL illumination mode (meters).
        Required when illumination_scan_method is SPATIAL.
        Must be positive. Determines phase curvature from finite-distance source.

    Notes
    -----
    When using ILLUMINATION mode:
    - Centers are interpreted as pixel coordinates and converted to k-space
    - The conversion uses the instrument's grid for proper scaling
    - For POINT sources, results are equivalent to APERTURE mode
    - For finite sources, partial coherence effects are modeled
    - ANGULAR method uses tilted plane waves (FPM/LED array style)
    - SPATIAL method models finite source distance with curved wavefronts
    """

    enable_caching: bool = True
    cache_max_size: int = 1000
    add_noise_by_default: bool = True
    scanning_mode: ScanningMode = field(default=ScanningMode.APERTURE)
    illumination_source_type: str = "POINT"
    illumination_radius: Optional[float] = None
    illumination_scan_method: IlluminationScanMethod = field(default=IlluminationScanMethod.ANGULAR)
    illumination_source_distance: Optional[float] = None

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.cache_max_size <= 0:
            raise ValueError(f"Cache max size must be positive, got {self.cache_max_size}")

        # Validate illumination source type
        valid_source_types = {"POINT", "GAUSSIAN", "CIRCULAR"}
        if self.illumination_source_type not in valid_source_types:
            raise ValueError(
                f"illumination_source_type must be one of {valid_source_types}, "
                f"got '{self.illumination_source_type}'"
            )

        # Validate illumination_radius for finite sources
        if self.scanning_mode == ScanningMode.ILLUMINATION:
            if self.illumination_source_type in {"GAUSSIAN", "CIRCULAR"}:
                if self.illumination_radius is None:
                    raise ValueError(
                        f"illumination_radius is required for {self.illumination_source_type} "
                        f"source type in ILLUMINATION mode"
                    )

        # Validate spatial illumination requirements
        if (
            self.scanning_mode == ScanningMode.ILLUMINATION
            and self.illumination_scan_method == IlluminationScanMethod.SPATIAL
        ):
            if self.illumination_source_distance is None:
                raise ValueError(
                    "illumination_source_distance is required for SPATIAL "
                    "illumination mode. Provide source-to-object distance in meters."
                )
            if self.illumination_source_distance <= 0:
                raise ValueError(
                    f"illumination_source_distance must be positive, "
                    f"got {self.illumination_source_distance}"
                )


class MeasurementSystem:
    """Progressive measurement system for SPIDS algorithm.

    **CRITICAL CLASS FOR SPIDS ALGORITHM**

    MeasurementSystem wraps any Instrument instance and maintains cumulative masks
    for progressive imaging. It enables SPIDS training where the neural network
    learns to reconstruct objects by matching both:
    1. Previous measurements through accumulated mask
    2. New measurements through current aperture/illumination

    The key innovation is that each training step uses:
    - Old measurement: Current reconstruction → accumulated mask → measurement
    - New measurement: Ground truth → new aperture/illumination → measurement

    This allows the loss function to ensure consistency with all previous
    measurements while incorporating new information.

    Scanning Modes
    --------------
    The MeasurementSystem supports two scanning modes:

    **APERTURE** (default):
        Traditional synthetic aperture. Sub-aperture scans k-space while
        illumination is uniform. Centers specify pixel positions in k-space.

    **ILLUMINATION**:
        FPM-style scanning illumination. Tilted illumination shifts object
        spectrum, with detection at DC. Centers are automatically converted
        from pixel to k-space coordinates. Supports finite-size sources
        (GAUSSIAN, CIRCULAR) for partial coherence modeling.

    Attributes
    ----------
    instrument : Instrument
        Any Instrument instance (Telescope, Microscope, Camera).
    config : MeasurementSystemConfig
        Configuration including scanning_mode and illumination parameters.
    cum_mask : Tensor
        Cumulative mask of all measured regions in k-space.
    sample_count : int
        Number of accumulated measurements.
    measurement_cache : Optional[MeasurementCache]
        Cache for measurement performance optimization.

    Methods
    -------
    add_mask(centers)
        Add new aperture/illumination positions to cumulative mask.
    measure_through_accumulated_mask(field)
        Measure through all previous apertures.
    measure(ground_truth, reconstruction, centers)
        Generate dual measurements for training.
    get_measurements(field, centers)
        Generate instrument measurements at specified positions.
    get_cache_stats()
        Get cache performance statistics.
    clear_cache()
        Clear measurement cache.

    Examples
    --------
    Standard aperture mode (default):

    >>> from prism.core.instruments import Telescope, TelescopeConfig
    >>> config = TelescopeConfig(n_pixels=512, aperture_radius_pixels=25)
    >>> telescope = Telescope(config)
    >>> measurement_system = MeasurementSystem(telescope)
    >>>
    >>> # First measurement
    >>> meas = measurement_system.measure(object, None, [[0, 0]])
    >>> measurement_system.add_mask([[0, 0]])
    >>>
    >>> # Second measurement - returns [old_mask_meas, new_meas]
    >>> meas = measurement_system.measure(object, reconstruction, [[10, 10]])
    >>> measurement_system.add_mask([[10, 10]])

    Scanning illumination mode with Microscope:

    >>> from prism.core.instruments import Microscope, MicroscopeConfig
    >>> from prism.core.measurement_system import MeasurementSystemConfig, ScanningMode
    >>> micro_config = MicroscopeConfig(n_pixels=256, numerical_aperture=0.9)
    >>> microscope = Microscope(micro_config)
    >>> ms_config = MeasurementSystemConfig(
    ...     scanning_mode=ScanningMode.ILLUMINATION,
    ...     illumination_source_type="POINT",  # Equivalent to aperture mode
    ... )
    >>> measurement_system = MeasurementSystem(microscope, config=ms_config)
    >>> # Now centers are converted to k-space illumination positions
    >>> meas = measurement_system.measure(object, None, [[10, 5]])

    Progressive Training Pattern:
        for center in sample_centers:
            # Get measurements: [old mask on reconstruction, new sample on ground truth]
            measurement = measurement_system.measure(ground_truth, reconstruction, [center])

            # Train model to match both measurements
            reconstruction = model()
            loss = loss_fn(reconstruction, measurement)
            loss.backward()
            optimizer.step()

            # Add this position to accumulated mask
            measurement_system.add_mask([center])

    See Also
    --------
    ScanningMode : Enum defining available scanning modes
    MeasurementSystemConfig : Configuration dataclass
    spids.core.instruments.microscope.Microscope : Instrument with illumination support
    """

    def __init__(
        self,
        instrument: Instrument,
        config: Optional[MeasurementSystemConfig] = None,
        measurement_cache: Optional[MeasurementCache] = None,
        line_acquisition: Optional["IncoherentLineAcquisition"] = None,
    ) -> None:
        """Initialize MeasurementSystem with an instrument.

        Parameters
        ----------
        instrument : Instrument
            Any Instrument instance (Telescope, Microscope, Camera).
        config : MeasurementSystemConfig, optional
            Configuration including scanning mode. If None, uses defaults
            (APERTURE mode with caching enabled).
        measurement_cache : MeasurementCache, optional
            Shared cache instance. If None, creates new cache.
        line_acquisition : IncoherentLineAcquisition, optional
            Line acquisition module for motion blur simulation.

        Raises
        ------
        ValueError
            If ILLUMINATION mode is requested but instrument doesn't support it.
        """
        self.instrument = instrument
        self.config = config if config is not None else MeasurementSystemConfig()
        self.config.validate()

        # Validate instrument supports illumination mode if requested
        if self.config.scanning_mode == ScanningMode.ILLUMINATION:
            if not self._instrument_supports_illumination():
                raise ValueError(
                    f"ILLUMINATION mode requires an instrument that supports scanning "
                    f"illumination (e.g., Microscope). Got {type(instrument).__name__}. "
                    f"Use APERTURE mode or switch to a Microscope instrument."
                )

        # Initialize cumulative mask and sample count
        n_pixels = instrument.config.n_pixels
        self.cum_mask = torch.zeros((n_pixels, n_pixels), dtype=torch.bool)
        self.sample_count = 0

        # Line acquisition for motion blur (optional)
        self.line_acquisition = line_acquisition

        # Measurement cache for performance
        self.measurement_cache: Optional[MeasurementCache]
        if self.config.enable_caching:
            self.measurement_cache = (
                measurement_cache if measurement_cache is not None else MeasurementCache()
            )
        else:
            self.measurement_cache = None

    def _instrument_supports_illumination(self) -> bool:
        """Check if the instrument supports scanning illumination mode.

        Returns
        -------
        bool
            True if instrument has illumination_center parameter in forward().
        """
        # Check if instrument has the illumination_center parameter
        # This is available in Microscope but not Telescope
        return hasattr(self.instrument, "_forward_scanning_illumination")

    @property
    def scanning_mode(self) -> ScanningMode:
        """Get the current scanning mode.

        Returns
        -------
        ScanningMode
            Current scanning mode (APERTURE or ILLUMINATION).
        """
        return self.config.scanning_mode

    def add_mask(
        self,
        centers: Union[Tensor, List[List[float]], None] = None,
        radius: Optional[float] = None,
        line_endpoints: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> None:
        """Add new aperture/illumination position(s) to cumulative mask.

        Updates the cumulative mask to include new aperture or illumination
        positions. This should be called after each successful training step
        to record which k-space regions have been measured.

        The behavior depends on the configured scanning_mode:

        - **APERTURE mode** (default): Centers specify sub-aperture positions
          in pixel coordinates. The mask is placed at these positions.
        - **ILLUMINATION mode**: Centers specify illumination positions in pixel
          coordinates. By the Fourier shift theorem, illumination at position k
          samples the same k-space region as an aperture at that position, so
          the mask is placed at the same positions. For finite-size sources
          (GAUSSIAN, CIRCULAR), the mask radius uses the configured
          illumination_radius converted to pixel units.

        Parameters
        ----------
        centers : Tensor or List[List[float]], optional
            Position centers as Tensor or [[y0, x0], [y1, x1], ...] in pixel
            coordinates from DC. In both modes, these specify which k-space
            regions are being sampled.
        radius : float, optional
            Aperture radius override in pixels. If None:
            - APERTURE mode: Uses instrument's default aperture radius
            - ILLUMINATION mode: Uses illumination_radius from config
              (converted to pixels) if set, otherwise uses default
        line_endpoints : Tuple[Tensor, Tensor], optional
            (start, end) tuple for line acquisition mode.

        Notes
        -----
        - Uses logical OR to combine masks (union of all positions)
        - Increments sample_count for tracking
        - Operations done without gradients
        - In illumination mode, the cumulative mask still represents
          k-space coverage, just built from illumination positions

        Examples
        --------
        Aperture mode (default):

        >>> measurement_system.add_mask([[0, 0]])  # Add first aperture
        >>> measurement_system.add_mask([[10, 10], [20, 20]])  # Add two more

        Illumination mode (configured in MeasurementSystemConfig):

        >>> # MeasurementSystem configured with ScanningMode.ILLUMINATION
        >>> measurement_system.add_mask([[10, 5]])  # Same k-space sampling

        Line mode:

        >>> start = torch.tensor([0.0, 0.0])
        >>> end = torch.tensor([10.0, 10.0])
        >>> measurement_system.add_mask(line_endpoints=(start, end))

        See Also
        --------
        ScanningMode : Enum defining available scanning modes
        """
        with torch.no_grad():
            # Line acquisition mode
            if line_endpoints is not None:
                if self.line_acquisition is None:
                    raise ValueError(
                        "Line endpoints provided but no line_acquisition module configured"
                    )
                start, end = line_endpoints
                line_mask = self.line_acquisition.generate_line_mask(start, end)
                self.cum_mask = torch.logical_or(self.cum_mask, line_mask)
                self.sample_count += 1
                return

            # Point acquisition mode
            if centers is None:
                raise ValueError("Either centers or line_endpoints must be provided")

            # Convert centers to list if tensor
            if isinstance(centers, Tensor):
                centers_list = centers.tolist()
            else:
                centers_list = centers

            # For spatial illumination, compute effective k-shift for mask placement
            if (
                self.config.scanning_mode == ScanningMode.ILLUMINATION
                and self.config.illumination_scan_method == IlluminationScanMethod.SPATIAL
            ):
                # Convert spatial positions to effective k-space shifts
                from .optics.fourier_utils import spatial_position_to_effective_k

                # Ensure source_distance is set
                if self.config.illumination_source_distance is None:
                    raise ValueError(
                        "illumination_source_distance must be set for SPATIAL illumination mode"
                    )

                effective_centers = []
                for pixel_center in centers_list:
                    spatial_pos = self._pixel_to_spatial(pixel_center)
                    k_shift = spatial_position_to_effective_k(
                        spatial_pos,
                        self.config.illumination_source_distance,
                        self.instrument.grid.wl,
                    )
                    # Convert k-shift back to pixels for mask
                    k_to_pixel_center = self._k_shift_to_pixel(k_shift)
                    effective_centers.append(list(k_to_pixel_center))
                centers_list = effective_centers

            # Determine mask radius based on mode
            mask_radius = self._get_mask_radius(radius)

            # Generate masks for all centers
            # Note: In both APERTURE and ILLUMINATION modes, the mask is placed
            # at the same pixel position because illumination at k samples the
            # same k-space region as an aperture at k (by Fourier shift theorem)
            if hasattr(self.instrument, "generate_aperture_masks"):
                # Use batch generation if available (e.g., Telescope)
                new_masks = self.instrument.generate_aperture_masks(
                    centers_list, radius=mask_radius
                )
                # Combine all new masks with cumulative mask
                combined_new_mask = new_masks.any(dim=0)
                self.cum_mask = torch.logical_or(self.cum_mask, combined_new_mask)
            else:
                # Fallback: generate individual masks
                for center in centers_list:
                    if hasattr(self.instrument, "generate_aperture_mask"):
                        mask = self.instrument.generate_aperture_mask(center, radius=mask_radius)
                        self.cum_mask = torch.logical_or(self.cum_mask, mask)

            self.sample_count += len(centers_list)

    def _get_mask_radius(self, radius_override: Optional[float] = None) -> Optional[float]:
        """Determine the mask radius based on mode and configuration.

        Parameters
        ----------
        radius_override : float, optional
            Explicit radius override. If provided, always uses this value.

        Returns
        -------
        float or None
            Mask radius in pixels, or None to use instrument default.

        Notes
        -----
        In ILLUMINATION mode with finite sources, the illumination_radius
        (in k-space 1/m units) is converted to pixel units using the grid.
        This ensures the cumulative mask reflects the actual k-space coverage.

        For GAUSSIAN sources, the mask radius is 2*sigma to capture ~95% of
        the Gaussian intensity. For CIRCULAR sources, the radius is used
        directly as it represents the actual cutoff.
        """
        # Explicit override takes priority
        if radius_override is not None:
            return radius_override

        # For ILLUMINATION mode, compute mask radius based on source type
        if self.config.scanning_mode == ScanningMode.ILLUMINATION:
            return self._compute_illumination_mask_radius()

        # Default: let instrument use its default
        return None

    def _compute_illumination_mask_radius(self) -> Optional[float]:
        """Compute mask radius for illumination mode based on source type.

        Returns
        -------
        float or None
            Mask radius in pixels, or None to use instrument default.

        Notes
        -----
        The mask radius interpretation depends on the source type:

        - **POINT**: No finite extent, use instrument default aperture radius.
          Point sources sample a region determined by the detection aperture.

        - **GAUSSIAN**: Uses 2*sigma to capture ~95% of the Gaussian intensity.
          The illumination_radius config parameter is the sigma (standard
          deviation) in k-space units.

        - **CIRCULAR**: Uses the radius directly as it represents the sharp
          cutoff of the circular top-hat profile.
        """
        if self.config.illumination_radius is None:
            # POINT source or no radius specified - use instrument default
            return None

        # Get base k-space radius
        k_radius = self.config.illumination_radius

        # Scale based on source type
        source_type = self.config.illumination_source_type

        if source_type == "GAUSSIAN":
            # For Gaussian, use 2*sigma to capture ~95% of intensity
            # (2 sigma captures 95.45% of a 2D Gaussian)
            effective_k_radius = 2.0 * k_radius
        elif source_type == "CIRCULAR":
            # For circular, use the radius directly (sharp cutoff)
            effective_k_radius = k_radius
        else:
            # POINT or unknown - use instrument default
            return None

        # Convert k-space radius to pixel units
        return self._k_radius_to_pixels(effective_k_radius)

    def _k_radius_to_pixels(self, k_radius: float) -> float:
        """Convert k-space radius to pixel units.

        Parameters
        ----------
        k_radius : float
            Radius in k-space (1/meters).

        Returns
        -------
        float
            Radius in pixel units.
        """
        if not hasattr(self.instrument, "grid"):
            raise ValueError("Cannot convert k-radius to pixels: instrument has no grid attribute")

        grid = self.instrument.grid

        # k-space resolution: dk = 1 / (n * dx) = 1 / FOV
        # So pixels = k_radius / dk = k_radius * (n * dx)
        dk = 1.0 / (grid.nx * grid.dx)
        return k_radius / dk

    def measure_through_accumulated_mask(self, field: Tensor) -> Tensor:
        """Measure field through accumulated mask.

        Simulates measuring the current reconstruction through all previously
        sampled apertures. Used to ensure reconstruction remains consistent
        with all prior measurements.

        Args:
            field: Field to measure (typically current reconstruction)

        Returns:
            Measurement through cumulative mask

        Example:
            >>> old_meas = measurement_system.measure_through_accumulated_mask(reconstruction)
        """
        # Propagate to k-space
        if hasattr(self.instrument, "propagate_to_kspace"):
            field_kspace = self.instrument.propagate_to_kspace(field)
        else:
            # Fallback: use instrument's forward method
            field_kspace = field

        # Apply cumulative mask
        field_kspace_masked = field_kspace * self.cum_mask

        # Propagate back to spatial domain
        measurement: Tensor
        if hasattr(self.instrument, "propagate_to_spatial"):
            measurement = self.instrument.propagate_to_spatial(field_kspace_masked)
        else:
            # Fallback: use abs
            measurement = torch.abs(field_kspace_masked)

        return measurement

    def _pixel_to_k_shift(self, pixel_center: List[float]) -> List[float]:
        """Convert pixel center to k-space shift.

        Uses the instrument's grid (if available) to convert from pixel
        coordinates to k-space coordinates in 1/meters.

        Parameters
        ----------
        pixel_center : List[float]
            Center position [py, px] in pixel units from DC.

        Returns
        -------
        List[float]
            k-space shift [ky, kx] in 1/meters.
        """
        # Import here to avoid circular imports
        from .optics.fourier_utils import pixel_to_k_shift

        # Get grid from instrument
        if hasattr(self.instrument, "grid"):
            grid = self.instrument.grid
            ky, kx = pixel_to_k_shift(pixel_center, grid)
            return [ky, kx]
        else:
            raise ValueError(
                "ILLUMINATION mode requires instrument with grid attribute. "
                f"Got {type(self.instrument).__name__}."
            )

    def _get_illumination_source_type(self) -> "IlluminationSourceType":
        """Get IlluminationSourceType enum from config string.

        Returns
        -------
        IlluminationSourceType
            The corresponding enum value.
        """
        from .optics.illumination import IlluminationSourceType

        source_type_map = {
            "POINT": IlluminationSourceType.POINT,
            "GAUSSIAN": IlluminationSourceType.GAUSSIAN,
            "CIRCULAR": IlluminationSourceType.CIRCULAR,
        }
        return source_type_map[self.config.illumination_source_type]

    def get_measurements(
        self,
        field: Tensor,
        aperture_centers: Union[Tensor, List[List[float]]],
        add_noise: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """Generate measurements through specified apertures/illuminations.

        Uses the instrument's forward method to generate measurements. The behavior
        depends on the configured scanning_mode:

        - APERTURE mode: Uses `aperture_center` parameter (default behavior)
        - ILLUMINATION mode: Converts pixel centers to k-space and uses
          `illumination_center` parameter

        Parameters
        ----------
        field : Tensor
            Input field to measure.
        aperture_centers : Tensor or List[List[float]]
            Center positions in pixel coordinates [y, x] from DC.
            In ILLUMINATION mode, these are converted to k-space coordinates.
        add_noise : bool
            Whether to add noise. Default False.
        **kwargs : Any
            Additional instrument-specific parameters.

        Returns
        -------
        Tensor
            Measurements through specified apertures/illuminations.

        Examples
        --------
        Aperture mode (default):

        >>> meas = measurement_system.get_measurements(field, [[0, 0], [10, 10]])

        Illumination mode (configured in MeasurementSystemConfig):

        >>> # MeasurementSystem configured with ScanningMode.ILLUMINATION
        >>> meas = measurement_system.get_measurements(field, [[10, 5]])
        >>> # Centers are converted to k-space and used as illumination positions
        """
        # Convert centers to list if needed
        if isinstance(aperture_centers, Tensor):
            centers_list = aperture_centers.tolist()
        else:
            centers_list = list(aperture_centers)

        # Route based on scanning mode and method
        if self.config.scanning_mode == ScanningMode.ILLUMINATION:
            if self.config.illumination_scan_method == IlluminationScanMethod.SPATIAL:
                return self._get_measurements_spatial_illumination(
                    field, centers_list, add_noise, **kwargs
                )
            else:  # ANGULAR (default, existing behavior)
                return self._get_measurements_illumination(field, centers_list, add_noise, **kwargs)
        else:  # APERTURE
            return self._get_measurements_aperture(field, centers_list, add_noise, **kwargs)

    def _get_measurements_aperture(
        self,
        field: Tensor,
        centers_list: List[List[float]],
        add_noise: bool,
        **kwargs: Any,
    ) -> Tensor:
        """Generate measurements using scanning aperture mode.

        Parameters
        ----------
        field : Tensor
            Input field to measure.
        centers_list : List[List[float]]
            Aperture center positions in pixel coordinates.
        add_noise : bool
            Whether to add noise.
        **kwargs : Any
            Additional instrument-specific parameters.

        Returns
        -------
        Tensor
            Measurements through specified apertures.
        """
        measurements = []
        for center in centers_list:
            meas = self.instrument.forward(
                field,
                aperture_center=center,
                add_noise=add_noise,
                **kwargs,
            )
            measurements.append(meas)

        if len(measurements) == 1:
            return measurements[0]
        else:
            return torch.stack(measurements)

    def _get_measurements_illumination(
        self,
        field: Tensor,
        centers_list: List[List[float]],
        add_noise: bool,
        **kwargs: Any,
    ) -> Tensor:
        """Generate measurements using scanning illumination mode.

        Converts pixel centers to k-space coordinates and uses the instrument's
        illumination_center parameter for FPM-style measurements.

        Parameters
        ----------
        field : Tensor
            Input field to measure.
        centers_list : List[List[float]]
            Center positions in pixel coordinates (converted to k-space).
        add_noise : bool
            Whether to add noise.
        **kwargs : Any
            Additional instrument-specific parameters.

        Returns
        -------
        Tensor
            Measurements through specified illumination positions.
        """
        # Get illumination parameters from config
        source_type = self._get_illumination_source_type()
        illum_radius = self.config.illumination_radius

        measurements = []
        for pixel_center in centers_list:
            # Convert pixel center to k-space
            k_center = self._pixel_to_k_shift(pixel_center)

            # Call instrument with illumination parameters
            meas = self.instrument.forward(
                field,
                illumination_center=k_center,
                illumination_radius=illum_radius,
                illumination_source_type=source_type,
                add_noise=add_noise,
                **kwargs,
            )
            measurements.append(meas)

        if len(measurements) == 1:
            return measurements[0]
        else:
            return torch.stack(measurements)

    def _get_measurements_spatial_illumination(
        self,
        field: Tensor,
        centers_list: List[List[float]],
        add_noise: bool,
        **kwargs: Any,
    ) -> Tensor:
        """Generate measurements using spatial illumination scanning.

        Centers are specified in pixel units and converted to physical
        coordinates (meters) using the grid.

        Parameters
        ----------
        field : Tensor
            Input field at object plane.
        centers_list : List[List[float]]
            List of centers [py, px] in pixel units from DC.
        add_noise : bool
            Whether to add detector noise.

        Returns
        -------
        Tensor
            Measurements, shape depends on number of centers.
        """
        source_type = self._get_illumination_source_type()
        source_distance = self.config.illumination_source_distance

        measurements = []
        for pixel_center in centers_list:
            # Convert pixel center to spatial coordinates (meters)
            spatial_center = self._pixel_to_spatial(pixel_center)

            # Call instrument's spatial illumination forward
            meas = self.instrument.forward(
                field,
                illumination_spatial_center=list(spatial_center),
                illumination_source_distance=source_distance,
                illumination_radius=self.config.illumination_radius,
                illumination_source_type=source_type,
                add_noise=add_noise,
                **kwargs,
            )
            measurements.append(meas)

        if len(measurements) == 1:
            return measurements[0]
        return torch.stack(measurements)

    def _pixel_to_spatial(self, pixel_center: List[float]) -> Tuple[float, float]:
        """Convert pixel center to spatial coordinates (meters)."""
        from .optics.fourier_utils import pixel_to_spatial

        return pixel_to_spatial(pixel_center, self.instrument.grid)

    def _k_shift_to_pixel(self, k_shift: Tuple[float, float]) -> Tuple[float, float]:
        """Convert k-space shift to pixel units.

        Parameters
        ----------
        k_shift : Tuple[float, float]
            k-space shift [ky, kx] in 1/meters.

        Returns
        -------
        Tuple[float, float]
            Pixel shift [py, px] from DC.
        """
        from .optics.fourier_utils import k_shift_to_pixel

        return k_shift_to_pixel(k_shift, self.instrument.grid)

    def _measure_line(
        self,
        ground_truth: Tensor,
        reconstruction: Optional[Tensor],
        line_endpoints: Tuple[Tensor, Tensor],
        add_noise: bool,
    ) -> Tensor:
        """Internal method for line acquisition measurements.

        Args:
            ground_truth: Ground truth object field
            reconstruction: Current reconstruction (None for first sample)
            line_endpoints: (start, end) tuple of [2] tensors
            add_noise: Whether to add noise

        Returns:
            Tensor: Measurements [2, H, W]
        """
        if self.line_acquisition is None:
            raise ValueError("Line endpoints provided but no line_acquisition module configured")

        start, end = line_endpoints

        # Propagate to k-space
        if hasattr(self.instrument, "propagate_to_kspace"):
            field_kspace = self.instrument.propagate_to_kspace(ground_truth)
        else:
            field_kspace = ground_truth

        # Compute new measurement through line
        new_meas = self.line_acquisition.forward(field_kspace, start, end, add_noise=add_noise)

        # Combine with accumulated mask measurement
        if reconstruction is None or self.sample_count == 0:
            # First sample: use new measurement for both
            dual_meas = torch.stack([new_meas, new_meas], dim=0)
        else:
            # Later samples: old mask + new measurement
            old_meas = self.measure_through_accumulated_mask(reconstruction)
            dual_meas = torch.stack([old_meas, new_meas], dim=0)

        # Ensure 4D output [2, C, H, W] for loss function compatibility
        if dual_meas.ndim == 3:
            dual_meas = dual_meas.unsqueeze(1)  # Add channel dimension

        return dual_meas.detach().clone()

    def measure(
        self,
        ground_truth: Tensor,
        reconstruction: Optional[Tensor] = None,
        centers: Union[Tensor, List[List[float]], None] = None,
        add_noise: Optional[bool] = None,
        line_endpoints: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs: Any,
    ) -> Tensor:
        """Measurement mode for SPIDS training.

        Generates dual measurements:
        - If first sample: [new_meas, new_meas]
        - Otherwise: [reconstruction_through_cum_mask, ground_truth_through_new_sample]

        The loss function compares model output against both measurements.

        The behavior depends on the configured scanning_mode:

        - **APERTURE mode** (default): Centers specify sub-aperture positions in
          pixel coordinates. Uses `aperture_center` parameter in instrument.forward().
        - **ILLUMINATION mode**: Centers specify illumination positions in pixel
          coordinates, which are automatically converted to k-space. Uses
          `illumination_center` parameter for FPM-style measurements.

        Parameters
        ----------
        ground_truth : Tensor
            Ground truth object field.
        reconstruction : Tensor, optional
            Current reconstruction. None for first sample.
        centers : Tensor or List[List[float]], optional
            Sample centers in pixel coordinates [y, x] from DC.
            Defaults to [[0, 0]]. In ILLUMINATION mode, automatically
            converted to k-space coordinates.
        add_noise : bool, optional
            Whether to add noise. Uses config default if None.
        line_endpoints : Tuple[Tensor, Tensor], optional
            (start, end) tuple for line acquisition mode.
        **kwargs : Any
            Additional instrument parameters.

        Returns
        -------
        Tensor
            Measurements [2, H, W] or [2, 1, H, W] for loss computation.

        Examples
        --------
        First sample (aperture mode):

        >>> meas = measurement_system.measure(ground_truth, None, [[0, 0]])
        >>> # meas[0] == meas[1] (both new measurement)

        Later samples:

        >>> meas = measurement_system.measure(ground_truth, reconstruction, [[10, 10]])
        >>> # meas[0] = reconstruction through cum_mask
        >>> # meas[1] = ground_truth through new aperture/illumination

        With illumination mode (configured in MeasurementSystemConfig):

        >>> # MeasurementSystem configured with ScanningMode.ILLUMINATION
        >>> meas = measurement_system.measure(ground_truth, None, [[10, 5]])
        >>> # Centers converted to k-space for illumination positioning

        Line acquisition mode:

        >>> start = torch.tensor([0.0, 0.0])
        >>> end = torch.tensor([10.0, 10.0])
        >>> meas = measurement_system.measure(
        ...     ground_truth, reconstruction, line_endpoints=(start, end)
        ... )

        See Also
        --------
        get_measurements : Generate measurements without dual output
        ScanningMode : Available scanning modes
        """
        # Set defaults
        if add_noise is None:
            add_noise = self.config.add_noise_by_default

        # Dispatch to line acquisition mode
        if line_endpoints is not None:
            return self._measure_line(ground_truth, reconstruction, line_endpoints, add_noise)

        # Point acquisition mode (supports both APERTURE and ILLUMINATION modes)
        with torch.no_grad():
            # Set defaults
            if centers is None:
                centers = [[0, 0]]

            # Convert to list for cache key
            if isinstance(centers, Tensor):
                centers_list = centers.tolist()
            else:
                centers_list = list(centers)

            # Try cache for ground truth measurement (if enabled)
            cached_meas = None
            if self.measurement_cache is not None:
                # Use cache with default telescope parameters
                # (r, is_sum, sum_pattern are legacy telescope params, use None defaults)
                cached_meas = self.measurement_cache.get(
                    tensor=ground_truth,
                    centers=centers_list,
                    r=None,  # Legacy parameter, not used in new system
                    is_sum=False,  # Legacy parameter, not used in new system
                    sum_pattern=None,  # Legacy parameter, not used in new system
                )

            # Compute new measurement through aperture
            if cached_meas is not None:
                # Cache hit
                new_meas_no_noise = cached_meas
            else:
                # Cache miss - compute measurement
                new_meas_no_noise = self.get_measurements(
                    ground_truth, centers_list, add_noise=False, **kwargs
                )

                # Store in cache (before noise)
                if self.measurement_cache is not None:
                    self.measurement_cache.put(
                        tensor=ground_truth,
                        centers=centers_list,
                        r=None,  # Legacy parameter
                        is_sum=False,  # Legacy parameter
                        sum_pattern=None,  # Legacy parameter
                        measurement=new_meas_no_noise,
                    )

            # Add noise after cache (fresh noise each time)
            if (
                add_noise
                and hasattr(self.instrument, "noise_model")
                and self.instrument.noise_model
            ):
                new_meas = self.instrument.noise_model(new_meas_no_noise, add_noise=True)
            else:
                new_meas = new_meas_no_noise

            # Combine with accumulated mask measurement
            if reconstruction is None or self.sample_count == 0:
                # First sample: use new measurement for both
                dual_meas = torch.stack([new_meas, new_meas], dim=0)
            else:
                # Later samples: old mask + new measurement
                old_meas = self.measure_through_accumulated_mask(reconstruction)
                dual_meas = torch.stack([old_meas, new_meas], dim=0)

            # Ensure 4D output [2, C, H, W] for loss function compatibility
            if dual_meas.ndim == 3:
                dual_meas = dual_meas.unsqueeze(1)  # Add channel dimension

            return dual_meas.detach().clone()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get measurement cache statistics.

        Returns:
            dict: Cache statistics including hit rate, size, etc.
                  Empty dict if caching disabled.

        Example:
            >>> stats = measurement_system.get_cache_stats()
            >>> print(f"Hit rate: {stats.get('hit_rate', 0):.1%}")
        """
        if self.measurement_cache is not None:
            return self.measurement_cache.get_stats()
        return {}

    def clear_cache(self) -> None:
        """Clear measurement cache and reset statistics.

        Example:
            >>> measurement_system.clear_cache()
        """
        if self.measurement_cache is not None:
            self.measurement_cache.clear_cache()

    def reset(self) -> None:
        """Reset cumulative mask and sample count.

        Clears all accumulated measurements. Use when starting a new
        reconstruction or training run.

        Example:
            >>> measurement_system.reset()
            >>> measurement_system.sample_count  # 0
        """
        self.cum_mask = torch.zeros_like(self.cum_mask)
        self.sample_count = 0

    def to(self, device: torch.device) -> "MeasurementSystem":
        """Move measurement system to specified device.

        Args:
            device: Target device

        Returns:
            Self for chaining

        Example:
            >>> measurement_system.to(torch.device("cuda"))
        """
        self.cum_mask = self.cum_mask.to(device)
        if hasattr(self.instrument, "to"):
            self.instrument.to(device)
        return self

    def get_info(self) -> Dict[str, Any]:
        """Get measurement system information.

        Returns
        -------
        dict
            System information including:
            - sample_count: Number of accumulated measurements
            - scanning_mode: Current scanning mode (APERTURE or ILLUMINATION)
            - caching_enabled: Whether caching is enabled
            - cumulative_mask_coverage: Fraction of k-space covered
            - instrument: Instrument-specific info
            - cache: Cache statistics (if enabled)
            - illumination_config: Illumination parameters (if ILLUMINATION mode)

        Example
        -------
        >>> info = measurement_system.get_info()
        >>> print(f"Samples: {info['sample_count']}")
        >>> print(f"Mode: {info['scanning_mode']}")
        """
        info: Dict[str, Any] = {
            "sample_count": self.sample_count,
            "scanning_mode": self.config.scanning_mode.name,
            "caching_enabled": self.config.enable_caching,
            "cumulative_mask_coverage": self.cum_mask.sum().item() / self.cum_mask.numel(),
        }

        # Add illumination configuration if in ILLUMINATION mode
        if self.config.scanning_mode == ScanningMode.ILLUMINATION:
            info["illumination_config"] = {
                "source_type": self.config.illumination_source_type,
                "radius": self.config.illumination_radius,
            }

        # Add instrument info
        if hasattr(self.instrument, "get_info"):
            info["instrument"] = self.instrument.get_info()

        # Add cache stats if available
        if self.measurement_cache is not None:
            info["cache"] = self.get_cache_stats()

        return info

    @property
    def n(self) -> Tensor:
        """Image size as tensor (backward compatibility with TelescopeAggregator).

        Returns:
            Tensor: Image size (n_pixels) as 0D tensor

        Example:
            >>> n_pixels = int(measurement_system.n.item())
        """
        return torch.tensor(self.instrument.config.n_pixels)

    def __call__(
        self,
        inputs: Tensor,
        centers: Union[Tensor, List[List[float]], None] = None,
    ) -> Tensor:
        """Generate measurements for loss computation (backward compatibility).

        This method provides backward compatibility with LossAggregator,
        which expects: telescope(inputs, centers) → [2, C, H, W].

        The method generates:
        - measurement[0]: inputs through cumulative mask (old measurements)
        - measurement[1]: inputs through new aperture (new measurement)

        Args:
            inputs: Model output (reconstruction) [B, C, H, W]
            centers: Aperture center positions for new measurement

        Returns:
            Tensor: Measurements [2, C, H, W] for loss computation

        Example:
            >>> # Used by LossAggregator
            >>> measurements = measurement_system(reconstruction, center)
            >>> # measurements[0] = reconstruction through cum_mask
            >>> # measurements[1] = reconstruction through new aperture
        """
        if centers is None:
            centers = [[0.0, 0.0]]

        # Convert centers to list if tensor
        if isinstance(centers, Tensor):
            centers_list = centers.tolist()
        else:
            centers_list = list(centers)

        # Ensure 4D input
        if inputs.ndim == 2:
            inputs = inputs.unsqueeze(0).unsqueeze(0)
        elif inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)

        # Old measurement: through cumulative mask
        old_meas = self.measure_through_accumulated_mask(inputs)

        # New measurement: through specified apertures
        new_meas = self.get_measurements(inputs, centers_list, add_noise=False)

        # Ensure consistent shape
        if old_meas.ndim == 2:
            old_meas = old_meas.unsqueeze(0)
        if new_meas.ndim == 2:
            new_meas = new_meas.unsqueeze(0)

        # For line sampling: multiple centers are averaged to simulate motion blur
        # new_meas shape: [N, C, H, W] for N centers along line → average to [C, H, W]
        if new_meas.shape[0] != old_meas.shape[0]:
            # Average all measurements along the line
            new_meas = new_meas.mean(dim=0, keepdim=True)

        # Stack [old, new]
        result = torch.stack([old_meas, new_meas], dim=0)

        return result

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing.

        Returns:
            Dict containing cum_mask and sample_count for serialization.

        Example:
            >>> state = measurement_system.state_dict()
            >>> torch.save(state, 'measurement_system.pt')
        """
        return {
            "cum_mask": self.cum_mask,
            "sample_count": self.sample_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary from checkpoint.

        Args:
            state_dict: Dictionary containing cum_mask and sample_count

        Example:
            >>> state = torch.load('measurement_system.pt')
            >>> measurement_system.load_state_dict(state)
        """
        if "cum_mask" in state_dict:
            self.cum_mask = state_dict["cum_mask"]
        if "sample_count" in state_dict:
            self.sample_count = state_dict["sample_count"]

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MeasurementSystem("
            f"instrument={self.instrument.__class__.__name__}, "
            f"mode={self.config.scanning_mode.name}, "
            f"samples={self.sample_count}, "
            f"coverage={self.cum_mask.sum().item() / self.cum_mask.numel():.2%})"
        )
