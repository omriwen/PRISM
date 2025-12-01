"""Test target generation for optical system validation.

This module provides abstract base classes and concrete implementations
for generating standard optical test patterns, particularly the USAF-1951
resolution chart used for calibrating and validating optical systems.

Key Features:
- Physical field size specification (meters)
- Configurable margins for proper diffraction simulation
- Multiple target types: USAF-1951, Checkerboard, Siemens Star
- Automatic scaling between pixel and physical coordinates

Important Note on Margins:
    For correct diffraction/propagation simulations, targets should have
    generous zero-value margins around the actual pattern. This prevents:
    1. Periodic boundary condition artifacts in FFT-based propagators
    2. Beam divergence clipping

    Default margin_ratio=0.25 means the actual pattern occupies the central
    50% of the image (25% margin on each side). For critical simulations,
    use margin_ratio=0.35 or higher.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class TargetConfig:
    """Configuration for test target generation.

    Attributes:
        size: Image size in pixels (square image)
        field_size: Physical field size in meters (optional, for physical scaling)
        margin_ratio: Ratio of image to use as zero-margin on each side (0.0-0.4)
                     Default 0.25 means actual pattern uses central 50% of image.
                     IMPORTANT: Set to 0 only if you're certain you don't need
                     proper diffraction margins.
        contrast: Contrast ratio (0-1), default 1.0 for maximum
        polarity: 'positive' (dark bars on bright) or 'negative' (bright on dark)
        device: PyTorch device ('cpu', 'cuda', etc.)
        background: Background intensity (0-1), default 0.0
        foreground: Foreground intensity (0-1), default 1.0
    """

    size: int = 1024
    field_size: Optional[float] = None  # Physical field size in meters
    margin_ratio: float = 0.25  # 25% margin on each side = 50% active area
    contrast: float = 1.0
    polarity: str = "positive"
    device: str = "cpu"
    background: float = 0.0
    foreground: float = 1.0

    def __post_init__(self) -> None:
        if self.contrast < 0 or self.contrast > 1:
            raise ValueError(f"Contrast must be in [0, 1], got {self.contrast}")
        if self.polarity not in ("positive", "negative"):
            raise ValueError(f"Polarity must be 'positive' or 'negative', got {self.polarity}")
        if self.margin_ratio < 0 or self.margin_ratio > 0.45:
            raise ValueError(f"margin_ratio must be in [0, 0.45], got {self.margin_ratio}")

        # Warn if margin is too small for diffraction
        if self.margin_ratio < 0.15:
            warnings.warn(
                f"margin_ratio={self.margin_ratio} may be insufficient for proper "
                "diffraction simulation. Consider using margin_ratio >= 0.25 for "
                "FFT-based propagators to avoid periodic boundary artifacts.",
                UserWarning,
            )

    @property
    def pixel_size(self) -> Optional[float]:
        """Physical size of each pixel in meters."""
        if self.field_size is not None:
            return self.field_size / self.size
        return None

    @property
    def active_size(self) -> int:
        """Size of the active (non-margin) region in pixels."""
        return int(self.size * (1 - 2 * self.margin_ratio))

    @property
    def active_field_size(self) -> Optional[float]:
        """Physical size of the active region in meters."""
        if self.field_size is not None:
            return self.field_size * (1 - 2 * self.margin_ratio)
        return None

    @property
    def margin_pixels(self) -> int:
        """Number of margin pixels on each side."""
        return int(self.size * self.margin_ratio)


class Target(ABC):
    """Abstract base class for test targets.

    All test targets must implement:
    - generate(): Create the target pattern tensor
    - resolution_elements: Property listing resolution elements
    """

    def __init__(self, config: TargetConfig):
        self.config = config

    @abstractmethod
    def generate(self) -> torch.Tensor:
        """Generate target pattern.

        Returns:
            torch.Tensor: Target image of shape (H, W) with values in [0, 1]
        """
        pass

    @property
    @abstractmethod
    def resolution_elements(self) -> dict:
        """Get resolution element information.

        Returns:
            dict: Mapping of element identifiers to spatial frequencies
        """
        pass

    def _apply_contrast_and_polarity(self, mask: torch.Tensor) -> torch.Tensor:
        """Apply contrast and polarity to binary mask.

        Args:
            mask: Boolean tensor (True = foreground, False = background)

        Returns:
            Grayscale image with applied contrast and polarity
        """
        # Convert to float
        image = mask.float()

        # Apply contrast
        fg = self.config.foreground
        bg = self.config.background
        image = image * (fg - bg) * self.config.contrast + bg

        # Apply polarity
        if self.config.polarity == "negative":
            image = 1.0 - image

        return image

    def _get_active_region_coords(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get coordinate grids for the active (non-margin) region.

        Returns:
            Tuple of (y_coords, x_coords) tensors in the range [-0.5, 0.5]
            representing normalized coordinates within the active region.
        """
        size = self.config.size
        active_size = self.config.active_size
        device = self.config.device

        # Create full coordinate grid
        y = torch.arange(size, device=device, dtype=torch.float32)
        x = torch.arange(size, device=device, dtype=torch.float32)

        # Normalize to [-0.5, 0.5] relative to active region center
        center = size / 2
        y_norm = (y - center) / active_size
        x_norm = (x - center) / active_size

        yy, xx = torch.meshgrid(y_norm, x_norm, indexing="ij")
        return yy, xx

    def _is_in_active_region(self, yy: torch.Tensor, xx: torch.Tensor) -> torch.Tensor:
        """Check if coordinates are within the active region.

        Args:
            yy, xx: Coordinate grids from _get_active_region_coords

        Returns:
            Boolean mask of points within active region
        """
        return (torch.abs(yy) <= 0.5) & (torch.abs(xx) <= 0.5)

    def validate_for_propagation(
        self, propagation_distance: float, wavelength: float, check_fresnel: bool = True
    ) -> Dict[str, Any]:
        """Validate target is suitable for diffraction propagation.

        Args:
            propagation_distance: Distance light will propagate (meters)
            wavelength: Wavelength of light (meters)
            check_fresnel: Whether to check Fresnel number criteria

        Returns:
            Dictionary with validation results and recommendations
        """
        result: Dict[str, Any] = {"valid": True, "warnings": [], "recommendations": []}

        if self.config.field_size is None:
            result["warnings"].append(
                "No physical field_size specified - cannot validate propagation parameters"
            )
            return result

        # Check margin adequacy
        margin_physical = self.config.margin_ratio * self.config.field_size
        beam_spread = wavelength * propagation_distance / margin_physical

        if beam_spread > margin_physical * 0.5:
            result["valid"] = False
            result["warnings"].append(
                f"Margin ({margin_physical * 1e6:.1f} µm) may be insufficient for "
                f"beam spread (~{beam_spread * 1e6:.1f} µm) at distance {propagation_distance * 1e3:.1f} mm"
            )
            result["recommendations"].append(
                "Increase margin_ratio or field_size, or reduce propagation_distance"
            )

        # Fresnel number check
        if check_fresnel:
            active_size = self.config.active_field_size
            if active_size is not None:
                fresnel = active_size**2 / (wavelength * propagation_distance)
                result["fresnel_number"] = fresnel
                if fresnel > 100:
                    result["recommendations"].append(
                        f"High Fresnel number ({fresnel:.1f}) suggests geometric optics may suffice"
                    )
                elif fresnel < 0.1:
                    result["recommendations"].append(
                        f"Low Fresnel number ({fresnel:.3f}) indicates far-field/Fraunhofer regime"
                    )

        return result


@dataclass
class USAF1951Config(TargetConfig):
    """Configuration for USAF-1951 resolution target.

    Additional attributes:
        groups: Tuple of group numbers to include (e.g., (0, 1, 2, 3, 4))
        elements_per_group: Number of elements per group (standard is 6)
        layout: 'standard' (full chart) or 'single_group' (one group only)
    """

    groups: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    elements_per_group: int = 6
    layout: str = "standard"

    @classmethod
    def from_physical_params(
        cls,
        field_size: float,
        resolution: int = 1024,
        groups: Optional[Tuple[int, ...]] = None,
        margin_ratio: float = 0.25,
        **kwargs: Any,
    ) -> "USAF1951Config":
        """Create config from physical parameters.

        Args:
            field_size: Physical field of view in meters
            resolution: Number of pixels
            groups: USAF groups to include (auto-selected if None)
            margin_ratio: Margin ratio for diffraction
            **kwargs: Additional TargetConfig parameters

        Returns:
            Configured USAF1951Config instance
        """
        if groups is None:
            # Auto-select groups based on field size
            # Smallest feature should be ~2-3 pixels
            pixel_size = field_size / resolution
            min_feature_size = pixel_size * 3  # At least 3 pixels
            active_field = field_size * (1 - 2 * margin_ratio)

            # Find appropriate groups
            # USAF bar width = 1 / (2 * 2^(group + (element-1)/6)) mm
            selected_groups = []
            for g in range(-2, 10):
                bar_width_mm = 1.0 / (2.0 * (2**g))  # Element 1
                bar_width_m = bar_width_mm * 1e-3

                # Check if this group's features are resolvable
                if bar_width_m >= min_feature_size and bar_width_m * 20 <= active_field:
                    selected_groups.append(g)

            if not selected_groups:
                # Default fallback
                selected_groups = [4, 5, 6, 7]

            groups = tuple(selected_groups)

        return cls(
            size=resolution,
            field_size=field_size,
            margin_ratio=margin_ratio,
            groups=groups,
            **kwargs,
        )


class USAF1951Target(Target):
    """USAF-1951 resolution target generator.

    Generates the standard USAF-1951 resolution chart with groups and elements
    arranged in the standard 6-element per group layout.

    The USAF-1951 chart consists of groups of 3-bar patterns at decreasing sizes.
    Each group has 6 elements, with spatial frequency doubling between groups.

    Example:
        >>> config = USAF1951Config(size=1024, groups=(0, 1, 2, 3, 4))
        >>> target = USAF1951Target(config)
        >>> image = target.generate()
        >>> print(f"Group 2, Element 4: {target.get_frequency_lp_mm(2, 4):.2f} lp/mm")
    """

    def __init__(self, config: USAF1951Config):
        super().__init__(config)
        self.usaf_config = config

    @staticmethod
    def get_frequency_lp_mm(group: int, element: int) -> float:
        """Calculate spatial frequency for given group/element.

        Args:
            group: Group number (-2 to 9)
            element: Element number (1 to 6)

        Returns:
            Spatial frequency in line pairs per millimeter
        """
        return 2 ** (group + (element - 1) / 6)

    @staticmethod
    def get_bar_width_mm(group: int, element: int) -> float:
        """Calculate bar width for given group/element.

        Args:
            group: Group number
            element: Element number

        Returns:
            Bar width in millimeters
        """
        freq = USAF1951Target.get_frequency_lp_mm(group, element)
        return 1.0 / (2.0 * freq)

    def get_bar_width_m(self, group: int, element: int) -> float:
        """Calculate bar width in meters.

        Args:
            group: Group number
            element: Element number

        Returns:
            Bar width in meters
        """
        return self.get_bar_width_mm(group, element) * 1e-3

    def get_bar_width_pixels(self, group: int, element: int) -> float:
        """Calculate bar width in pixels for current configuration.

        Args:
            group: Group number
            element: Element number

        Returns:
            Bar width in pixels (may be fractional)
        """
        if self.config.pixel_size is None:
            raise ValueError("Cannot compute pixel width without field_size")

        bar_width_m = self.get_bar_width_m(group, element)
        return bar_width_m / self.config.pixel_size

    def generate_element(
        self,
        group: int,
        element: int,
        center: Tuple[float, float],
        orientation: str = "horizontal",
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate single 3-bar element.

        Args:
            group: Group number
            element: Element number (1-6)
            center: (cy, cx) center position in pixels
            orientation: "horizontal" or "vertical"
            mask: Existing mask to add to (in-place modification)

        Returns:
            Boolean mask with 3-bar pattern (or modified input mask)
        """
        size = self.config.size
        device = self.config.device

        # Calculate bar dimensions in pixels
        if self.config.pixel_size is not None:
            bar_width_px = self.get_bar_width_pixels(group, element)
        else:
            # Fallback: compute based on relative scale within image
            # Use active region as reference (1mm total for standard USAF)
            active_size_px = self.config.active_size
            bar_width_mm = self.get_bar_width_mm(group, element)
            # Scale: active region represents ~1mm
            bar_width_px = bar_width_mm * active_size_px

        # Minimum feature size check
        if bar_width_px < 1.0:
            return (
                mask
                if mask is not None
                else torch.zeros((size, size), dtype=torch.bool, device=device)
            )

        # Element is 5 bar widths wide (3 bars + 2 spaces)
        element_size_px = 5 * bar_width_px

        # Create coordinate grids
        cy, cx = center
        y = torch.arange(size, device=device, dtype=torch.float32) - cy
        x = torch.arange(size, device=device, dtype=torch.float32) - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Initialize mask if not provided
        if mask is None:
            mask = torch.zeros((size, size), dtype=torch.bool, device=device)

        # Create 3-bar pattern
        # USAF bars: 3 bars of width bar_width, separated by spaces of bar_width
        # Bar centers at: -2, 0, +2 (in units of bar_width)
        # Bar edges: [-2.5, -1.5], [-0.5, 0.5], [1.5, 2.5] (in units of bar_width)
        bar_edges = [(-2.5, -1.5), (-0.5, 0.5), (1.5, 2.5)]

        if orientation == "horizontal":
            # Bars parallel to x-axis: bars extend in x, separated in y
            within_width = torch.abs(xx) < (element_size_px / 2)
            for y_min, y_max in bar_edges:
                in_bar = (yy >= y_min * bar_width_px) & (yy < y_max * bar_width_px)
                mask |= within_width & in_bar
        else:  # vertical
            # Bars parallel to y-axis: bars extend in y, separated in x
            within_height = torch.abs(yy) < (element_size_px / 2)
            for x_min, x_max in bar_edges:
                in_bar = (xx >= x_min * bar_width_px) & (xx < x_max * bar_width_px)
                mask |= within_height & in_bar

        return mask

    def generate(self) -> torch.Tensor:
        """Generate complete USAF-1951 target with proper margins.

        Layout: Compact square arrangement with H and V triplets side-by-side.
        Each group row contains: [H triplet] [gap] [V triplet]
        Groups are stacked vertically from bottom to top.

        Returns:
            Target image tensor of shape (size, size)
        """
        size = self.config.size
        device = self.config.device
        active_size = self.config.active_size

        # Initialize background (all zeros including margins)
        mask = torch.zeros((size, size), dtype=torch.bool, device=device)

        # Standard layout: arrange groups in pattern within active region
        groups = self.usaf_config.groups
        n_groups = len(groups)

        if n_groups == 0:
            return self._apply_contrast_and_polarity(mask)

        # Helper to get bar width in pixels for a group/element
        def get_bar_width_px(group: int, element: int) -> float:
            if self.config.pixel_size is not None:
                return self.get_bar_width_pixels(group, element)
            else:
                bar_width_mm = self.get_bar_width_mm(group, element)
                return bar_width_mm * active_size

        # Image center
        center_x = size / 2
        center_y = size / 2

        # Calculate total height needed for all groups (side-by-side layout)
        # Each group needs height = element_size (since H and V are side by side)
        group_info: list[tuple[int, float, float]] = []
        total_height = 0.0
        for group in groups:
            bar_width = get_bar_width_px(group, 1)
            element_size = 5 * bar_width
            if element_size >= 2:  # Skip if too small
                group_info.append((group, bar_width, element_size))
                total_height += element_size

        if not group_info:
            return self._apply_contrast_and_polarity(mask)

        # Add inter-group gaps (0.3 * smallest element size)
        smallest_element = min(info[2] for info in group_info)
        inter_group_gap = smallest_element * 0.3
        total_height += inter_group_gap * (len(group_info) - 1)

        # Center the pattern vertically
        start_y = center_y - total_height / 2

        current_y = start_y
        for group, bar_width, element_size in group_info:
            # For side-by-side layout:
            # - Horizontal triplet on the left
            # - Vertical triplet on the right
            # - Small gap between them

            h_v_gap = element_size * 0.2  # Gap between H and V triplets
            total_width = 2 * element_size + h_v_gap

            # Check if fits in active region width
            if total_width > active_size * 0.95:
                # Skip this group if too wide
                continue

            # Y center for this row
            row_center_y = current_y + element_size / 2

            # Horizontal triplet on the left
            h_center_x = center_x - element_size / 2 - h_v_gap / 2
            self.generate_element(group, 1, (row_center_y, h_center_x), "horizontal", mask)

            # Vertical triplet on the right
            v_center_x = center_x + element_size / 2 + h_v_gap / 2
            self.generate_element(group, 1, (row_center_y, v_center_x), "vertical", mask)

            # Move to next row
            current_y += element_size + inter_group_gap

        # Apply contrast and polarity
        return self._apply_contrast_and_polarity(mask)

    @property
    def resolution_elements(self) -> dict:
        """Get resolution information for all elements.

        Returns:
            Dictionary mapping (group, element) to frequency in lp/mm
        """
        elements = {}
        for group in self.usaf_config.groups:
            for element in range(1, self.usaf_config.elements_per_group + 1):
                key = f"G{group}E{element}"
                freq = self.get_frequency_lp_mm(group, element)
                bar_width_mm = self.get_bar_width_mm(group, element)
                elements[key] = {
                    "frequency_lp_mm": freq,
                    "bar_width_mm": bar_width_mm,
                    "bar_width_um": bar_width_mm * 1000,
                }
        return elements

    def get_resolvable_elements(
        self, resolution_limit: float, unit: str = "m"
    ) -> List[Tuple[int, int]]:
        """Get list of elements that can be resolved at given resolution.

        Args:
            resolution_limit: Resolution limit (smallest resolvable feature)
            unit: Unit of resolution_limit ('m', 'um', 'nm', 'mm')

        Returns:
            List of (group, element) tuples that can be resolved
        """
        # Convert to meters
        unit_factors = {"m": 1, "mm": 1e-3, "um": 1e-6, "nm": 1e-9}
        if unit not in unit_factors:
            raise ValueError(f"Unknown unit '{unit}'. Use: {list(unit_factors.keys())}")

        res_m = resolution_limit * unit_factors[unit]

        resolvable = []
        for group in self.usaf_config.groups:
            for element in range(1, self.usaf_config.elements_per_group + 1):
                bar_width = self.get_bar_width_m(group, element)
                if bar_width >= res_m:
                    resolvable.append((group, element))

        return resolvable


@dataclass
class CheckerboardConfig(TargetConfig):
    """Configuration for checkerboard test target.

    Suitable for drone/aerial imaging validation where USAF-1951 scales
    would be impractical.

    Additional attributes:
        square_size: Size of each square in meters (required for physical scaling)
        n_squares: Number of squares per side (auto-calculated if square_size given)
    """

    square_size: Optional[float] = None  # Physical square size in meters
    n_squares: Optional[int] = None  # Number of squares per side

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.field_size is None:
            raise ValueError("CheckerboardConfig requires field_size to be specified")

        if self.square_size is None and self.n_squares is None:
            raise ValueError("Must specify either square_size or n_squares")

        # Calculate the other parameter if only one is given
        active_field = self.field_size * (1 - 2 * self.margin_ratio)

        if self.square_size is not None and self.n_squares is None:
            self.n_squares = int(active_field / self.square_size)
            if self.n_squares < 2:
                raise ValueError(
                    f"square_size={self.square_size}m is too large for "
                    f"active field {active_field}m (need at least 2 squares)"
                )
        elif self.n_squares is not None and self.square_size is None:
            self.square_size = active_field / self.n_squares

    @classmethod
    def from_gsd(
        cls,
        field_size: float,
        gsd: float,
        squares_per_gsd: int = 10,
        resolution: int = 1024,
        margin_ratio: float = 0.25,
        **kwargs: Any,
    ) -> "CheckerboardConfig":
        """Create checkerboard sized relative to Ground Sampling Distance.

        Args:
            field_size: Physical field of view in meters
            gsd: Ground Sampling Distance in meters
            squares_per_gsd: How many GSD units per square (default 10)
            resolution: Number of pixels
            margin_ratio: Margin ratio for diffraction
            **kwargs: Additional TargetConfig parameters

        Returns:
            Configured CheckerboardConfig instance
        """
        square_size = gsd * squares_per_gsd
        return cls(
            size=resolution,
            field_size=field_size,
            margin_ratio=margin_ratio,
            square_size=square_size,
            **kwargs,
        )


class CheckerboardTarget(Target):
    """Checkerboard test target for aerial/drone imaging validation.

    Creates a regular checkerboard pattern useful for:
    - GSD (Ground Sampling Distance) validation
    - MTF (Modulation Transfer Function) measurement
    - Spatial calibration

    Example:
        >>> config = CheckerboardConfig(
        ...     size=512,
        ...     field_size=36.0,  # 36m field of view
        ...     square_size=0.5,  # 50cm squares
        ... )
        >>> target = CheckerboardTarget(config)
        >>> image = target.generate()
    """

    def __init__(self, config: CheckerboardConfig):
        super().__init__(config)
        self.checker_config = config

    def generate(self) -> torch.Tensor:
        """Generate checkerboard pattern with proper margins.

        Returns:
            Checkerboard image tensor of shape (size, size)
        """
        size = self.config.size
        device = self.config.device
        margin = self.config.margin_pixels
        active_size = self.config.active_size

        n_squares = self.checker_config.n_squares
        if n_squares is None:
            raise ValueError("n_squares not set in config")

        # Create coordinate grids for active region
        y = torch.arange(size, device=device, dtype=torch.float32)
        x = torch.arange(size, device=device, dtype=torch.float32)
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        # Shift to active region coordinates
        yy_active = yy - margin
        xx_active = xx - margin

        # Create checkerboard within active region
        pixels_per_square = active_size / n_squares

        # Compute which square each pixel belongs to
        square_y = (yy_active / pixels_per_square).floor().long()
        square_x = (xx_active / pixels_per_square).floor().long()

        # Checkerboard pattern: (i + j) % 2
        checker = ((square_y + square_x) % 2) == 0

        # Mask to active region
        in_active = (
            (yy_active >= 0)
            & (yy_active < active_size)
            & (xx_active >= 0)
            & (xx_active < active_size)
        )

        # Apply pattern only in active region
        mask = checker & in_active

        return self._apply_contrast_and_polarity(mask)

    @property
    def resolution_elements(self) -> dict:
        """Get resolution information for the checkerboard.

        Returns:
            Dictionary with checkerboard parameters
        """
        square_size = self.checker_config.square_size
        n_squares = self.checker_config.n_squares

        # Spatial frequency (cycles per unit distance)
        if square_size is not None:
            freq_cycles_per_m = 1.0 / (2 * square_size)  # Full period is 2 squares
            return {
                "square_size_m": square_size,
                "square_size_cm": square_size * 100 if square_size else None,
                "n_squares": n_squares,
                "spatial_frequency_cycles_per_m": freq_cycles_per_m,
                "fundamental_period_m": 2 * square_size,
            }
        return {"n_squares": n_squares}

    @property
    def gsd_samples_per_square(self) -> Optional[float]:
        """Get number of pixel samples per square.

        Returns:
            Number of pixels per square, or None if physical size not set
        """
        if self.config.pixel_size is None or self.checker_config.square_size is None:
            return None
        return self.checker_config.square_size / self.config.pixel_size


@dataclass
class PointSourceConfig(TargetConfig):
    """Configuration for point source target (for PSF measurement)."""

    n_sources: int = 1
    source_positions: Optional[List[Tuple[float, float]]] = None  # Normalized coords
    source_intensities: Optional[List[float]] = None


class PointSourceTarget(Target):
    """Point source target for PSF measurement and calibration.

    Creates one or more delta-function point sources for measuring
    the optical system's point spread function.
    """

    def __init__(self, config: PointSourceConfig):
        super().__init__(config)
        self.point_config = config

    def generate(self) -> torch.Tensor:
        """Generate point source pattern.

        Returns:
            Image with point source(s)
        """
        size = self.config.size
        device = self.config.device

        # Initialize background
        image = torch.zeros((size, size), device=device)

        # Default: single point at center
        if self.point_config.source_positions is None:
            positions = [(0.0, 0.0)]  # Center in normalized coords
        else:
            positions = self.point_config.source_positions

        if self.point_config.source_intensities is None:
            intensities = [1.0] * len(positions)
        else:
            intensities = self.point_config.source_intensities

        center = size // 2
        active_size = self.config.active_size

        for (y_norm, x_norm), intensity in zip(positions, intensities):
            # Convert normalized coordinates to pixels
            y_px = int(center + y_norm * active_size / 2)
            x_px = int(center + x_norm * active_size / 2)

            # Ensure within bounds
            if 0 <= y_px < size and 0 <= x_px < size:
                image[y_px, x_px] = intensity * self.config.foreground

        return image

    @property
    def resolution_elements(self) -> dict:
        """Get point source information."""
        return {
            "n_sources": self.point_config.n_sources,
            "positions": self.point_config.source_positions,
        }


def create_target(target_type: str, config: Optional[TargetConfig] = None, **kwargs: Any) -> Target:
    """Factory function for creating test targets.

    Args:
        target_type: Type of target ('usaf1951', 'checkerboard', 'point_source')
        config: Target configuration (optional if kwargs provided)
        **kwargs: Config parameters passed to appropriate config class

    Returns:
        Target instance

    Example:
        >>> target = create_target('usaf1951', size=1024, groups=(0, 1, 2))
        >>> image = target.generate()

        >>> target = create_target(
        ...     'checkerboard',
        ...     field_size=36.0,
        ...     square_size=0.5
        ... )
    """
    target_type = target_type.lower()

    if config is None:
        if target_type == "usaf1951":
            config = USAF1951Config(**kwargs)
        elif target_type == "checkerboard":
            config = CheckerboardConfig(**kwargs)
        elif target_type == "point_source":
            config = PointSourceConfig(**kwargs)
        else:
            raise ValueError(
                f"Unknown target type: {target_type}. "
                f"Available: 'usaf1951', 'checkerboard', 'point_source'"
            )

    if target_type == "usaf1951":
        if not isinstance(config, USAF1951Config):
            # Convert base config to USAF config
            base_dict = {
                k: v for k, v in vars(config).items() if k in TargetConfig.__dataclass_fields__
            }
            config = USAF1951Config(**base_dict, **kwargs)
        return USAF1951Target(config)

    elif target_type == "checkerboard":
        if not isinstance(config, CheckerboardConfig):
            base_dict = {
                k: v for k, v in vars(config).items() if k in TargetConfig.__dataclass_fields__
            }
            config = CheckerboardConfig(**base_dict, **kwargs)
        return CheckerboardTarget(config)

    elif target_type == "point_source":
        if not isinstance(config, PointSourceConfig):
            base_dict = {
                k: v for k, v in vars(config).items() if k in TargetConfig.__dataclass_fields__
            }
            config = PointSourceConfig(**base_dict, **kwargs)
        return PointSourceTarget(config)

    else:
        raise ValueError(
            f"Unknown target type: {target_type}. "
            f"Available: 'usaf1951', 'checkerboard', 'point_source'"
        )


def create_usaf_target(
    field_size: float,
    resolution: int = 1024,
    groups: Optional[Tuple[int, ...]] = None,
    margin_ratio: float = 0.25,
    **kwargs: Any,
) -> USAF1951Target:
    """Convenience function to create USAF-1951 target with physical parameters.

    Args:
        field_size: Physical field of view in meters
        resolution: Number of pixels (default 1024)
        groups: USAF groups to include (auto-selected if None)
        margin_ratio: Margin ratio for proper diffraction (default 0.25)
        **kwargs: Additional config parameters

    Returns:
        Configured USAF1951Target instance

    Example:
        >>> from prism.config.constants import um
        >>> target = create_usaf_target(
        ...     field_size=100 * um,  # 100 µm field
        ...     resolution=512,
        ...     groups=(5, 6, 7),
        ... )
        >>> image = target.generate()
    """
    config = USAF1951Config.from_physical_params(
        field_size=field_size,
        resolution=resolution,
        groups=groups,
        margin_ratio=margin_ratio,
        **kwargs,
    )
    return USAF1951Target(config)


def create_checkerboard_target(
    field_size: float,
    square_size: float,
    resolution: int = 1024,
    margin_ratio: float = 0.25,
    **kwargs: Any,
) -> CheckerboardTarget:
    """Convenience function to create checkerboard target with physical parameters.

    Args:
        field_size: Physical field of view in meters
        square_size: Size of each square in meters
        resolution: Number of pixels (default 1024)
        margin_ratio: Margin ratio for proper diffraction (default 0.25)
        **kwargs: Additional config parameters

    Returns:
        Configured CheckerboardTarget instance

    Example:
        >>> target = create_checkerboard_target(
        ...     field_size=50.0,  # 50m field
        ...     square_size=0.5,  # 50cm squares
        ...     resolution=512,
        ... )
        >>> image = target.generate()
    """
    config = CheckerboardConfig(
        size=resolution,
        field_size=field_size,
        square_size=square_size,
        margin_ratio=margin_ratio,
        **kwargs,
    )
    return CheckerboardTarget(config)
