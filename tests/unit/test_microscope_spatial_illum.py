"""Unit tests for Microscope spatial illumination forward model."""

import pytest
import torch

from prism.core.instruments import Microscope, MicroscopeConfig
from prism.core.optics.illumination import IlluminationSourceType


class TestMicroscopeSpatialIllumination:
    """Tests for _forward_spatial_illumination()."""

    @pytest.fixture
    def microscope(self) -> Microscope:
        return Microscope(
            MicroscopeConfig(
                n_pixels=64,
                pixel_size=1e-6,
                numerical_aperture=0.5,
                wavelength=520e-9,
                magnification=40.0,
            )
        )

    @pytest.fixture
    def test_object(self, microscope: Microscope) -> torch.Tensor:
        n = microscope.config.n_pixels
        return torch.ones(n, n, dtype=torch.complex64)

    def test_output_shape(self, microscope: Microscope, test_object: torch.Tensor) -> None:
        """Test output has correct shape."""
        result = microscope._forward_spatial_illumination(
            test_object,
            spatial_center=[0.0, 0.0],
            source_distance=10e-3,
        )
        assert result.shape == test_object.shape

    def test_shifted_source_changes_output(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test that different source positions give different outputs."""
        result_center = microscope._forward_spatial_illumination(
            test_object,
            spatial_center=[0.0, 0.0],
            source_distance=10e-3,
        )
        result_shifted = microscope._forward_spatial_illumination(
            test_object,
            spatial_center=[10e-6, 0.0],
            source_distance=10e-3,
        )

        # Outputs should differ
        diff = (result_center - result_shifted).abs().mean()
        assert diff > 1e-6

    def test_requires_source_distance(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test that source_distance is required."""
        with pytest.raises(TypeError):
            microscope._forward_spatial_illumination(
                test_object,
                spatial_center=[0.0, 0.0],
                # Missing source_distance
            )

    def test_output_is_intensity(self, microscope: Microscope, test_object: torch.Tensor) -> None:
        """Test that output is real-valued intensity (non-negative)."""
        result = microscope._forward_spatial_illumination(
            test_object,
            spatial_center=[0.0, 0.0],
            source_distance=10e-3,
        )
        assert result.dtype in [torch.float32, torch.float64]
        assert (result >= 0).all()

    def test_gaussian_source_type(self, microscope: Microscope, test_object: torch.Tensor) -> None:
        """Test GAUSSIAN source type with illumination_radius."""
        result = microscope._forward_spatial_illumination(
            test_object,
            spatial_center=[0.0, 0.0],
            source_distance=10e-3,
            illumination_source_type=IlluminationSourceType.GAUSSIAN,
            illumination_radius=5e-6,
        )
        assert result.shape == test_object.shape
        assert (result >= 0).all()

    def test_circular_source_type(self, microscope: Microscope, test_object: torch.Tensor) -> None:
        """Test CIRCULAR source type with illumination_radius."""
        result = microscope._forward_spatial_illumination(
            test_object,
            spatial_center=[0.0, 0.0],
            source_distance=10e-3,
            illumination_source_type=IlluminationSourceType.CIRCULAR,
            illumination_radius=5e-6,
        )
        assert result.shape == test_object.shape
        assert (result >= 0).all()

    def test_gaussian_requires_radius(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test that GAUSSIAN source requires illumination_radius."""
        with pytest.raises(ValueError, match="GAUSSIAN source requires"):
            microscope._forward_spatial_illumination(
                test_object,
                spatial_center=[0.0, 0.0],
                source_distance=10e-3,
                illumination_source_type=IlluminationSourceType.GAUSSIAN,
                # Missing illumination_radius
            )

    def test_circular_requires_radius(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test that CIRCULAR source requires illumination_radius."""
        with pytest.raises(ValueError, match="CIRCULAR source requires"):
            microscope._forward_spatial_illumination(
                test_object,
                spatial_center=[0.0, 0.0],
                source_distance=10e-3,
                illumination_source_type=IlluminationSourceType.CIRCULAR,
                # Missing illumination_radius
            )

    def test_add_noise_changes_output(
        self, microscope: Microscope, test_object: torch.Tensor
    ) -> None:
        """Test that add_noise parameter adds noise to output."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        result_noisy1 = microscope._forward_spatial_illumination(
            test_object,
            spatial_center=[0.0, 0.0],
            source_distance=10e-3,
            add_noise=True,
        )

        torch.manual_seed(43)
        result_noisy2 = microscope._forward_spatial_illumination(
            test_object,
            spatial_center=[0.0, 0.0],
            source_distance=10e-3,
            add_noise=True,
        )

        # Different random seeds should give different noisy outputs
        diff = (result_noisy1 - result_noisy2).abs().mean()
        assert diff > 1e-6

    def test_batched_input_2d(self, microscope: Microscope) -> None:
        """Test with 2D input (single image)."""
        n = microscope.config.n_pixels
        field = torch.ones(n, n, dtype=torch.complex64)

        result = microscope._forward_spatial_illumination(
            field,
            spatial_center=[0.0, 0.0],
            source_distance=10e-3,
        )

        assert result.shape == (n, n)
        assert result.ndim == 2

    def test_batched_input_3d(self, microscope: Microscope) -> None:
        """Test with 3D input (single channel, single batch)."""
        n = microscope.config.n_pixels
        field = torch.ones(1, n, n, dtype=torch.complex64)

        result = microscope._forward_spatial_illumination(
            field,
            spatial_center=[0.0, 0.0],
            source_distance=10e-3,
        )

        # Should squeeze back to 2D
        assert result.shape == (n, n)
        assert result.ndim == 2

    def test_batched_input_4d(self, microscope: Microscope) -> None:
        """Test with 4D input (batched with channels).

        Note: The current implementation squeezes 3D inputs (B=1, C=1, H, W),
        so for proper 4D batch processing, the method processes only the
        first element. This test documents the current behavior.
        """
        n = microscope.config.n_pixels
        # Single batch, single channel - will be squeezed
        field = torch.ones(1, 1, n, n, dtype=torch.complex64)

        result = microscope._forward_spatial_illumination(
            field,
            spatial_center=[0.0, 0.0],
            source_distance=10e-3,
        )

        # Current behavior: squeezes 3D input back to 2D
        # If B=1 and C=1, the squeeze removes those dimensions
        assert result.ndim == 2
        assert result.shape == (n, n)
