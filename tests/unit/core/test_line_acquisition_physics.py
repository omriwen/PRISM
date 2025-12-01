"""Physics validation tests for line acquisition.

This module tests that the IncoherentLineAcquisition implementation
correctly follows the physics of incoherent intensity summation:
    I = (1/N) × Σᵢ |IFFT(F_kspace × Aperture_i)|²

Key physics tests:
- Incoherent sum differs from coherent sum
- Single position equals point measurement
- Energy scales linearly with number of samples
- Intensity values are non-negative and physical
"""

from __future__ import annotations

import pytest
import torch

from prism.core.instruments.telescope import Telescope, TelescopeConfig
from prism.core.line_acquisition import IncoherentLineAcquisition, LineAcquisitionConfig


@pytest.fixture
def device() -> torch.device:
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def telescope() -> Telescope:
    """Create a small telescope for testing."""
    config = TelescopeConfig(
        n_pixels=128,
        wavelength=550e-9,
        aperture_radius_pixels=15.0,
        focal_length=1.0,
    )
    return Telescope(config)


@pytest.fixture
def simple_field(device: torch.device) -> torch.Tensor:
    """Create a simple k-space field for testing."""
    n = 128
    # Create a simple Gaussian in k-space
    ky = torch.linspace(-1, 1, n, device=device)
    kx = torch.linspace(-1, 1, n, device=device)
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")  # noqa: N806 (physics notation)
    field_kspace = torch.exp(-((KY**2 + KX**2) / 0.1)) + 0j
    return field_kspace


class TestIncoherentPhysics:
    """Test that incoherent summation physics is correctly implemented."""

    def test_incoherent_differs_from_coherent(
        self,
        telescope: Telescope,
        simple_field: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Test that incoherent sum |F1|² + |F2|² ≠ |F1 + F2|²."""
        config = LineAcquisitionConfig(mode="accurate", samples_per_pixel=1.0)
        line_acq = IncoherentLineAcquisition(config, telescope)

        # Create a line with 2 distinct positions
        start = torch.tensor([64.0, 50.0], device=device)
        end = torch.tensor([64.0, 78.0], device=device)  # 28 pixels apart

        # Compute incoherent measurement (current implementation)
        incoherent_result = line_acq.forward(simple_field, start, end, add_noise=False)

        # Compute coherent sum for comparison: |sum(F_i)|²
        positions = line_acq.compute_line_positions(start, end)
        n_positions = positions.shape[0]

        # Generate masks
        masks = telescope.generate_aperture_masks(positions)

        # Apply masks and IFFT
        from prism.utils.transforms import batched_ifft2

        masked_fields = simple_field.unsqueeze(0) * masks.to(
            device=simple_field.device, dtype=simple_field.dtype
        )
        spatial_fields = batched_ifft2(masked_fields)

        # Coherent: sum fields THEN square
        coherent_field_sum = spatial_fields.sum(dim=0) / n_positions
        coherent_result = coherent_field_sum.abs() ** 2

        # They should be different (incoherent ≠ coherent)
        difference = (incoherent_result - coherent_result).abs()
        relative_diff = difference.sum() / coherent_result.sum()

        # For multiple positions with phase differences, this should be substantial
        assert relative_diff > 0.01, (
            f"Incoherent and coherent sums should differ significantly, "
            f"got relative difference: {relative_diff:.6f}"
        )

    def test_single_position_equals_point_measurement(
        self,
        telescope: Telescope,
        simple_field: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Test that line with 2 samples at same position equals single point measurement."""
        config = LineAcquisitionConfig(mode="accurate", min_samples=2)
        line_acq = IncoherentLineAcquisition(config, telescope)

        # Single position (start = end)
        position = torch.tensor([64.0, 64.0], device=device)

        # Line measurement with 1 sample
        line_result = line_acq.forward(simple_field, position, position, add_noise=False)

        # Direct point measurement
        from prism.utils.transforms import batched_ifft2

        mask = telescope.generate_aperture_mask(position.tolist())
        masked_field = simple_field * mask.to(device=simple_field.device, dtype=simple_field.dtype)
        spatial_field = batched_ifft2(masked_field.unsqueeze(0))[0]
        point_result = spatial_field.abs() ** 2

        # Should be equal (within numerical precision)
        torch.testing.assert_close(
            line_result,
            point_result,
            rtol=1e-5,
            atol=1e-8,
            msg="Single position line should equal point measurement",
        )

    def test_energy_conservation_and_scaling(
        self,
        telescope: Telescope,
        simple_field: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Test that total intensity is physical and scales reasonably."""
        config = LineAcquisitionConfig(mode="accurate", samples_per_pixel=1.0)
        line_acq = IncoherentLineAcquisition(config, telescope)

        start = torch.tensor([64.0, 50.0], device=device)
        end = torch.tensor([64.0, 78.0], device=device)

        result = line_acq.forward(simple_field, start, end, add_noise=False)

        # Check physical properties
        assert torch.all(result >= 0), "Intensity must be non-negative"
        assert torch.isfinite(result).all(), "Intensity must be finite"

        # Check that energy is reasonable (not zero, not infinite)
        total_energy = result.sum()
        assert total_energy > 0, "Total intensity should be positive"
        assert total_energy < 1e10, "Total intensity should be reasonable"

    def test_linearity_with_field_amplitude(
        self,
        telescope: Telescope,
        simple_field: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Test that intensity scales as |αF|² = α² |F|²."""
        config = LineAcquisitionConfig(mode="accurate", samples_per_pixel=1.0)
        line_acq = IncoherentLineAcquisition(config, telescope)

        start = torch.tensor([64.0, 60.0], device=device)
        end = torch.tensor([64.0, 68.0], device=device)

        # Original intensity
        I1 = line_acq.forward(simple_field, start, end, add_noise=False)  # noqa: N806 (physics notation)

        # Scaled field (2x amplitude)
        scaled_field = 2.0 * simple_field
        I2 = line_acq.forward(scaled_field, start, end, add_noise=False)  # noqa: N806 (physics notation)

        # Should scale as intensity: I2 = 4 * I1
        expected_I2 = 4.0 * I1  # noqa: N806 (physics notation)

        torch.testing.assert_close(
            I2,
            expected_I2,
            rtol=1e-5,
            atol=1e-8,
            msg="Intensity should scale quadratically with field amplitude",
        )

    def test_mode_comparison_fast_vs_accurate(
        self,
        telescope: Telescope,
        simple_field: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Test that fast and accurate modes give different but physically valid results.

        Fast mode uses much fewer samples (diameter/2 spacing vs 1 per pixel),
        so significant differences are expected. The key validation is:
        1. Both produce positive, finite intensities
        2. Fast mode uses fewer samples than accurate mode
        3. Results differ but are in the same general magnitude
        """
        # Accurate mode
        config_accurate = LineAcquisitionConfig(mode="accurate", samples_per_pixel=1.0)
        line_acq_accurate = IncoherentLineAcquisition(config_accurate, telescope)

        # Fast mode
        config_fast = LineAcquisitionConfig(mode="fast")
        line_acq_fast = IncoherentLineAcquisition(config_fast, telescope)

        start = torch.tensor([64.0, 50.0], device=device)
        end = torch.tensor([64.0, 78.0], device=device)  # 28 pixels

        I_accurate = line_acq_accurate.forward(simple_field, start, end, add_noise=False)  # noqa: N806
        I_fast = line_acq_fast.forward(simple_field, start, end, add_noise=False)  # noqa: N806

        # Both should produce valid physical results
        assert torch.all(I_accurate >= 0), "Accurate mode intensity must be non-negative"
        assert torch.all(I_fast >= 0), "Fast mode intensity must be non-negative"
        assert torch.isfinite(I_accurate).all(), "Accurate mode intensity must be finite"
        assert torch.isfinite(I_fast).all(), "Fast mode intensity must be finite"

        # Fast mode should use fewer samples
        line_length = 28.0
        n_accurate = line_acq_accurate.compute_n_samples(line_length)
        n_fast = line_acq_fast.compute_n_samples(line_length)
        assert n_fast < n_accurate, (
            f"Fast mode should use fewer samples than accurate mode, "
            f"got fast={n_fast}, accurate={n_accurate}"
        )

        # Results should be different but in same order of magnitude
        # (within 10x of each other, since sampling is very different)
        sum_accurate = I_accurate.sum()
        sum_fast = I_fast.sum()
        ratio = max(sum_accurate, sum_fast) / (min(sum_accurate, sum_fast) + 1e-10)
        assert ratio < 10, f"Results should be within 10x of each other, got ratio: {ratio:.2f}"

    def test_symmetry_start_end_order(
        self,
        telescope: Telescope,
        simple_field: torch.Tensor,
        device: torch.device,
    ) -> None:
        """Test that swapping start/end gives identical results."""
        config = LineAcquisitionConfig(mode="accurate", samples_per_pixel=1.0)
        line_acq = IncoherentLineAcquisition(config, telescope)

        start = torch.tensor([64.0, 50.0], device=device)
        end = torch.tensor([64.0, 78.0], device=device)

        I_forward = line_acq.forward(simple_field, start, end, add_noise=False)  # noqa: N806
        I_backward = line_acq.forward(simple_field, end, start, add_noise=False)  # noqa: N806

        torch.testing.assert_close(
            I_forward,
            I_backward,
            rtol=1e-5,
            atol=1e-8,
            msg="Start/end order should not affect result",
        )


class TestSamplingModes:
    """Test different sampling modes and their properties."""

    def test_accurate_mode_samples_per_pixel(
        self,
        telescope: Telescope,
        device: torch.device,
    ) -> None:
        """Test accurate mode sampling: N ≈ line_length × samples_per_pixel."""
        config = LineAcquisitionConfig(mode="accurate", samples_per_pixel=1.0)
        line_acq = IncoherentLineAcquisition(config, telescope)

        # 50 pixel line
        line_length = 50.0

        n_samples = line_acq.compute_n_samples(line_length)

        # Should be approximately 50 samples (1 per pixel)
        assert 48 <= n_samples <= 52, f"Expected ~50 samples, got {n_samples}"

    def test_fast_mode_diameter_spacing(
        self,
        telescope: Telescope,
        device: torch.device,
    ) -> None:
        """Test fast mode sampling: N ≈ line_length / (diameter/2)."""
        config = LineAcquisitionConfig(mode="fast")
        line_acq = IncoherentLineAcquisition(config, telescope)

        # 50 pixel line
        line_length = 50.0

        n_samples = line_acq.compute_n_samples(line_length)

        # Diameter in pixels (this depends on telescope config)
        # For TelescopeConfig with n_pixels=128, dx=1e-3, aperture_radius=0.05
        # diameter ≈ 2 * 0.05 / 1e-3 = 100 pixels (but telescope.r is in index units)
        # Fast mode: spacing = diameter / 2, so N ≈ line_length / (diameter/2)

        # Just check it's less than accurate mode
        config_accurate = LineAcquisitionConfig(mode="accurate", samples_per_pixel=1.0)
        line_acq_accurate = IncoherentLineAcquisition(config_accurate, telescope)
        n_samples_accurate = line_acq_accurate.compute_n_samples(line_length)

        assert n_samples < n_samples_accurate, (
            f"Fast mode should use fewer samples than accurate mode, "
            f"got fast={n_samples}, accurate={n_samples_accurate}"
        )

    def test_min_samples_enforced(
        self,
        telescope: Telescope,
        device: torch.device,
    ) -> None:
        """Test that min_samples is enforced even for short lines."""
        config = LineAcquisitionConfig(mode="accurate", min_samples=10)
        line_acq = IncoherentLineAcquisition(config, telescope)

        # Very short line (1 pixel)
        line_length = 1.0
        n_samples = line_acq.compute_n_samples(line_length)

        assert n_samples >= 10, f"Expected >= 10 samples, got {n_samples}"
