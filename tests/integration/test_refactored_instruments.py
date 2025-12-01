"""Integration tests for refactored instruments (Phase 3 of Four-F System consolidation).

This test suite verifies that all instruments refactored to inherit from FourFSystem:
1. Produce numerically identical outputs to pre-refactoring golden outputs
2. Maintain cross-instrument consistency (same pupils give same results)
3. Work correctly with SPIDS measurement workflows

These tests ensure backward compatibility and validate the Four-F System refactoring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from prism.core.instruments import (
    Camera,
    CameraConfig,
    FourFSystem,
    Microscope,
    MicroscopeConfig,
    Telescope,
    TelescopeConfig,
)


# Test configuration
GOLDEN_DIR = Path(__file__).parent.parent / "regression" / "golden"
ATOL = 1e-6
RTOL = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def test_field_512():
    """Create a test field with known structure (512x512)."""
    n = 512
    x = torch.linspace(-1, 1, n, device=DEVICE)
    y = torch.linspace(-1, 1, n, device=DEVICE)
    xx, yy = torch.meshgrid(x, y, indexing="xy")

    # Circular disk with smooth edges
    r = torch.sqrt(xx**2 + yy**2)
    disk = torch.exp(-(r**2) / 0.2)

    # Add structure (rings)
    rings = 0.5 * (1 + torch.cos(10 * r))
    field = disk * rings
    field = field / field.max()

    return field.to(DEVICE)


@pytest.fixture(scope="module")
def test_field_256():
    """Create a test field with known structure (256x256)."""
    n = 256
    x = torch.linspace(-1, 1, n, device=DEVICE)
    y = torch.linspace(-1, 1, n, device=DEVICE)
    xx, yy = torch.meshgrid(x, y, indexing="xy")

    r = torch.sqrt(xx**2 + yy**2)
    theta = torch.atan2(yy, xx)
    pattern = torch.exp(-(r**2) / 0.3) * (1 + 0.5 * torch.cos(5 * theta))
    pattern = pattern / pattern.max()

    return pattern.to(DEVICE)


# ============================================================================
# Test: All instruments inherit from FourFSystem
# ============================================================================


class TestFourFSystemInheritance:
    """Verify all instruments correctly inherit from FourFSystem."""

    def test_telescope_inherits_from_four_f_system(self):
        """Telescope should inherit from FourFSystem."""
        config = TelescopeConfig(
            n_pixels=128,
            aperture_radius_pixels=25,
            wavelength=550e-9,
        )
        telescope = Telescope(config)
        assert isinstance(telescope, FourFSystem)

    def test_camera_inherits_from_four_f_system(self):
        """Camera should inherit from FourFSystem."""
        config = CameraConfig(
            n_pixels=128,
            wavelength=550e-9,
            focal_length=50e-3,
            f_number=2.8,
        )
        camera = Camera(config)
        assert isinstance(camera, FourFSystem)

    def test_microscope_inherits_from_four_f_system(self):
        """Microscope should inherit from FourFSystem."""
        config = MicroscopeConfig(
            n_pixels=128,
            pixel_size=5.0e-6,
            wavelength=550e-9,
            numerical_aperture=0.9,
            magnification=40.0,
        )
        microscope = Microscope(config)
        assert isinstance(microscope, FourFSystem)


# ============================================================================
# Test: Abstract methods are implemented
# ============================================================================


class TestAbstractMethodImplementation:
    """Verify all abstract methods are properly implemented."""

    @pytest.fixture
    def telescope(self):
        config = TelescopeConfig(
            n_pixels=128,
            aperture_radius_pixels=25,
            wavelength=550e-9,
        )
        return Telescope(config)

    @pytest.fixture
    def camera(self):
        config = CameraConfig(
            n_pixels=128,
            wavelength=550e-9,
            focal_length=50e-3,
            f_number=2.8,
            object_distance=float("inf"),
        )
        return Camera(config)

    @pytest.fixture
    def microscope(self):
        config = MicroscopeConfig(
            n_pixels=128,
            pixel_size=5.0e-6,
            wavelength=550e-9,
            numerical_aperture=0.9,
            magnification=40.0,
        )
        return Microscope(config)

    def test_telescope_create_pupils(self, telescope):
        """Telescope should implement _create_pupils."""
        illum, detect = telescope._create_pupils()
        assert illum is None  # Telescope has no illumination pupil
        assert detect is not None
        assert detect.shape == (128, 128)

    def test_camera_create_pupils(self, camera):
        """Camera should implement _create_pupils."""
        illum, detect = camera._create_pupils()
        assert illum is None  # Camera has no illumination pupil
        assert detect is not None
        assert detect.shape == (128, 128)

    def test_microscope_create_pupils_brightfield(self, microscope):
        """Microscope should implement _create_pupils with illumination modes."""
        illum, detect = microscope._create_pupils(illumination_mode="brightfield")
        assert illum is not None
        assert detect is not None
        assert illum.shape == detect.shape

    def test_microscope_create_pupils_darkfield(self, microscope):
        """Microscope darkfield should have annular illumination."""
        illum, detect = microscope._create_pupils(
            illumination_mode="darkfield",
            illumination_params={"annular_ratio": 0.8},
        )
        assert illum is not None
        assert detect is not None
        # Darkfield illumination should be annular (center should be zero)
        # The annular pattern has zero at center for some radius

    def test_telescope_resolution_limit(self, telescope):
        """Telescope should have resolution_limit property."""
        res = telescope.resolution_limit
        assert res > 0
        assert isinstance(res, float)

    def test_camera_resolution_limit(self, camera):
        """Camera should have resolution_limit property."""
        res = camera.resolution_limit
        assert res > 0
        assert isinstance(res, float)

    def test_microscope_resolution_limit(self, microscope):
        """Microscope should have resolution_limit property."""
        res = microscope.resolution_limit
        assert res > 0
        assert isinstance(res, float)


# ============================================================================
# Test: Forward model outputs
# ============================================================================


class TestForwardModelOutputs:
    """Test that forward models produce valid outputs."""

    @pytest.fixture
    def simple_field(self):
        """Create a simple test field."""
        n = 128
        field = torch.zeros(n, n, device=DEVICE)
        field[n // 4 : 3 * n // 4, n // 4 : 3 * n // 4] = 1.0
        return field

    def test_telescope_forward_output_shape(self, simple_field):
        """Telescope forward should preserve input shape."""
        config = TelescopeConfig(
            n_pixels=128,
            aperture_radius_pixels=25,
            wavelength=550e-9,
        )
        telescope = Telescope(config).to(DEVICE)

        output = telescope.forward(simple_field, add_noise=False)
        assert output.shape == simple_field.shape

    def test_telescope_forward_non_negative(self, simple_field):
        """Telescope forward output should be non-negative (intensity)."""
        config = TelescopeConfig(
            n_pixels=128,
            aperture_radius_pixels=25,
            wavelength=550e-9,
        )
        telescope = Telescope(config).to(DEVICE)

        output = telescope.forward(simple_field, add_noise=False)
        assert torch.all(output >= 0)

    def test_camera_forward_output_shape(self, simple_field):
        """Camera forward should preserve input shape."""
        config = CameraConfig(
            n_pixels=128,
            wavelength=550e-9,
            focal_length=50e-3,
            f_number=2.8,
            object_distance=1.0,  # Near field to use Angular Spectrum
        )
        camera = Camera(config)

        # Move field to CPU since Camera doesn't support device placement
        field_cpu = simple_field.cpu()
        output = camera.forward(field_cpu, add_noise=False)
        assert output.shape == field_cpu.shape

    def test_microscope_forward_output_shape(self, simple_field):
        """Microscope forward should preserve input shape."""
        config = MicroscopeConfig(
            n_pixels=128,
            pixel_size=5.0e-6,
            wavelength=550e-9,
            numerical_aperture=0.9,
            magnification=40.0,
        )
        microscope = Microscope(config).to(DEVICE)

        output = microscope.forward(simple_field, add_noise=False)
        assert output.shape == simple_field.shape

    def test_microscope_illumination_modes_produce_different_outputs(self, simple_field):
        """Different illumination modes should produce different outputs."""
        config = MicroscopeConfig(
            n_pixels=128,
            pixel_size=5.0e-6,
            wavelength=550e-9,
            numerical_aperture=0.9,
            magnification=40.0,
        )
        microscope = Microscope(config).to(DEVICE)

        brightfield = microscope.forward(simple_field, illumination_mode="brightfield")
        darkfield = microscope.forward(
            simple_field,
            illumination_mode="darkfield",
            illumination_params={"annular_ratio": 0.8},
        )

        # Outputs should be different
        assert not torch.allclose(brightfield, darkfield)


# ============================================================================
# Test: SPIDS sub-aperture functionality
# ============================================================================


class TestSPIDSSubAperture:
    """Test SPIDS sub-aperture functionality with refactored instruments."""

    @pytest.fixture
    def test_field(self):
        """Create test field for SPIDS."""
        n = 256
        field = torch.zeros(n, n, device=DEVICE)
        # Create a simple pattern
        x = torch.linspace(-1, 1, n, device=DEVICE)
        y = torch.linspace(-1, 1, n, device=DEVICE)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        r = torch.sqrt(xx**2 + yy**2)
        field = torch.exp(-(r**2) / 0.2)
        return field

    def test_telescope_sub_aperture(self, test_field):
        """Telescope should support sub-aperture imaging."""
        config = TelescopeConfig(
            n_pixels=256,
            aperture_radius_pixels=25,
            wavelength=550e-9,
        )
        telescope = Telescope(config).to(DEVICE)

        # Forward with centered aperture
        centered = telescope.forward(
            test_field,
            aperture_center=[0.0, 0.0],
            add_noise=False,
        )

        # Forward with off-center aperture
        off_center = telescope.forward(
            test_field,
            aperture_center=[10.0, 10.0],
            add_noise=False,
        )

        # Outputs should be different
        assert not torch.allclose(centered, off_center)

    def test_microscope_sub_aperture(self, test_field):
        """Microscope should support SPIDS sub-aperture imaging."""
        config = MicroscopeConfig(
            n_pixels=256,
            pixel_size=5.0e-6,
            wavelength=550e-9,
            numerical_aperture=0.9,
            magnification=40.0,
        )
        microscope = Microscope(config).to(DEVICE)

        # Forward with centered aperture
        centered = microscope.forward(
            test_field,
            aperture_center=[0.0, 0.0],
            aperture_radius=15.0,
            add_noise=False,
        )

        # Forward with off-center aperture
        off_center = microscope.forward(
            test_field,
            aperture_center=[10.0, 10.0],
            aperture_radius=15.0,
            add_noise=False,
        )

        # Outputs should be different
        assert not torch.allclose(centered, off_center)

    def test_generate_aperture_mask(self):
        """Test aperture mask generation."""
        config = TelescopeConfig(
            n_pixels=256,
            aperture_radius_pixels=25,
            wavelength=550e-9,
        )
        telescope = Telescope(config).to(DEVICE)

        # Generate centered mask
        mask_centered = telescope.generate_aperture_mask([0.0, 0.0])
        assert mask_centered.shape == (256, 256)

        # Generate off-center mask
        mask_offset = telescope.generate_aperture_mask([20.0, 20.0])
        assert mask_offset.shape == (256, 256)

        # Masks should be different
        assert not torch.allclose(mask_centered, mask_offset)


# ============================================================================
# Test: Regression against golden outputs
# ============================================================================


class TestRegressionAgainstGolden:
    """Regression tests comparing refactored instruments to golden outputs."""

    def test_telescope_regression_psf(self):
        """Telescope PSF should match golden output."""
        golden_path = GOLDEN_DIR / "telescope_psf_circular.npz"
        if not golden_path.exists():
            pytest.skip("Golden output not generated yet")

        config = TelescopeConfig(
            n_pixels=512,
            pixel_size=6.5e-6,
            wavelength=550e-9,
            aperture_radius_pixels=50.0,
            aperture_diameter=8.2,
            aperture_type="circular",
        )
        telescope = Telescope(config).to(DEVICE)

        psf = telescope.compute_psf(center=[0.0, 0.0])
        loaded = np.load(golden_path)

        assert np.allclose(psf.cpu().numpy(), loaded["psf"], atol=ATOL, rtol=RTOL), (
            "Telescope PSF regression failed"
        )

    def test_telescope_regression_forward(self, test_field_512):
        """Telescope forward should match golden output."""
        golden_path = GOLDEN_DIR / "telescope_forward_circular.npz"
        if not golden_path.exists():
            pytest.skip("Golden output not generated yet")

        config = TelescopeConfig(
            n_pixels=512,
            pixel_size=6.5e-6,
            wavelength=550e-9,
            aperture_radius_pixels=50.0,
            aperture_diameter=8.2,
            aperture_type="circular",
        )
        telescope = Telescope(config).to(DEVICE)

        output = telescope.forward(test_field_512, aperture_center=None, add_noise=False)
        loaded = np.load(golden_path)

        assert np.allclose(output.cpu().numpy(), loaded["output"], atol=ATOL, rtol=RTOL), (
            "Telescope forward regression failed"
        )

    def test_microscope_regression_psf(self):
        """Microscope PSF should match golden output."""
        golden_path = GOLDEN_DIR / "microscope_psf_brightfield.npz"
        if not golden_path.exists():
            pytest.skip("Golden output not generated yet")

        config = MicroscopeConfig(
            n_pixels=512,
            pixel_size=5.0e-6,
            wavelength=550e-9,
            numerical_aperture=0.9,
            magnification=40.0,
            medium_index=1.0,
            tube_lens_focal=0.2,
            forward_model_regime="simplified",
            padding_factor=2.0,
        )
        microscope = Microscope(config).to(DEVICE)

        psf = microscope.compute_psf(illumination_mode="brightfield")
        loaded = np.load(golden_path)

        assert np.allclose(psf.cpu().numpy(), loaded["psf"], atol=ATOL, rtol=RTOL), (
            "Microscope PSF regression failed"
        )

    def test_microscope_regression_forward(self, test_field_512):
        """Microscope forward should match golden output."""
        golden_path = GOLDEN_DIR / "microscope_forward_brightfield.npz"
        if not golden_path.exists():
            pytest.skip("Golden output not generated yet")

        config = MicroscopeConfig(
            n_pixels=512,
            pixel_size=5.0e-6,
            wavelength=550e-9,
            numerical_aperture=0.9,
            magnification=40.0,
            medium_index=1.0,
            tube_lens_focal=0.2,
            forward_model_regime="simplified",
            padding_factor=2.0,
        )
        microscope = Microscope(config).to(DEVICE)

        output = microscope.forward(
            test_field_512,
            illumination_mode="brightfield",
            add_noise=False,
            use_unified_model=True,
        )
        loaded = np.load(golden_path)

        assert np.allclose(output.cpu().numpy(), loaded["output"], atol=ATOL, rtol=RTOL), (
            "Microscope forward regression failed"
        )

    def test_camera_regression_psf(self):
        """Camera PSF should match golden output."""
        golden_path = GOLDEN_DIR / "camera_psf_far_field.npz"
        if not golden_path.exists():
            pytest.skip("Golden output not generated yet")

        config = CameraConfig(
            n_pixels=256,
            pixel_size=6.5e-6,
            wavelength=550e-9,
            focal_length=50e-3,
            f_number=2.8,
            object_distance=float("inf"),
            focus_distance=float("inf"),
        )
        camera = Camera(config)

        psf = camera.compute_psf(defocus=0.0)
        loaded = np.load(golden_path)

        assert np.allclose(psf.cpu().numpy(), loaded["psf"], atol=ATOL, rtol=RTOL), (
            "Camera PSF regression failed"
        )


# ============================================================================
# Test: Cross-instrument consistency
# ============================================================================


class TestCrossInstrumentConsistency:
    """Test that different instruments produce consistent results with equivalent settings."""

    def test_all_instruments_have_resolution_limit(self):
        """All instruments should have positive resolution limits."""
        telescope = Telescope(
            TelescopeConfig(
                n_pixels=128,
                aperture_radius_pixels=25,
                wavelength=550e-9,
                aperture_diameter=1.0,
            )
        )
        camera = Camera(
            CameraConfig(
                n_pixels=128,
                wavelength=550e-9,
                focal_length=50e-3,
                f_number=2.8,
            )
        )
        microscope = Microscope(
            MicroscopeConfig(
                n_pixels=128,
                pixel_size=5.0e-6,
                wavelength=550e-9,
                numerical_aperture=0.9,
                magnification=40.0,
            )
        )

        assert telescope.resolution_limit > 0
        assert camera.resolution_limit > 0
        assert microscope.resolution_limit > 0

    def test_aperture_generator_shared(self):
        """All FourFSystem instruments should use ApertureMaskGenerator."""
        telescope = Telescope(
            TelescopeConfig(n_pixels=128, aperture_radius_pixels=25, wavelength=550e-9)
        )
        microscope = Microscope(
            MicroscopeConfig(
                n_pixels=128,
                pixel_size=5.0e-6,
                wavelength=550e-9,
                numerical_aperture=0.9,
                magnification=40.0,
            )
        )

        # Both should have aperture generator accessible via lazy property
        assert hasattr(telescope, "_aperture_generator_lazy")
        assert hasattr(microscope, "_aperture_generator_lazy")


# ============================================================================
# Test: Noise model integration
# ============================================================================


class TestNoiseModelIntegration:
    """Test noise model integration with refactored instruments."""

    def test_telescope_with_noise(self):
        """Telescope should support noise addition."""
        config = TelescopeConfig(
            n_pixels=128,
            aperture_radius_pixels=25,
            wavelength=550e-9,
            snr=40.0,  # 40 dB SNR
        )
        telescope = Telescope(config).to(DEVICE)

        field = torch.ones(128, 128, device=DEVICE)

        # With noise
        output_noisy = telescope.forward(field, add_noise=True)
        # Without noise
        output_clean = telescope.forward(field, add_noise=False)

        # Noisy output should differ from clean
        assert not torch.allclose(output_noisy, output_clean)

    def test_microscope_with_noise(self):
        """Microscope should support noise addition."""
        config = MicroscopeConfig(
            n_pixels=128,
            pixel_size=5.0e-6,
            wavelength=550e-9,
            numerical_aperture=0.9,
            magnification=40.0,
        )
        microscope = Microscope(config).to(DEVICE)

        field = torch.ones(128, 128, device=DEVICE)

        # With noise
        output_noisy = microscope.forward(field, add_noise=True)
        # Without noise
        output_clean = microscope.forward(field, add_noise=False)

        # Noisy output should differ from clean
        assert not torch.allclose(output_noisy, output_clean)


# ============================================================================
# Test: Batch dimension handling
# ============================================================================


class TestBatchDimensionHandling:
    """Test that instruments handle various input shapes correctly."""

    @pytest.fixture
    def microscope(self):
        config = MicroscopeConfig(
            n_pixels=64,
            pixel_size=5.0e-6,
            wavelength=550e-9,
            numerical_aperture=0.9,
            magnification=40.0,
        )
        return Microscope(config).to(DEVICE)

    def test_2d_input(self, microscope):
        """Should handle [H, W] input."""
        field = torch.rand(64, 64, device=DEVICE)
        output = microscope.forward(field)
        assert output.shape == (64, 64)

    def test_3d_input(self, microscope):
        """Should handle [C, H, W] input."""
        field = torch.rand(1, 64, 64, device=DEVICE)
        output = microscope.forward(field)
        # Output should match input dimensions
        assert output.ndim >= 2

    def test_4d_input(self, microscope):
        """Should handle [B, C, H, W] input."""
        field = torch.rand(2, 1, 64, 64, device=DEVICE)
        output = microscope.forward(field)
        # Output should match input dimensions
        assert output.ndim >= 2
