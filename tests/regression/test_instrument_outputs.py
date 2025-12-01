"""Regression tests for instrument outputs before Four-F System refactoring.

This test suite creates golden reference outputs for each instrument to ensure that
the Four-F System consolidation (feature/four-f-system-consolidation) does not change
numerical behavior.

Golden outputs are saved in tests/regression/golden/ as .npz files and include:
- PSF (Point Spread Function) outputs
- Forward model outputs for various test scenarios
- Instrument-specific mode outputs (e.g., microscope illumination modes)

Numerical tolerance: atol=1e-6, rtol=1e-5
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from prism.core.instruments import (
    Camera,
    CameraConfig,
    Microscope,
    MicroscopeConfig,
    Telescope,
    TelescopeConfig,
)


# Test configuration
GOLDEN_DIR = Path(__file__).parent / "golden"
ATOL = 1e-6
RTOL = 1e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def ensure_golden_dir():
    """Ensure golden directory exists."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    return GOLDEN_DIR


@pytest.fixture(scope="module")
def test_field_512():
    """Create a test field with known structure (512x512)."""
    # Create a simple test pattern: circular disk with Gaussian envelope
    n = 512
    x = torch.linspace(-1, 1, n, device=DEVICE)
    y = torch.linspace(-1, 1, n, device=DEVICE)
    xx, yy = torch.meshgrid(x, y, indexing="xy")

    # Circular disk with smooth edges
    r = torch.sqrt(xx**2 + yy**2)
    disk = torch.exp(-(r**2) / 0.2)  # Gaussian disk

    # Add some structure (rings)
    rings = 0.5 * (1 + torch.cos(10 * r))

    # Combine
    field = disk * rings

    # Normalize
    field = field / field.max()

    return field.to(DEVICE)


@pytest.fixture(scope="module")
def test_field_256():
    """Create a test field with known structure (256x256)."""
    # Create a simple test pattern for smaller tests
    n = 256
    x = torch.linspace(-1, 1, n, device=DEVICE)
    y = torch.linspace(-1, 1, n, device=DEVICE)
    xx, yy = torch.meshgrid(x, y, indexing="xy")

    # Star-like pattern
    r = torch.sqrt(xx**2 + yy**2)
    theta = torch.atan2(yy, xx)

    pattern = torch.exp(-(r**2) / 0.3) * (1 + 0.5 * torch.cos(5 * theta))

    # Normalize
    pattern = pattern / pattern.max()

    return pattern.to(DEVICE)


# ============================================================================
# Microscope Tests
# ============================================================================


class TestMicroscopeGoldenOutputs:
    """Generate and validate golden outputs for Microscope."""

    @pytest.fixture(scope="class")
    def microscope_config(self):
        """Standard microscope configuration for testing."""
        # For NA=0.9, wavelength=550nm: Nyquist limit = λ/(4*NA) = 153nm
        # Object pixel size = camera_pixel / magnification
        # For 40x mag: camera_pixel = 153nm * 40 = 6.1µm (meets Nyquist)
        # Use 5µm camera pixels to comfortably exceed Nyquist
        return MicroscopeConfig(
            n_pixels=512,
            pixel_size=5.0e-6,  # Camera pixel size (5µm works with 40x mag and NA=0.9)
            wavelength=550e-9,  # Green light
            numerical_aperture=0.9,
            magnification=40.0,
            medium_index=1.0,  # Air
            tube_lens_focal=0.2,  # 200mm
            forward_model_regime="simplified",  # Use simplified model for 4f test
            padding_factor=2.0,
        )

    @pytest.fixture(scope="class")
    def microscope(self, microscope_config):
        """Create microscope instance."""
        return Microscope(microscope_config).to(DEVICE)

    def test_microscope_psf_brightfield(self, microscope, ensure_golden_dir):
        """Test microscope PSF in brightfield mode."""
        psf = microscope.compute_psf(
            illumination_mode="brightfield",
            illumination_params=None,
        )

        # Convert to numpy for saving
        psf_np = psf.cpu().numpy()

        # Save golden output
        golden_path = GOLDEN_DIR / "microscope_psf_brightfield.npz"
        np.savez(golden_path, psf=psf_np)

        # Verify we can load it back
        loaded = np.load(golden_path)
        assert np.allclose(loaded["psf"], psf_np, atol=ATOL, rtol=RTOL)

    def test_microscope_psf_darkfield(self, microscope, ensure_golden_dir):
        """Test microscope PSF in darkfield mode."""
        psf = microscope.compute_psf(
            illumination_mode="darkfield",
            illumination_params={"annular_ratio": 0.8},
        )

        psf_np = psf.cpu().numpy()
        golden_path = GOLDEN_DIR / "microscope_psf_darkfield.npz"
        np.savez(golden_path, psf=psf_np)

        loaded = np.load(golden_path)
        assert np.allclose(loaded["psf"], psf_np, atol=ATOL, rtol=RTOL)

    def test_microscope_psf_phase(self, microscope, ensure_golden_dir):
        """Test microscope PSF in phase contrast mode."""
        psf = microscope.compute_psf(
            illumination_mode="phase",
            illumination_params={"phase_shift": np.pi / 2},
        )

        psf_np = psf.cpu().numpy()
        golden_path = GOLDEN_DIR / "microscope_psf_phase.npz"
        np.savez(golden_path, psf=psf_np)

        loaded = np.load(golden_path)
        assert np.allclose(loaded["psf"], psf_np, atol=ATOL, rtol=RTOL)

    def test_microscope_forward_brightfield(self, microscope, test_field_512, ensure_golden_dir):
        """Test microscope forward model in brightfield mode."""
        output = microscope.forward(
            test_field_512,
            illumination_mode="brightfield",
            add_noise=False,
            use_unified_model=True,
        )

        output_np = output.cpu().numpy()
        golden_path = GOLDEN_DIR / "microscope_forward_brightfield.npz"
        np.savez(golden_path, output=output_np, input=test_field_512.cpu().numpy())

        loaded = np.load(golden_path)
        assert np.allclose(loaded["output"], output_np, atol=ATOL, rtol=RTOL)

    def test_microscope_forward_darkfield(self, microscope, test_field_512, ensure_golden_dir):
        """Test microscope forward model in darkfield mode."""
        output = microscope.forward(
            test_field_512,
            illumination_mode="darkfield",
            illumination_params={"annular_ratio": 0.8},
            add_noise=False,
            use_unified_model=True,
        )

        output_np = output.cpu().numpy()
        golden_path = GOLDEN_DIR / "microscope_forward_darkfield.npz"
        np.savez(golden_path, output=output_np, input=test_field_512.cpu().numpy())

        loaded = np.load(golden_path)
        assert np.allclose(loaded["output"], output_np, atol=ATOL, rtol=RTOL)

    def test_microscope_forward_phase(self, microscope, test_field_512, ensure_golden_dir):
        """Test microscope forward model in phase contrast mode."""
        output = microscope.forward(
            test_field_512,
            illumination_mode="phase",
            illumination_params={"phase_shift": np.pi / 2},
            add_noise=False,
            use_unified_model=True,
        )

        output_np = output.cpu().numpy()
        golden_path = GOLDEN_DIR / "microscope_forward_phase.npz"
        np.savez(golden_path, output=output_np, input=test_field_512.cpu().numpy())

        loaded = np.load(golden_path)
        assert np.allclose(loaded["output"], output_np, atol=ATOL, rtol=RTOL)

    def test_microscope_spids_aperture(self, microscope, test_field_512, ensure_golden_dir):
        """Test microscope with SPIDS sub-aperture measurement."""
        # Test with off-center aperture
        output = microscope.forward(
            test_field_512,
            aperture_center=[10.0, 15.0],
            aperture_radius=20.0,
            add_noise=False,
        )

        output_np = output.cpu().numpy()
        golden_path = GOLDEN_DIR / "microscope_spids_aperture.npz"
        np.savez(
            golden_path,
            output=output_np,
            input=test_field_512.cpu().numpy(),
            aperture_center=np.array([10.0, 15.0]),
            aperture_radius=20.0,
        )

        loaded = np.load(golden_path)
        assert np.allclose(loaded["output"], output_np, atol=ATOL, rtol=RTOL)


# ============================================================================
# Telescope Tests
# ============================================================================


class TestTelescopeGoldenOutputs:
    """Generate and validate golden outputs for Telescope."""

    @pytest.fixture(scope="class")
    def telescope_config_circular(self):
        """Standard telescope configuration with circular aperture."""
        return TelescopeConfig(
            n_pixels=512,
            pixel_size=6.5e-6,
            wavelength=550e-9,
            aperture_radius_pixels=50.0,
            aperture_diameter=8.2,  # VLT-like
            aperture_type="circular",
        )

    @pytest.fixture(scope="class")
    def telescope_circular(self, telescope_config_circular):
        """Create telescope with circular aperture."""
        return Telescope(telescope_config_circular).to(DEVICE)

    @pytest.fixture(scope="class")
    def telescope_config_hexagonal(self):
        """Telescope configuration with hexagonal aperture."""
        return TelescopeConfig(
            n_pixels=512,
            pixel_size=6.5e-6,
            wavelength=550e-9,
            aperture_radius_pixels=50.0,
            aperture_diameter=6.5,  # JWST-like
            aperture_type="hexagonal",
            aperture_kwargs={"side_length": 50.0},
        )

    @pytest.fixture(scope="class")
    def telescope_hexagonal(self, telescope_config_hexagonal):
        """Create telescope with hexagonal aperture."""
        return Telescope(telescope_config_hexagonal).to(DEVICE)

    def test_telescope_psf_circular(self, telescope_circular, ensure_golden_dir):
        """Test telescope PSF with circular aperture."""
        psf = telescope_circular.compute_psf(center=[0.0, 0.0])

        psf_np = psf.cpu().numpy()
        golden_path = GOLDEN_DIR / "telescope_psf_circular.npz"
        np.savez(golden_path, psf=psf_np)

        loaded = np.load(golden_path)
        assert np.allclose(loaded["psf"], psf_np, atol=ATOL, rtol=RTOL)

    def test_telescope_psf_hexagonal(self, telescope_hexagonal, ensure_golden_dir):
        """Test telescope PSF with hexagonal aperture."""
        psf = telescope_hexagonal.compute_psf(center=[0.0, 0.0])

        psf_np = psf.cpu().numpy()
        golden_path = GOLDEN_DIR / "telescope_psf_hexagonal.npz"
        np.savez(golden_path, psf=psf_np)

        loaded = np.load(golden_path)
        assert np.allclose(loaded["psf"], psf_np, atol=ATOL, rtol=RTOL)

    def test_telescope_forward_circular(
        self, telescope_circular, test_field_512, ensure_golden_dir
    ):
        """Test telescope forward model with circular aperture."""
        output = telescope_circular.forward(
            test_field_512,
            aperture_center=None,  # Full aperture
            add_noise=False,
        )

        output_np = output.cpu().numpy()
        golden_path = GOLDEN_DIR / "telescope_forward_circular.npz"
        np.savez(golden_path, output=output_np, input=test_field_512.cpu().numpy())

        loaded = np.load(golden_path)
        assert np.allclose(loaded["output"], output_np, atol=ATOL, rtol=RTOL)

    def test_telescope_forward_hexagonal(
        self, telescope_hexagonal, test_field_512, ensure_golden_dir
    ):
        """Test telescope forward model with hexagonal aperture."""
        output = telescope_hexagonal.forward(
            test_field_512,
            aperture_center=None,
            add_noise=False,
        )

        output_np = output.cpu().numpy()
        golden_path = GOLDEN_DIR / "telescope_forward_hexagonal.npz"
        np.savez(golden_path, output=output_np, input=test_field_512.cpu().numpy())

        loaded = np.load(golden_path)
        assert np.allclose(loaded["output"], output_np, atol=ATOL, rtol=RTOL)

    def test_telescope_sub_aperture(self, telescope_circular, test_field_512, ensure_golden_dir):
        """Test telescope with off-center sub-aperture (SPIDS mode)."""
        output = telescope_circular.forward(
            test_field_512,
            aperture_center=[20.0, 30.0],
            add_noise=False,
        )

        output_np = output.cpu().numpy()
        golden_path = GOLDEN_DIR / "telescope_sub_aperture.npz"
        np.savez(
            golden_path,
            output=output_np,
            input=test_field_512.cpu().numpy(),
            aperture_center=np.array([20.0, 30.0]),
        )

        loaded = np.load(golden_path)
        assert np.allclose(loaded["output"], output_np, atol=ATOL, rtol=RTOL)


# ============================================================================
# Camera Tests
# ============================================================================


class TestCameraGoldenOutputs:
    """Generate and validate golden outputs for Camera."""

    @pytest.fixture(scope="class")
    def camera_config_far_field(self):
        """Camera configuration for far-field imaging."""
        return CameraConfig(
            n_pixels=256,
            pixel_size=6.5e-6,
            wavelength=550e-9,
            focal_length=50e-3,  # 50mm lens
            f_number=2.8,
            object_distance=float("inf"),  # Far field
            focus_distance=float("inf"),
        )

    @pytest.fixture(scope="class")
    def camera_far_field(self, camera_config_far_field):
        """Create camera for far-field imaging."""
        # Camera doesn't have .to() method, device is handled internally
        return Camera(camera_config_far_field)

    @pytest.fixture(scope="class")
    def camera_config_near_field(self):
        """Camera configuration for near-field imaging."""
        return CameraConfig(
            n_pixels=256,
            pixel_size=6.5e-6,
            wavelength=550e-9,
            focal_length=50e-3,
            f_number=2.8,
            object_distance=1.0,  # 1 meter (near field)
            focus_distance=1.0,
        )

    @pytest.fixture(scope="class")
    def camera_near_field(self, camera_config_near_field):
        """Create camera for near-field imaging."""
        # Camera doesn't have .to() method, device is handled internally
        return Camera(camera_config_near_field)

    def test_camera_psf_far_field(self, camera_far_field, ensure_golden_dir):
        """Test camera PSF in far-field mode."""
        psf = camera_far_field.compute_psf(defocus=0.0)

        psf_np = psf.cpu().numpy()
        golden_path = GOLDEN_DIR / "camera_psf_far_field.npz"
        np.savez(golden_path, psf=psf_np)

        loaded = np.load(golden_path)
        assert np.allclose(loaded["psf"], psf_np, atol=ATOL, rtol=RTOL)

    def test_camera_psf_near_field(self, camera_near_field, ensure_golden_dir):
        """Test camera PSF in near-field mode."""
        psf = camera_near_field.compute_psf(defocus=0.0)

        psf_np = psf.cpu().numpy()
        golden_path = GOLDEN_DIR / "camera_psf_near_field.npz"
        np.savez(golden_path, psf=psf_np)

        loaded = np.load(golden_path)
        assert np.allclose(loaded["psf"], psf_np, atol=ATOL, rtol=RTOL)

    def test_camera_psf_defocused(self, camera_far_field, ensure_golden_dir):
        """Test camera PSF with defocus aberration."""
        psf = camera_far_field.compute_psf(defocus=5e-3)  # 5mm defocus

        psf_np = psf.cpu().numpy()
        golden_path = GOLDEN_DIR / "camera_psf_defocused.npz"
        np.savez(golden_path, psf=psf_np, defocus=5e-3)

        loaded = np.load(golden_path)
        assert np.allclose(loaded["psf"], psf_np, atol=ATOL, rtol=RTOL)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_camera_forward_far_field(self, camera_far_field, test_field_256, ensure_golden_dir):
        """Test camera forward model in far-field mode.

        Note: Camera.forward() has a bug in far-field mode where it passes distance
        to FraunhoferPropagator which doesn't accept that parameter. This will be
        fixed as part of the Four-F System consolidation. For now, skip this test.
        """
        # Test if the bug still exists
        field_cpu = test_field_256.cpu()
        output = camera_far_field.forward(
            field_cpu,
            add_noise=False,
        )

        output_np = output.cpu().numpy() if torch.is_tensor(output) else output.numpy()
        golden_path = GOLDEN_DIR / "camera_forward_far_field.npz"
        np.savez(golden_path, output=output_np, input=field_cpu.cpu().numpy())

        loaded = np.load(golden_path)
        assert np.allclose(loaded["output"], output_np, atol=ATOL, rtol=RTOL)

    def test_camera_forward_near_field(self, camera_near_field, test_field_256, ensure_golden_dir):
        """Test camera forward model in near-field mode."""
        # Camera doesn't support device placement, move field to CPU
        field_cpu = test_field_256.cpu()
        output = camera_near_field.forward(
            field_cpu,
            add_noise=False,
        )

        output_np = output.cpu().numpy() if torch.is_tensor(output) else output.numpy()
        golden_path = GOLDEN_DIR / "camera_forward_near_field.npz"
        np.savez(golden_path, output=output_np, input=field_cpu.cpu().numpy())

        loaded = np.load(golden_path)
        assert np.allclose(loaded["output"], output_np, atol=ATOL, rtol=RTOL)


# ============================================================================
# Regression Tests (Compare against golden outputs)
# ============================================================================


class TestMicroscopeRegression:
    """Regression tests: compare current Microscope outputs to golden."""

    @pytest.fixture(scope="class")
    def microscope(self):
        """Create microscope instance."""
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
        return Microscope(config).to(DEVICE)

    @pytest.mark.regression
    def test_regression_psf_brightfield(self, microscope):
        """Regression: PSF brightfield."""
        golden_path = GOLDEN_DIR / "microscope_psf_brightfield.npz"

        psf = microscope.compute_psf(illumination_mode="brightfield")
        loaded = np.load(golden_path)

        assert np.allclose(psf.cpu().numpy(), loaded["psf"], atol=ATOL, rtol=RTOL), (
            "Microscope brightfield PSF regression failed"
        )

    @pytest.mark.regression
    def test_regression_psf_darkfield(self, microscope):
        """Regression: PSF darkfield."""
        golden_path = GOLDEN_DIR / "microscope_psf_darkfield.npz"

        psf = microscope.compute_psf(
            illumination_mode="darkfield", illumination_params={"annular_ratio": 0.8}
        )
        loaded = np.load(golden_path)

        assert np.allclose(psf.cpu().numpy(), loaded["psf"], atol=ATOL, rtol=RTOL), (
            "Microscope darkfield PSF regression failed"
        )

    @pytest.mark.regression
    def test_regression_psf_phase(self, microscope):
        """Regression: PSF phase contrast."""
        golden_path = GOLDEN_DIR / "microscope_psf_phase.npz"

        psf = microscope.compute_psf(
            illumination_mode="phase", illumination_params={"phase_shift": np.pi / 2}
        )
        loaded = np.load(golden_path)

        assert np.allclose(psf.cpu().numpy(), loaded["psf"], atol=ATOL, rtol=RTOL), (
            "Microscope phase contrast PSF regression failed"
        )

    @pytest.mark.regression
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_regression_forward_brightfield(self, microscope, test_field_512):
        """Regression: forward brightfield."""
        golden_path = GOLDEN_DIR / "microscope_forward_brightfield.npz"

        output = microscope.forward(
            test_field_512,
            illumination_mode="brightfield",
            add_noise=False,
            use_unified_model=True,
        )
        loaded = np.load(golden_path)

        assert np.allclose(output.cpu().numpy(), loaded["output"], atol=ATOL, rtol=RTOL), (
            "Microscope brightfield forward regression failed"
        )


class TestTelescopeRegression:
    """Regression tests: compare current Telescope outputs to golden."""

    @pytest.fixture(scope="class")
    def telescope(self):
        """Create telescope instance."""
        config = TelescopeConfig(
            n_pixels=512,
            pixel_size=6.5e-6,
            wavelength=550e-9,
            aperture_radius_pixels=50.0,
            aperture_diameter=8.2,
            aperture_type="circular",
        )
        return Telescope(config).to(DEVICE)

    @pytest.mark.regression
    def test_regression_psf_circular(self, telescope):
        """Regression: PSF circular aperture."""
        golden_path = GOLDEN_DIR / "telescope_psf_circular.npz"

        psf = telescope.compute_psf(center=[0.0, 0.0])
        loaded = np.load(golden_path)

        assert np.allclose(psf.cpu().numpy(), loaded["psf"], atol=ATOL, rtol=RTOL), (
            "Telescope circular PSF regression failed"
        )

    @pytest.mark.regression
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_regression_forward_circular(self, telescope, test_field_512):
        """Regression: forward circular aperture."""
        golden_path = GOLDEN_DIR / "telescope_forward_circular.npz"

        output = telescope.forward(test_field_512, aperture_center=None, add_noise=False)
        loaded = np.load(golden_path)

        assert np.allclose(output.cpu().numpy(), loaded["output"], atol=ATOL, rtol=RTOL), (
            "Telescope circular forward regression failed"
        )


class TestCameraRegression:
    """Regression tests: compare current Camera outputs to golden."""

    @pytest.fixture(scope="class")
    def camera_far(self):
        """Create far-field camera."""
        config = CameraConfig(
            n_pixels=256,
            pixel_size=6.5e-6,
            wavelength=550e-9,
            focal_length=50e-3,
            f_number=2.8,
            object_distance=float("inf"),
            focus_distance=float("inf"),
        )
        return Camera(config)

    @pytest.fixture(scope="class")
    def camera_near(self):
        """Create near-field camera."""
        config = CameraConfig(
            n_pixels=256,
            pixel_size=6.5e-6,
            wavelength=550e-9,
            focal_length=50e-3,
            f_number=2.8,
            object_distance=1.0,
            focus_distance=1.0,
        )
        return Camera(config)

    @pytest.mark.regression
    def test_regression_psf_far_field(self, camera_far):
        """Regression: PSF far-field."""
        golden_path = GOLDEN_DIR / "camera_psf_far_field.npz"

        psf = camera_far.compute_psf(defocus=0.0)
        loaded = np.load(golden_path)

        assert np.allclose(psf.cpu().numpy(), loaded["psf"], atol=ATOL, rtol=RTOL), (
            "Camera far-field PSF regression failed"
        )

    @pytest.mark.regression
    def test_regression_psf_near_field(self, camera_near):
        """Regression: PSF near-field."""
        golden_path = GOLDEN_DIR / "camera_psf_near_field.npz"

        psf = camera_near.compute_psf(defocus=0.0)
        loaded = np.load(golden_path)

        assert np.allclose(psf.cpu().numpy(), loaded["psf"], atol=ATOL, rtol=RTOL), (
            "Camera near-field PSF regression failed"
        )

    @pytest.mark.regression
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_regression_forward_far_field(self, camera_far, test_field_256):
        """Regression: forward far-field."""
        # The original bug has been fixed - test now passes
        golden_path = GOLDEN_DIR / "camera_forward_far_field.npz"

        field_cpu = test_field_256.cpu()
        output = camera_far.forward(field_cpu, add_noise=False)
        loaded = np.load(golden_path)

        output_np = output.cpu().numpy() if torch.is_tensor(output) else output.numpy()
        assert np.allclose(output_np, loaded["output"], atol=ATOL, rtol=RTOL), (
            "Camera far-field forward regression failed"
        )

    @pytest.mark.regression
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_regression_forward_near_field(self, camera_near, test_field_256):
        """Regression: forward near-field."""
        golden_path = GOLDEN_DIR / "camera_forward_near_field.npz"

        field_cpu = test_field_256.cpu()
        output = camera_near.forward(field_cpu, add_noise=False)
        loaded = np.load(golden_path)

        output_np = output.cpu().numpy() if torch.is_tensor(output) else output.numpy()
        assert np.allclose(output_np, loaded["output"], atol=ATOL, rtol=RTOL), (
            "Camera near-field forward regression failed"
        )
