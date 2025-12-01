"""Unit tests for optical instruments (Microscope and Camera)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from prism.core.instruments import (
    Camera,
    CameraConfig,
    InstrumentConfig,
    Microscope,
    MicroscopeConfig,
    Telescope,
    TelescopeConfig,
    create_instrument,
)


class TestMicroscope:
    """Test microscope functionality."""

    def test_microscope_creation(self) -> None:
        """Test microscope instantiation."""
        config = MicroscopeConfig(
            numerical_aperture=1.4,
            magnification=100,
            wavelength=532e-9,
            medium_index=1.515,  # Oil immersion
            n_pixels=256,
            pixel_size=6.5e-6,
        )
        microscope = Microscope(config)
        assert microscope is not None
        assert microscope.get_instrument_type() == "microscope"

    def test_microscope_creation_air(self) -> None:
        """Test microscope with air objective."""
        config = MicroscopeConfig(
            numerical_aperture=0.9,
            magnification=60,  # Increased to satisfy Nyquist sampling
            wavelength=532e-9,
            medium_index=1.0,  # Air
            n_pixels=256,
            pixel_size=6.5e-6,
        )
        microscope = Microscope(config)
        assert microscope is not None
        assert microscope.get_instrument_type() == "microscope"

    def test_microscope_psf_2d(self) -> None:
        """Test 2D PSF computation."""
        config = MicroscopeConfig(
            numerical_aperture=1.4,
            magnification=100,
            wavelength=532e-9,
            medium_index=1.515,
            n_pixels=128,
            pixel_size=6.5e-6,
        )
        microscope = Microscope(config)
        psf = microscope.compute_psf()

        # Check PSF properties
        assert psf.shape == (128, 128)
        assert psf.max() <= 1.0 + 1e-6  # Normalized (allow small numerical error)
        assert psf.min() >= 0  # Non-negative

    def test_microscope_psf_3d(self) -> None:
        """Test 3D PSF computation."""
        config = MicroscopeConfig(
            numerical_aperture=1.4,
            magnification=100,
            wavelength=532e-9,
            medium_index=1.515,
            n_pixels=64,
            pixel_size=6.5e-6,
        )
        microscope = Microscope(config)
        psf = microscope.compute_psf(z_slices=11)

        # Check PSF properties
        assert psf.shape == (11, 64, 64)
        assert psf.max() <= 1.0 + 1e-6
        assert psf.min() >= 0

    def test_na_validation(self) -> None:
        """Test NA validation."""
        with pytest.raises(ValueError, match="NA.*cannot exceed medium index"):
            # NA > medium index should fail
            MicroscopeConfig(
                numerical_aperture=1.6,
                medium_index=1.33,  # Water
                magnification=100,
                wavelength=532e-9,
                n_pixels=256,
                pixel_size=6.5e-6,
            )

    def test_sampling_validation(self) -> None:
        """Test Nyquist sampling validation."""
        with pytest.raises(ValueError, match="Undersampling"):
            # Undersampling should fail
            MicroscopeConfig(
                numerical_aperture=1.4,
                magnification=10,  # Too low for pixel size
                wavelength=532e-9,
                medium_index=1.515,
                pixel_size=20e-6,  # Too large
                n_pixels=256,
            )

    def test_resolution_limit(self) -> None:
        """Test theoretical resolution calculation."""
        config = MicroscopeConfig(
            numerical_aperture=1.4,
            wavelength=532e-9,
            magnification=100,
            medium_index=1.515,
            n_pixels=256,
            pixel_size=6.5e-6,
        )
        microscope = Microscope(config)

        # Abbe limit: 0.61 * λ / NA
        expected = 0.61 * 532e-9 / 1.4
        assert abs(microscope.resolution_limit - expected) < 1e-12

    def test_brightfield_imaging(self) -> None:
        """Test brightfield imaging mode."""
        config = MicroscopeConfig(
            numerical_aperture=1.4,
            magnification=100,
            wavelength=532e-9,
            medium_index=1.515,
            n_pixels=128,
            pixel_size=6.5e-6,
        )
        microscope = Microscope(config)

        # Create simple test sample (point source)
        sample = torch.zeros(128, 128, dtype=torch.complex64)
        sample[64, 64] = 1.0 + 0j

        # Forward imaging
        image = microscope.forward(sample, illumination_mode="brightfield")

        assert image.shape == (128, 128)
        assert image.dtype == torch.float32
        assert image.min() >= 0
        assert image.max() > 0

    def test_darkfield_imaging(self) -> None:
        """Test darkfield imaging mode."""
        config = MicroscopeConfig(
            numerical_aperture=1.4,
            magnification=100,
            wavelength=532e-9,
            medium_index=1.515,
            n_pixels=128,
            pixel_size=6.5e-6,
        )
        microscope = Microscope(config)

        # Create sample
        sample = torch.zeros(128, 128, dtype=torch.complex64)
        sample[64, 64] = 1.0 + 0j

        # Forward imaging with darkfield
        image = microscope.forward(sample, illumination_mode="darkfield")

        assert image.shape == (128, 128)
        assert image.dtype == torch.float32
        assert image.min() >= 0

    def test_phase_contrast(self) -> None:
        """Test phase contrast imaging mode."""
        config = MicroscopeConfig(
            numerical_aperture=1.4,
            magnification=100,
            wavelength=532e-9,
            medium_index=1.515,
            n_pixels=128,
            pixel_size=6.5e-6,
        )
        microscope = Microscope(config)

        # Create phase sample
        sample = torch.ones(128, 128, dtype=torch.complex64)
        # Add phase variation
        phase = torch.zeros(128, 128)
        phase[60:68, 60:68] = np.pi / 2
        sample = torch.exp(1j * phase).to(torch.complex64)

        # Forward imaging with phase contrast
        image = microscope.forward(sample, illumination_mode="phase")

        assert image.shape == (128, 128)
        assert image.dtype == torch.float32
        assert image.min() >= 0

    def test_dic_imaging(self) -> None:
        """Test DIC imaging mode."""
        config = MicroscopeConfig(
            numerical_aperture=1.4,
            magnification=100,
            wavelength=532e-9,
            medium_index=1.515,
            n_pixels=128,
            pixel_size=6.5e-6,
        )
        microscope = Microscope(config)

        # Create sample
        sample = torch.ones(128, 128, dtype=torch.complex64)

        # Forward imaging with DIC
        image = microscope.forward(sample, illumination_mode="dic")

        assert image.shape == (128, 128)
        assert image.dtype == torch.float32
        assert image.min() >= 0

    def test_get_info(self) -> None:
        """Test information summary."""
        config = MicroscopeConfig(
            numerical_aperture=1.4,
            magnification=100,
            wavelength=532e-9,
            medium_index=1.515,
            n_pixels=256,
            pixel_size=6.5e-6,
        )
        microscope = Microscope(config)

        info = microscope.get_info()

        assert isinstance(info, dict)
        assert "type" in info  # Note: uses 'type' not 'instrument_type'
        assert info["type"] == "microscope"
        assert "numerical_aperture" in info
        assert "magnification" in info
        assert "resolution_limit" in info


class TestCamera:
    """Test camera functionality."""

    def test_camera_creation(self) -> None:
        """Test camera instantiation."""
        config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            sensor_size=(36e-3, 24e-3),
            n_pixels=512,
            pixel_size=6.5e-6,
            wavelength=550e-9,
        )
        camera = Camera(config)
        assert camera is not None
        assert camera.get_instrument_type() == "camera"

    def test_camera_finite_distance(self) -> None:
        """Test camera with finite object distance."""
        config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            object_distance=2.0,
            n_pixels=512,
            pixel_size=6.5e-6,
            wavelength=550e-9,
        )
        camera = Camera(config)
        assert camera is not None

    def test_thin_lens_equation(self) -> None:
        """Test thin lens calculations."""
        config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            object_distance=2.0,
            n_pixels=256,
            pixel_size=6.5e-6,
            wavelength=550e-9,
        )
        camera = Camera(config)

        # 1/f = 1/do + 1/di
        di = camera.calculate_image_distance()
        expected = 1 / (1 / 0.05 - 1 / 2.0)
        assert abs(di - expected) < 1e-6

    def test_depth_of_field(self) -> None:
        """Test DOF calculation."""
        config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            focus_distance=2.0,
            n_pixels=256,
            pixel_size=6.5e-6,
            wavelength=550e-9,
        )
        camera = Camera(config)

        near, far = camera.calculate_depth_of_field()
        assert near < 2.0 < far
        assert near > 0

    def test_magnification(self) -> None:
        """Test magnification calculation."""
        config = CameraConfig(
            focal_length=100e-3,
            f_number=2.8,
            object_distance=0.5,  # 0.5m
            n_pixels=256,
            pixel_size=6.5e-6,
            wavelength=550e-9,
        )
        camera = Camera(config)

        mag = camera.calculate_magnification()
        # At 0.5m object distance with 100mm lens
        expected_di = 1 / (1 / 0.1 - 1 / 0.5)
        expected_mag = -expected_di / 0.5
        assert abs(mag - expected_mag) < 0.01

    def test_camera_psf(self) -> None:
        """Test PSF computation."""
        config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            n_pixels=128,
            pixel_size=6.5e-6,
            wavelength=550e-9,
        )
        camera = Camera(config)
        psf = camera.compute_psf()

        # Check PSF properties
        assert psf.shape == (128, 128)
        assert psf.max() <= 1.0 + 1e-6  # Normalized
        assert psf.min() >= 0

    def test_defocused_psf(self) -> None:
        """Test defocused PSF generation."""
        config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            n_pixels=128,
            pixel_size=6.5e-6,
            wavelength=550e-9,
        )
        camera = Camera(config)
        psf_defocus = camera.compute_psf(defocus=1e-3)  # 1mm defocus

        assert psf_defocus.shape == (128, 128)
        assert psf_defocus.max() <= 1.0 + 1e-6
        assert psf_defocus.min() >= 0

    def test_forward_imaging(self) -> None:
        """Test forward imaging with point source."""
        config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            object_distance=2.0,  # Finite distance to use AngularSpectrum
            n_pixels=128,
            pixel_size=6.5e-6,
            wavelength=550e-9,
        )
        camera = Camera(config)

        # Create point source
        scene = torch.zeros(128, 128, dtype=torch.complex64)
        scene[64, 64] = 1.0 + 0j

        # Image formation
        image = camera.forward(scene, add_noise=False)

        assert image.shape == (128, 128)
        assert image.dtype == torch.float32
        assert image.min() >= 0
        assert image.max() > 0

    def test_forward_imaging_with_noise(self) -> None:
        """Test forward imaging with sensor noise."""
        config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            object_distance=2.0,  # Finite distance to use AngularSpectrum
            n_pixels=128,
            pixel_size=6.5e-6,
            wavelength=550e-9,
        )
        camera = Camera(config)

        # Create point source
        scene = torch.zeros(128, 128, dtype=torch.complex64)
        scene[64, 64] = 1.0 + 0j

        # Image formation with noise
        image = camera.forward(scene, add_noise=True)

        assert image.shape == (128, 128)
        assert image.dtype == torch.float32

    def test_resolution_limit(self) -> None:
        """Test theoretical resolution calculation."""
        config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            wavelength=550e-9,
            n_pixels=256,
            pixel_size=6.5e-6,
        )
        camera = Camera(config)

        # Diffraction-limited spot size: 2.44 * λ * f/#
        expected = 2.44 * 550e-9 * 2.8
        assert abs(camera.resolution_limit - expected) < 1e-12

    def test_fresnel_number_calculation(self) -> None:
        """Test Fresnel number calculation for propagation regime."""
        config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            object_distance=2.0,
            wavelength=550e-9,
            n_pixels=256,
            pixel_size=6.5e-6,
        )
        camera = Camera(config)

        fresnel_number = camera._calculate_fresnel_number()

        # F = a²/(λz) where a is aperture radius
        aperture_radius = camera.aperture_diameter / 2
        expected = aperture_radius**2 / (550e-9 * 2.0)
        assert abs(fresnel_number - expected) < 1e-6

    def test_get_info(self) -> None:
        """Test information summary."""
        config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            object_distance=2.0,
            n_pixels=256,
            pixel_size=6.5e-6,
            wavelength=550e-9,
        )
        camera = Camera(config)

        info = camera.get_info()

        assert isinstance(info, dict)
        assert "type" in info  # Note: uses 'type' not 'instrument_type'
        assert info["type"] == "camera"
        assert "focal_length" in info
        assert "f_number" in info
        assert "resolution_limit" in info


class TestInstrumentFactory:
    """Test instrument factory function."""

    def test_create_telescope(self) -> None:
        """Test telescope creation via factory."""
        config = TelescopeConfig(
            aperture_diameter=8.2,
            focal_length=120.0,
            wavelength=550e-9,
            n_pixels=256,
            pixel_size=13e-6,
            aperture_radius_pixels=50,
        )
        instrument = create_instrument(config)

        assert isinstance(instrument, Telescope)
        assert instrument.get_instrument_type() == "telescope"

    def test_create_microscope(self) -> None:
        """Test microscope creation via factory."""
        config = MicroscopeConfig(
            numerical_aperture=1.4,
            magnification=100,
            wavelength=532e-9,
            medium_index=1.515,
            n_pixels=256,
            pixel_size=6.5e-6,
        )
        instrument = create_instrument(config)

        assert isinstance(instrument, Microscope)
        assert instrument.get_instrument_type() == "microscope"

    def test_create_camera(self) -> None:
        """Test camera creation via factory."""
        config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            object_distance=2.0,
            n_pixels=256,
            pixel_size=6.5e-6,
            wavelength=550e-9,
        )
        instrument = create_instrument(config)

        assert isinstance(instrument, Camera)
        assert instrument.get_instrument_type() == "camera"

    def test_base_config_error(self) -> None:
        """Test that base InstrumentConfig raises error."""
        config = InstrumentConfig(wavelength=550e-9, n_pixels=256, pixel_size=6.5e-6)

        with pytest.raises(TypeError, match="base InstrumentConfig"):
            create_instrument(config)

    def test_invalid_config_type(self) -> None:
        """Test that invalid config type raises error."""

        # Create a mock config object that passes validation but isn't a real instrument config
        class InvalidConfig:
            def validate(self) -> None:
                pass

        invalid_config = InvalidConfig()
        with pytest.raises(TypeError, match="Unknown instrument configuration type"):
            create_instrument(invalid_config)  # type: ignore


class TestInstrumentComparison:
    """Test comparison between different instruments."""

    def test_resolution_limit_comparison(self) -> None:
        """Compare resolution limits across instruments."""
        wavelength = 550e-9

        # Telescope
        tel_config = TelescopeConfig(
            aperture_diameter=8.2,
            focal_length=120.0,
            wavelength=wavelength,
            n_pixels=256,
            pixel_size=13e-6,
            aperture_radius_pixels=50,
        )
        telescope = Telescope(config=tel_config)

        # Microscope
        mic_config = MicroscopeConfig(
            numerical_aperture=1.4,
            magnification=100,
            wavelength=wavelength,
            medium_index=1.515,
            n_pixels=256,
            pixel_size=6.5e-6,
        )
        microscope = Microscope(mic_config)

        # Camera
        cam_config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            wavelength=wavelength,
            n_pixels=256,
            pixel_size=6.5e-6,
        )
        camera = Camera(cam_config)

        # Microscope should have better (smaller) resolution limit than camera
        assert microscope.resolution_limit < camera.resolution_limit
        # Telescope resolution is angular, so different units - just check it's positive
        assert telescope.resolution_limit > 0

    def test_psf_normalization(self) -> None:
        """Test that all instruments produce normalized PSFs."""
        wavelength = 550e-9

        # Telescope PSF
        tel = Telescope(
            config=TelescopeConfig(
                aperture_diameter=8.2,
                focal_length=120.0,
                wavelength=wavelength,
                n_pixels=128,
                pixel_size=13e-6,
                aperture_radius_pixels=50,
            )
        )
        tel_psf = tel.compute_psf()

        # Microscope PSF
        mic = Microscope(
            MicroscopeConfig(
                numerical_aperture=1.4,
                magnification=100,
                wavelength=wavelength,
                medium_index=1.515,
                n_pixels=128,
                pixel_size=6.5e-6,
            )
        )
        mic_psf = mic.compute_psf()

        # Camera PSF
        cam = Camera(
            CameraConfig(
                focal_length=50e-3,
                f_number=2.8,
                wavelength=wavelength,
                n_pixels=128,
                pixel_size=6.5e-6,
            )
        )
        cam_psf = cam.compute_psf()

        # All should be normalized
        assert abs(tel_psf.max() - 1.0) < 1e-6
        assert abs(mic_psf.max() - 1.0) < 1e-6
        assert abs(cam_psf.max() - 1.0) < 1e-6

        # All should be positive
        assert (tel_psf >= 0).all()
        assert (mic_psf >= 0).all()
        assert (cam_psf >= 0).all()
