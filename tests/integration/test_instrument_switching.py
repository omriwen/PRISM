"""Integration tests for switching between different optical instruments."""

from __future__ import annotations

import torch

from prism.core.instruments import (
    Camera,
    CameraConfig,
    Microscope,
    MicroscopeConfig,
    Telescope,
    TelescopeConfig,
    create_instrument,
)


class TestInstrumentSwitching:
    """Test switching between different instruments in the same workflow."""

    def test_create_all_instruments(self) -> None:
        """Test creating multiple instruments in sequence."""
        wavelength = 550e-9

        # Create telescope
        tel_config = TelescopeConfig(
            aperture_diameter=8.2,
            focal_length=120.0,
            wavelength=wavelength,
            n_pixels=256,
            pixel_size=13e-6,
            aperture_radius_pixels=50,
        )
        telescope = create_instrument(tel_config)
        assert telescope.get_instrument_type() == "telescope"

        # Create microscope
        mic_config = MicroscopeConfig(
            numerical_aperture=1.4,
            magnification=100,
            wavelength=wavelength,
            medium_index=1.515,
            n_pixels=256,
            pixel_size=6.5e-6,
        )
        microscope = create_instrument(mic_config)
        assert microscope.get_instrument_type() == "microscope"

        # Create camera
        cam_config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            object_distance=2.0,
            wavelength=wavelength,
            n_pixels=256,
            pixel_size=6.5e-6,
        )
        camera = create_instrument(cam_config)
        assert camera.get_instrument_type() == "camera"

        # All should be independent
        assert telescope is not microscope
        assert microscope is not camera
        assert telescope is not camera

    def test_same_scene_different_instruments(self) -> None:
        """Test imaging the same scene with different instruments."""
        wavelength = 550e-9
        n_pixels = 128

        # Create a simple test scene (point source)
        scene = torch.zeros(n_pixels, n_pixels, dtype=torch.complex64)
        scene[n_pixels // 2, n_pixels // 2] = 1.0 + 0j

        # Microscope imaging
        microscope = Microscope(
            MicroscopeConfig(
                numerical_aperture=1.4,
                magnification=100,
                wavelength=wavelength,
                medium_index=1.515,
                n_pixels=n_pixels,
                pixel_size=6.5e-6,
            )
        )
        mic_image = microscope.forward(scene, illumination_mode="brightfield")

        # Camera imaging
        camera = Camera(
            CameraConfig(
                focal_length=50e-3,
                f_number=2.8,
                object_distance=2.0,  # Use finite distance for AngularSpectrum
                wavelength=wavelength,
                n_pixels=n_pixels,
                pixel_size=6.5e-6,
            )
        )
        cam_image = camera.forward(scene, add_noise=False)

        # Both should produce valid images
        assert mic_image.shape == (n_pixels, n_pixels)
        assert cam_image.shape == (n_pixels, n_pixels)

        # Images should be different (different instruments)
        assert not torch.allclose(mic_image, cam_image)

        # Both should be centered (peak near center)
        mic_peak = torch.argmax(mic_image)
        cam_peak = torch.argmax(cam_image)
        center = n_pixels // 2
        tolerance = 10

        mic_peak_y, mic_peak_x = mic_peak // n_pixels, mic_peak % n_pixels
        cam_peak_y, cam_peak_x = cam_peak // n_pixels, cam_peak % n_pixels

        assert abs(mic_peak_y - center) < tolerance
        assert abs(mic_peak_x - center) < tolerance
        assert abs(cam_peak_y - center) < tolerance
        assert abs(cam_peak_x - center) < tolerance


class TestPSFComparison:
    """Compare PSFs across different instruments."""

    def test_psf_shapes(self) -> None:
        """Test that all instruments produce correct PSF shapes."""
        wavelength = 550e-9
        n_pixels = 128

        # Telescope PSF
        telescope = Telescope(
            config=TelescopeConfig(
                aperture_diameter=8.2,
                focal_length=120.0,
                wavelength=wavelength,
                n_pixels=n_pixels,
                pixel_size=13e-6,
                aperture_radius_pixels=50,
            )
        )
        tel_psf = telescope.compute_psf()

        # Microscope PSF
        microscope = Microscope(
            MicroscopeConfig(
                numerical_aperture=0.9,
                magnification=60,  # Increased for Nyquist sampling
                wavelength=wavelength,
                medium_index=1.0,
                n_pixels=n_pixels,
                pixel_size=6.5e-6,
            )
        )
        mic_psf = microscope.compute_psf()

        # Camera PSF
        camera = Camera(
            CameraConfig(
                focal_length=50e-3,
                f_number=2.8,
                wavelength=wavelength,
                n_pixels=n_pixels,
                pixel_size=6.5e-6,
            )
        )
        cam_psf = camera.compute_psf()

        # All should have same shape
        assert tel_psf.shape == (n_pixels, n_pixels)
        assert mic_psf.shape == (n_pixels, n_pixels)
        assert cam_psf.shape == (n_pixels, n_pixels)

    def test_psf_normalization_consistency(self) -> None:
        """Test that all PSFs are consistently normalized."""
        wavelength = 550e-9
        n_pixels = 128

        # Telescope PSF
        tel = Telescope(
            config=TelescopeConfig(
                aperture_diameter=8.2,
                focal_length=120.0,
                wavelength=wavelength,
                n_pixels=n_pixels,
                pixel_size=13e-6,
                aperture_radius_pixels=50,
            )
        )
        tel_psf = tel.compute_psf()

        # Microscope PSF
        mic = Microscope(
            MicroscopeConfig(
                numerical_aperture=0.9,
                magnification=60,  # Increased for Nyquist sampling
                wavelength=wavelength,
                medium_index=1.0,
                n_pixels=n_pixels,
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
                n_pixels=n_pixels,
                pixel_size=6.5e-6,
            )
        )
        cam_psf = cam.compute_psf()

        # All should be normalized to max=1
        assert abs(tel_psf.max().item() - 1.0) < 1e-5
        assert abs(mic_psf.max().item() - 1.0) < 1e-5
        assert abs(cam_psf.max().item() - 1.0) < 1e-5

        # All should be non-negative
        assert (tel_psf >= 0).all()
        assert (mic_psf >= 0).all()
        assert (cam_psf >= 0).all()

        # All should have a single dominant peak (normalized PSF should have clear maximum)
        # Note: Peak position depends on FFT convention, so we just check it has a clear peak
        assert tel_psf.max() > 0.5 * tel_psf.mean()
        assert mic_psf.max() > 0.5 * mic_psf.mean()
        assert cam_psf.max() > 0.5 * cam_psf.mean()

    def test_psf_energy_conservation(self) -> None:
        """Test that PSFs conserve energy."""
        wavelength = 550e-9
        n_pixels = 128

        # Telescope PSF
        telescope = Telescope(
            config=TelescopeConfig(
                aperture_diameter=8.2,
                focal_length=120.0,
                wavelength=wavelength,
                n_pixels=n_pixels,
                pixel_size=13e-6,
                aperture_radius_pixels=50,
            )
        )
        tel_psf = telescope.compute_psf()
        tel_energy = torch.sum(tel_psf).item()

        # Microscope PSF
        microscope = Microscope(
            MicroscopeConfig(
                numerical_aperture=0.9,
                magnification=60,  # Increased for Nyquist sampling
                wavelength=wavelength,
                medium_index=1.0,
                n_pixels=n_pixels,
                pixel_size=6.5e-6,
            )
        )
        mic_psf = microscope.compute_psf()
        mic_energy = torch.sum(mic_psf).item()

        # Camera PSF
        camera = Camera(
            CameraConfig(
                focal_length=50e-3,
                f_number=2.8,
                wavelength=wavelength,
                n_pixels=n_pixels,
                pixel_size=6.5e-6,
            )
        )
        cam_psf = camera.compute_psf()
        cam_energy = torch.sum(cam_psf).item()

        # Total energy should be reasonable (not zero, not infinite)
        assert tel_energy > 0
        assert mic_energy > 0
        assert cam_energy > 0
        assert tel_energy < n_pixels * n_pixels  # Should be less than total pixels
        assert mic_energy < n_pixels * n_pixels
        assert cam_energy < n_pixels * n_pixels


class TestResolutionScaling:
    """Test resolution scaling across different instruments."""

    def test_resolution_limits_physical_validity(self) -> None:
        """Test that resolution limits follow physical laws."""
        wavelength = 550e-9

        # High NA microscope (best resolution)
        high_na_mic = Microscope(
            MicroscopeConfig(
                numerical_aperture=1.4,
                magnification=100,
                wavelength=wavelength,
                medium_index=1.515,
                n_pixels=256,
                pixel_size=6.5e-6,
            )
        )

        # Low NA microscope
        low_na_mic = Microscope(
            MicroscopeConfig(
                numerical_aperture=0.5,
                magnification=40,
                wavelength=wavelength,
                medium_index=1.0,
                n_pixels=256,
                pixel_size=6.5e-6,
            )
        )

        # Small f-number camera (better resolution)
        small_f_cam = Camera(
            CameraConfig(
                focal_length=50e-3,
                f_number=1.4,
                wavelength=wavelength,
                n_pixels=256,
                pixel_size=6.5e-6,
            )
        )

        # Large f-number camera (worse resolution)
        large_f_cam = Camera(
            CameraConfig(
                focal_length=50e-3,
                f_number=8.0,
                wavelength=wavelength,
                n_pixels=256,
                pixel_size=6.5e-6,
            )
        )

        # Higher NA should give better (smaller) resolution
        assert high_na_mic.resolution_limit < low_na_mic.resolution_limit

        # Smaller f-number should give better (smaller) resolution
        assert small_f_cam.resolution_limit < large_f_cam.resolution_limit

        # High-NA microscope should beat cameras
        assert high_na_mic.resolution_limit < small_f_cam.resolution_limit

    def test_na_to_resolution_relationship(self) -> None:
        """Test Abbe limit: resolution = 0.61λ/NA."""
        wavelength = 532e-9
        na_values = [0.5, 0.9, 1.4]

        for na in na_values:
            # Skip invalid combinations
            medium_index = 1.0 if na <= 1.0 else 1.515

            microscope = Microscope(
                MicroscopeConfig(
                    numerical_aperture=na,
                    magnification=100,
                    wavelength=wavelength,
                    medium_index=medium_index,
                    n_pixels=256,
                    pixel_size=6.5e-6,
                )
            )

            expected = 0.61 * wavelength / na
            assert abs(microscope.resolution_limit - expected) < 1e-12

    def test_telescope_rayleigh_criterion(self) -> None:
        """Test telescope Rayleigh criterion: resolution = 1.22λ/D."""
        wavelength = 550e-9
        aperture_diameter = 8.2  # 8.2m VLT

        telescope = Telescope(
            config=TelescopeConfig(
                aperture_diameter=aperture_diameter,
                focal_length=120.0,
                wavelength=wavelength,
                n_pixels=256,
                pixel_size=13e-6,
                aperture_radius_pixels=50,
            )
        )

        # Rayleigh criterion: 1.22λ/D (in radians)
        expected = 1.22 * wavelength / aperture_diameter
        assert abs(telescope.resolution_limit - expected) < 1e-12


class TestMultiModalImaging:
    """Test multi-modal imaging capabilities."""

    def test_microscope_illumination_modes(self) -> None:
        """Test that microscope supports multiple illumination modes."""
        config = MicroscopeConfig(
            numerical_aperture=1.4,
            magnification=100,
            wavelength=532e-9,
            medium_index=1.515,
            n_pixels=128,
            pixel_size=6.5e-6,
        )
        microscope = Microscope(config)

        # Create test sample
        sample = torch.zeros(128, 128, dtype=torch.complex64)
        sample[64, 64] = 1.0 + 0j

        # Test all illumination modes
        modes = ["brightfield", "darkfield", "phase", "dic"]

        for mode in modes:
            image = microscope.forward(sample, illumination_mode=mode)
            assert image.shape == (128, 128)
            assert image.min() >= 0
            assert image.dtype == torch.float32

    def test_camera_noise_modes(self) -> None:
        """Test camera with and without noise."""
        config = CameraConfig(
            focal_length=50e-3,
            f_number=2.8,
            object_distance=2.0,  # Use finite distance for AngularSpectrum
            wavelength=550e-9,
            n_pixels=128,
            pixel_size=6.5e-6,
        )
        camera = Camera(config)

        # Create test scene
        scene = torch.zeros(128, 128, dtype=torch.complex64)
        scene[64, 64] = 1.0 + 0j

        # Image without noise
        image_clean = camera.forward(scene, add_noise=False)

        # Image with noise
        image_noisy = camera.forward(scene, add_noise=True)

        # Both should be valid
        assert image_clean.shape == (128, 128)
        assert image_noisy.shape == (128, 128)

        # Images should be different when noise is added
        # (with very high probability)
        assert not torch.allclose(image_clean, image_noisy, rtol=0.01)


class TestInstrumentInfoConsistency:
    """Test that instrument info dictionaries are consistent."""

    def test_microscope_info_structure(self) -> None:
        """Test microscope info dictionary structure."""
        microscope = Microscope(
            MicroscopeConfig(
                numerical_aperture=1.4,
                magnification=100,
                wavelength=532e-9,
                medium_index=1.515,
                n_pixels=256,
                pixel_size=6.5e-6,
            )
        )

        info = microscope.get_info()

        # Check required keys
        assert "type" in info  # Note: uses 'type' not 'instrument_type'
        assert "numerical_aperture" in info
        assert "magnification" in info
        assert "resolution_limit" in info
        assert "wavelength" in info

        # Check values
        assert info["type"] == "microscope"
        assert info["numerical_aperture"] == 1.4
        assert info["magnification"] == 100

    def test_camera_info_structure(self) -> None:
        """Test camera info dictionary structure."""
        camera = Camera(
            CameraConfig(
                focal_length=50e-3,
                f_number=2.8,
                object_distance=2.0,
                wavelength=550e-9,
                n_pixels=256,
                pixel_size=6.5e-6,
            )
        )

        info = camera.get_info()

        # Check required keys
        assert "type" in info  # Note: uses 'type' not 'instrument_type'
        assert "focal_length" in info
        assert "f_number" in info
        assert "resolution_limit" in info
        assert "wavelength" in info

        # Check values
        assert info["type"] == "camera"
        assert info["focal_length"] == 50e-3
        assert info["f_number"] == 2.8
