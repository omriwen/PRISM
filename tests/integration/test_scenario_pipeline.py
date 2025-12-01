"""Integration tests: Scenario system with SPIDS instruments and pipeline.

This module tests end-to-end integration of the scenario configuration system
with SPIDS instruments and imaging pipelines.
"""

from __future__ import annotations

import torch

from prism.core.instruments import (
    Camera,
    CameraConfig,
    Microscope,
    MicroscopeConfig,
    create_instrument,
)
from prism.core.targets import USAF1951Config, USAF1951Target, create_target
from prism.scenarios import (
    DroneBuilder,
    DroneScenarioConfig,
    MicroscopeBuilder,
    MicroscopeScenarioConfig,
    get_scenario_preset,
    list_scenario_presets,
)


class TestMicroscopeScenarioToInstrument:
    """Test microscope scenario to instrument conversion and usage."""

    def test_microscope_scenario_to_working_instrument(self) -> None:
        """Test microscope scenario converts to working instrument."""
        # Load preset
        scenario = get_scenario_preset("microscope_40x_air")

        # Convert to instrument config
        config = scenario.to_instrument_config()

        # Verify config is valid
        assert isinstance(config, MicroscopeConfig)
        config.validate()

        # Create instrument
        microscope = Microscope(config)

        # Verify instrument is functional
        assert microscope.get_instrument_type() == "microscope"
        assert microscope.resolution_limit > 0

    def test_microscope_scenario_forward_pass(self) -> None:
        """Test forward pass through microscope from scenario."""
        # Create scenario with parameters that satisfy Nyquist sampling
        # For 40x 0.9NA: min sampling = lambda/(4*NA) = 550nm/(4*0.9) = 153nm
        # Object pixel size = sensor_pixel_size / magnification
        # Need: sensor_pixel_size / 40 < 153nm -> sensor_pixel_size < 6.12um
        scenario = MicroscopeScenarioConfig(
            objective_spec="40x_0.9NA_air",
            illumination_mode="brightfield",
            n_pixels=128,  # Smaller for test speed
            sensor_pixel_size=5.0e-6,  # Smaller pixels to satisfy Nyquist
        )

        # Convert to instrument
        config = scenario.to_instrument_config()
        microscope = Microscope(config)

        # Create test field (point source)
        test_field = torch.zeros(128, 128, dtype=torch.complex64)
        test_field[64, 64] = 1.0 + 0j

        # Forward pass
        output = microscope.forward(test_field, illumination_mode="brightfield")

        # Verify output
        assert output.shape == (128, 128)
        assert output.dtype == torch.float32
        assert output.min() >= 0

    def test_microscope_builder_to_instrument(self) -> None:
        """Test builder pattern creates functional instrument."""
        scenario = (
            MicroscopeBuilder()
            .objective("60x_1.2NA_water")
            .illumination("phase")
            .wavelength_nm(532)
            .sensor_pixels(256, 6.5)
            .build()
        )

        config = scenario.to_instrument_config()
        microscope = create_instrument(config)

        assert microscope.get_instrument_type() == "microscope"

        # Test different illumination modes
        test_field = torch.zeros(256, 256, dtype=torch.complex64)
        test_field[128, 128] = 1.0 + 0j

        for mode in ["brightfield", "darkfield", "phase"]:
            output = microscope.forward(test_field, illumination_mode=mode)
            assert output.shape == (256, 256)
            assert output.min() >= 0

    def test_high_resolution_microscope_preset(self) -> None:
        """Test high-resolution 100x oil immersion preset."""
        scenario = get_scenario_preset("microscope_100x_oil")

        # Verify physics parameters
        assert scenario.lateral_resolution_nm < 250  # Should be ~240nm

        # Create instrument
        config = scenario.to_instrument_config()
        config.n_pixels = 128  # Override for test speed
        microscope = Microscope(config)

        # Verify resolution
        expected_resolution = 0.61 * scenario.wavelength / 1.4
        assert abs(microscope.resolution_limit - expected_resolution) < 1e-12


class TestDroneScenarioToInstrument:
    """Test drone scenario to instrument conversion and usage."""

    def test_drone_scenario_to_working_instrument(self) -> None:
        """Test drone scenario converts to working instrument config."""
        # Load preset
        scenario = get_scenario_preset("drone_50m_survey")

        # Convert to instrument config
        config = scenario.to_instrument_config()

        # Verify config is valid
        assert isinstance(config, CameraConfig)
        config.validate()

        # Create instrument
        camera = Camera(config)

        # Verify instrument is functional
        assert camera.get_instrument_type() == "camera"
        assert camera.resolution_limit > 0

    def test_drone_scenario_forward_pass(self) -> None:
        """Test forward pass through drone camera from scenario."""
        # Create scenario
        scenario = DroneScenarioConfig(
            lens_spec="35mm_f2.8",
            sensor_spec="aps_c",
            altitude_m=30.0,
            n_pixels=128,  # Smaller for test speed
        )

        # Convert to instrument
        config = scenario.to_instrument_config()
        camera = Camera(config)

        # Create test scene
        test_scene = torch.zeros(128, 128, dtype=torch.complex64)
        test_scene[64, 64] = 1.0 + 0j

        # Forward pass
        output = camera.forward(test_scene, add_noise=False)

        # Verify output
        assert output.shape == (128, 128)
        assert output.dtype == torch.float32
        assert output.min() >= 0

    def test_drone_builder_to_instrument(self) -> None:
        """Test builder pattern creates functional instrument."""
        scenario = (
            DroneBuilder()
            .lens("50mm_f4.0")
            .sensor("full_frame")
            .altitude(100)
            .wavelength_nm(550)
            .n_pixels(128)  # Set n_pixels to match test scene
            .build()
        )

        config = scenario.to_instrument_config()
        camera = create_instrument(config)

        assert camera.get_instrument_type() == "camera"

        # Test forward pass with matching size
        test_scene = torch.zeros(128, 128, dtype=torch.complex64)
        test_scene[64, 64] = 1.0 + 0j

        # With and without noise
        output_clean = camera.forward(test_scene, add_noise=False)
        output_noisy = camera.forward(test_scene, add_noise=True)

        assert output_clean.shape == (128, 128)
        assert output_noisy.shape == (128, 128)


class TestScenarioWithUSAFTarget:
    """Test imaging USAF-1951 target through scenario instruments."""

    def test_usaf_target_through_microscope(self) -> None:
        """Test imaging USAF-1951 target through microscope scenario."""
        # Create USAF-1951 target (small size for test speed)
        target_config = USAF1951Config(size=256, groups=(0, 1, 2))
        target = USAF1951Target(target_config)
        ground_truth = target.generate()

        # Verify target generation
        assert ground_truth.shape == (256, 256)
        assert ground_truth.min() >= 0
        assert ground_truth.max() <= 1

        # Create microscope scenario with matching size
        # Use smaller pixel size to satisfy Nyquist sampling
        # For 40x 0.9NA: min sampling = lambda/(4*NA) = 550nm/(4*0.9) = 153nm
        # Need: sensor_pixel_size / 40 < 153nm -> sensor_pixel_size < 6.12um
        scenario = MicroscopeScenarioConfig(
            objective_spec="40x_0.9NA_air",
            illumination_mode="brightfield",
            n_pixels=256,
            sensor_pixel_size=5.0e-6,  # Smaller to satisfy Nyquist
        )
        config = scenario.to_instrument_config()
        microscope = Microscope(config)

        # Convert target to complex field (amplitude modulation)
        field = ground_truth.to(torch.complex64)

        # Image the target
        image = microscope.forward(field, illumination_mode="brightfield")

        # Verify output
        assert image.shape == (256, 256)
        assert image.min() >= 0
        assert image.dtype == torch.float32

    def test_usaf_target_factory(self) -> None:
        """Test USAF-1951 target creation via factory function."""
        target = create_target("usaf1951", size=128, groups=(0, 1))
        image = target.generate()

        assert image.shape == (128, 128)
        assert image.min() >= 0
        assert image.max() <= 1

        # Verify resolution elements are available
        elements = target.resolution_elements
        assert "G0E1" in elements
        assert "G1E6" in elements

    def test_usaf_frequency_with_microscope_resolution(self) -> None:
        """Test that USAF frequencies relate to microscope resolution."""
        # Create microscope with known resolution
        scenario = MicroscopeScenarioConfig(
            objective_spec="40x_0.9NA_air",
            wavelength=550e-9,
        )

        # Get resolution from scenario (in nm, convert to m)
        resolution_m = scenario.lateral_resolution_nm * 1e-9

        # The microscope should resolve USAF elements with
        # bar width > resolution limit

        # Group 2, Element 1 has frequency 4 lp/mm = bar width 0.125mm = 125µm
        # This should be resolvable by the microscope
        bar_width_g2e1 = USAF1951Target.get_bar_width_mm(2, 1) * 1e-3  # Convert to m
        assert bar_width_g2e1 > resolution_m  # Should be resolvable


class TestAllPresetsValid:
    """Validate all scenario presets work with instruments."""

    def test_all_microscope_presets_create_instruments(self) -> None:
        """Test all microscope presets create valid instruments."""
        microscope_presets = list_scenario_presets("microscope")

        for preset_name in microscope_presets:
            scenario = get_scenario_preset(preset_name)

            # Validate scenario
            scenario.validate()

            # Convert to instrument config
            config = scenario.to_instrument_config()
            assert isinstance(config, MicroscopeConfig)
            config.validate()

            # Create instrument (with reduced pixels for speed)
            config.n_pixels = 64
            microscope = Microscope(config)

            # Verify basic properties
            assert microscope.get_instrument_type() == "microscope"
            assert microscope.resolution_limit > 0

    def test_all_drone_presets_create_instruments(self) -> None:
        """Test all drone presets create valid instruments."""
        drone_presets = list_scenario_presets("drone")

        for preset_name in drone_presets:
            scenario = get_scenario_preset(preset_name)

            # Validate scenario
            scenario.validate()

            # Convert to instrument config
            config = scenario.to_instrument_config()
            assert isinstance(config, CameraConfig)
            config.validate()

            # Create instrument (with reduced pixels for speed)
            config.n_pixels = 64
            camera = Camera(config)

            # Verify basic properties
            assert camera.get_instrument_type() == "camera"
            assert camera.resolution_limit > 0


class TestScenarioPhysicsConsistency:
    """Test that scenario physics parameters are consistent with instruments."""

    def test_microscope_resolution_consistency(self) -> None:
        """Test microscope resolution is consistent between scenario and instrument."""
        for preset_name in list_scenario_presets("microscope"):
            scenario = get_scenario_preset(preset_name)

            # Get scenario resolution (in nm)
            scenario_resolution_nm = scenario.lateral_resolution_nm

            # Create instrument
            config = scenario.to_instrument_config()
            config.n_pixels = 64
            microscope = Microscope(config)

            # Get instrument resolution (in m), convert to nm
            instrument_resolution_nm = microscope.resolution_limit * 1e9

            # Should be approximately equal (allow 5% tolerance)
            assert (
                abs(scenario_resolution_nm - instrument_resolution_nm) / scenario_resolution_nm
                < 0.05
            )

    def test_drone_gsd_calculation(self) -> None:
        """Test drone GSD is properly calculated."""
        # Create scenario with known parameters
        scenario = DroneScenarioConfig(
            lens_spec="50mm_f4.0",
            sensor_spec="full_frame",  # 6.5µm pixels
            altitude_m=100.0,
        )

        # GSD = altitude * pixel_pitch / focal_length
        # GSD = 100m * 6.5e-6m / 0.05m = 1.3cm
        expected_gsd_cm = (100.0 * 6.5e-6 / 50e-3) * 100

        assert abs(scenario.actual_gsd_cm - expected_gsd_cm) < 0.01

    def test_fresnel_number_affects_propagator(self) -> None:
        """Test that Fresnel number is calculated for proper propagator selection."""
        # Close range (should use angular spectrum)
        close_scenario = DroneScenarioConfig(
            lens_spec="50mm_f4.0",
            altitude_m=10.0,
        )

        # Long range (should use Fraunhofer)
        far_scenario = DroneScenarioConfig(
            lens_spec="50mm_f4.0",
            altitude_m=100.0,
        )

        # Fresnel number should increase with decreasing distance
        # (a² / λz increases as z decreases)
        assert close_scenario.fresnel_number > far_scenario.fresnel_number
