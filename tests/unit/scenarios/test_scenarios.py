"""Unit tests for scenario configuration system."""

import pytest

from prism.core.instruments import CameraConfig, MicroscopeConfig
from prism.scenarios import (
    DroneBuilder,
    DroneScenarioConfig,
    LensSpec,
    MicroscopeBuilder,
    MicroscopeScenarioConfig,
    ObjectiveSpec,
    SensorSpec,
    get_preset_description,
    get_scenario_preset,
    list_scenario_presets,
)


class TestObjectiveSpec:
    """Test microscope objective specification parsing."""

    def test_from_string_100x_oil(self):
        """Test parsing 100x oil immersion objective."""
        spec = ObjectiveSpec.from_string("100x_1.4NA_oil")

        assert spec.magnification == 100
        assert spec.numerical_aperture == pytest.approx(1.4)
        assert spec.immersion_medium == "oil"
        assert spec.medium_index == pytest.approx(1.515)

    def test_from_string_40x_air(self):
        """Test parsing 40x air objective."""
        spec = ObjectiveSpec.from_string("40x_0.9NA_air")

        assert spec.magnification == 40
        assert spec.numerical_aperture == pytest.approx(0.9)
        assert spec.immersion_medium == "air"
        assert spec.medium_index == pytest.approx(1.0)

    def test_from_string_60x_water(self):
        """Test parsing 60x water immersion objective."""
        spec = ObjectiveSpec.from_string("60x_1.2NA_water")

        assert spec.magnification == 60
        assert spec.numerical_aperture == pytest.approx(1.2)
        assert spec.immersion_medium == "water"
        assert spec.medium_index == pytest.approx(1.33)

    def test_from_string_invalid_format(self):
        """Test error on invalid format."""
        with pytest.raises(ValueError, match="Invalid objective spec"):
            ObjectiveSpec.from_string("invalid_spec")

    def test_string_representation(self):
        """Test string representation."""
        spec = ObjectiveSpec(magnification=100, numerical_aperture=1.4, immersion_medium="oil")
        assert str(spec) == "100x_1.4NA_oil"


class TestMicroscopeScenario:
    """Test microscope scenario configuration."""

    def test_basic_configuration(self):
        """Test basic microscope configuration."""
        scenario = MicroscopeScenarioConfig(objective_spec="40x_0.9NA_air")

        assert scenario.scenario_type == "microscope"
        assert scenario.objective_spec.magnification == 40
        assert scenario.objective_spec.numerical_aperture == pytest.approx(0.9)
        assert scenario.illumination_mode == "brightfield"

    def test_resolution_calculation(self):
        """Test Abbe resolution limit calculation."""
        scenario = MicroscopeScenarioConfig(objective_spec="100x_1.4NA_oil", wavelength=550e-9)

        # Abbe limit: 0.61 * lambda / NA
        # 0.61 * 550nm / 1.4 = 239.6 nm
        expected_resolution = 0.61 * 550 / 1.4
        assert scenario.lateral_resolution_nm == pytest.approx(expected_resolution, rel=0.01)

    def test_axial_resolution_calculation(self):
        """Test axial resolution calculation."""
        scenario = MicroscopeScenarioConfig(objective_spec="100x_1.4NA_oil", wavelength=550e-9)

        # Axial resolution: 2 * lambda * n / NA^2
        # 2 * 550nm * 1.515 / 1.4^2 = 849 nm = 0.849 µm
        n = 1.515  # Oil
        expected_axial = (2 * 550 * n / (1.4**2)) / 1000  # Convert nm to µm
        assert scenario.axial_resolution_um == pytest.approx(expected_axial, rel=0.01)

    def test_field_of_view_calculation(self):
        """Test FOV calculation."""
        scenario = MicroscopeScenarioConfig(
            objective_spec="40x_0.9NA_air", sensor_pixel_size=6.5e-6, n_pixels=1024
        )

        # Object pixel size = sensor_pixel_size / magnification
        # FOV = object_pixel_size * n_pixels
        object_pixel_size = 6.5e-6 / 40
        expected_fov = object_pixel_size * 1024 * 1e6  # Convert to µm
        assert scenario.field_of_view_um == pytest.approx(expected_fov, rel=0.01)

    def test_invalid_na_exceeds_medium_index(self):
        """Test validation: NA cannot exceed medium index."""
        with pytest.raises(ValueError, match="NA.*cannot exceed"):
            MicroscopeScenarioConfig(objective_spec="100x_1.5NA_air")  # NA > 1 in air

    def test_undersampling_warning(self):
        """Test undersampling validation."""
        with pytest.raises(ValueError, match="undersampling"):
            # Very large pixels for high magnification = undersampling
            MicroscopeScenarioConfig(
                objective_spec="100x_1.4NA_oil",
                sensor_pixel_size=20e-6,  # Very large pixels
                n_pixels=1024,
            )

    def test_invalid_illumination_mode(self):
        """Test validation of illumination mode."""
        with pytest.raises(ValueError, match="Illumination mode"):
            MicroscopeScenarioConfig(objective_spec="40x_0.9NA_air", illumination_mode="invalid")

    def test_to_instrument_config(self):
        """Test conversion to MicroscopeConfig."""
        scenario = MicroscopeScenarioConfig(objective_spec="100x_1.4NA_oil", wavelength=550e-9)

        config = scenario.to_instrument_config()

        assert isinstance(config, MicroscopeConfig)
        assert config.wavelength == pytest.approx(550e-9)
        assert config.numerical_aperture == pytest.approx(1.4)
        assert config.magnification == 100
        assert config.medium_index == pytest.approx(1.515)

    def test_get_info(self):
        """Test get_info dictionary."""
        scenario = MicroscopeScenarioConfig(objective_spec="40x_0.9NA_air")

        info = scenario.get_info()

        assert "scenario_type" in info
        assert info["scenario_type"] == "microscope"
        assert "magnification" in info
        assert info["magnification"] == 40
        assert "lateral_resolution_nm" in info


class TestMicroscopeBuilder:
    """Test microscope builder pattern."""

    def test_basic_builder(self):
        """Test basic builder usage."""
        scenario = (
            MicroscopeBuilder()
            .objective("40x_0.9NA_air")
            .illumination("phase")
            .wavelength_nm(488)
            .build()
        )

        assert scenario.objective_spec.magnification == 40
        assert scenario.illumination_mode == "phase"
        assert scenario.wavelength == pytest.approx(488e-9)

    def test_builder_with_sensor(self):
        """Test builder with sensor parameters."""
        scenario = MicroscopeBuilder().objective("100x_1.4NA_oil").sensor_pixels(2048, 3.45).build()

        assert scenario.n_pixels == 2048
        assert scenario.sensor_pixel_size == pytest.approx(3.45e-6)

    def test_builder_with_name_description(self):
        """Test builder with name and description."""
        scenario = (
            MicroscopeBuilder()
            .objective("40x_0.9NA_air")
            .name("Custom Microscope")
            .description("Test microscope setup")
            .build()
        )

        assert scenario.name == "Custom Microscope"
        assert scenario.description == "Test microscope setup"


class TestLensSpec:
    """Test camera lens specification parsing."""

    def test_from_string_50mm(self):
        """Test parsing 50mm lens."""
        spec = LensSpec.from_string("50mm_f2.8")

        assert spec.focal_length_mm == 50
        assert spec.f_number == pytest.approx(2.8)
        assert spec.aperture_diameter_mm == pytest.approx(50 / 2.8)

    def test_from_string_35mm(self):
        """Test parsing 35mm lens."""
        spec = LensSpec.from_string("35mm_f4.0")

        assert spec.focal_length_mm == 35
        assert spec.f_number == pytest.approx(4.0)
        assert spec.aperture_diameter_mm == pytest.approx(35 / 4.0)

    def test_from_string_invalid_format(self):
        """Test error on invalid format."""
        with pytest.raises(ValueError, match="Invalid lens spec"):
            LensSpec.from_string("invalid")

    def test_string_representation(self):
        """Test string representation."""
        spec = LensSpec(focal_length_mm=50, f_number=2.8)
        assert str(spec) == "50mm_f2.8"


class TestSensorSpec:
    """Test camera sensor specification."""

    def test_full_frame(self):
        """Test full-frame sensor."""
        spec = SensorSpec.from_name("full_frame")

        assert spec.name == "full_frame"
        assert spec.width_mm == 36.0
        assert spec.height_mm == 24.0
        assert spec.pixel_pitch_um == 6.5

    def test_aps_c(self):
        """Test APS-C sensor."""
        spec = SensorSpec.from_name("aps_c")

        assert spec.name == "aps_c"
        assert spec.width_mm == 23.5
        assert spec.pixel_pitch_um == 3.9

    def test_1_inch(self):
        """Test 1-inch sensor."""
        spec = SensorSpec.from_name("1_inch")

        assert spec.name == "1_inch"
        assert spec.width_mm == 13.2

    def test_megapixels_calculation(self):
        """Test megapixels calculation."""
        spec = SensorSpec.from_name("full_frame")

        n_pixels_x = int(spec.width_mm * 1000 / spec.pixel_pitch_um)
        n_pixels_y = int(spec.height_mm * 1000 / spec.pixel_pitch_um)
        expected_mp = (n_pixels_x * n_pixels_y) / 1e6

        assert spec.megapixels == pytest.approx(expected_mp, rel=0.01)

    def test_invalid_sensor_name(self):
        """Test error on invalid sensor name."""
        with pytest.raises(ValueError, match="Unknown sensor type"):
            SensorSpec.from_name("invalid_sensor")


class TestDroneScenario:
    """Test drone camera scenario configuration."""

    def test_basic_configuration(self):
        """Test basic drone configuration."""
        scenario = DroneScenarioConfig(lens_spec="50mm_f4.0", altitude_m=50.0)

        assert scenario.scenario_type == "drone"
        assert scenario.lens_spec.focal_length_mm == 50
        assert scenario.altitude_m == 50.0

    def test_gsd_calculation(self):
        """Test ground sampling distance calculation."""
        scenario = DroneScenarioConfig(
            lens_spec="50mm_f4.0",
            sensor_spec="full_frame",  # 36mm sensor, 6.5µm pixels
            altitude_m=50.0,
        )

        # GSD = altitude * pixel_pitch / focal_length
        # GSD = 50m * 6.5e-6m / 0.05m = 6.5e-3 m = 0.65 cm
        expected_gsd = (50.0 * 6.5e-6) / (50e-3) * 100  # Convert to cm
        assert scenario.actual_gsd_cm == pytest.approx(expected_gsd, rel=0.05)

    def test_swath_width_calculation(self):
        """Test swath width calculation."""
        scenario = DroneScenarioConfig(
            lens_spec="50mm_f4.0", sensor_spec="full_frame", altitude_m=100.0
        )

        # Swath = altitude * sensor_width / focal_length
        # Swath = 100m * 0.036m / 0.05m = 72m
        expected_swath = (100.0 * 36e-3) / (50e-3)
        assert scenario.swath_width_m == pytest.approx(expected_swath, rel=0.01)

    def test_fresnel_number_calculation(self):
        """Test Fresnel number calculation."""
        scenario = DroneScenarioConfig(lens_spec="50mm_f4.0", altitude_m=50.0)

        # F = a² / (λ * z)
        # a = aperture_radius = (50mm / 4.0) / 2 = 6.25mm
        a = (50e-3 / 4.0) / 2
        wavelength = 550e-9
        z = 50.0
        expected_fresnel = (a**2) / (wavelength * z)

        assert scenario.fresnel_number == pytest.approx(expected_fresnel, rel=0.01)

    def test_motion_blur_calculation(self):
        """Test motion blur calculation."""
        scenario = DroneScenarioConfig(
            lens_spec="50mm_f4.0",
            sensor_spec="full_frame",
            altitude_m=50.0,
            ground_speed_mps=10.0,
        )

        # Motion blur should be non-zero
        assert scenario.motion_blur_pixels > 0

    def test_hover_no_motion_blur(self):
        """Test hover mode has no motion blur."""
        scenario = DroneScenarioConfig(
            lens_spec="50mm_f4.0",
            sensor_spec="full_frame",
            altitude_m=50.0,
            ground_speed_mps=0.0,  # Hover
        )

        assert scenario.motion_blur_pixels == 0.0

    def test_invalid_altitude_negative(self):
        """Test validation: negative altitude."""
        with pytest.raises(ValueError, match="Altitude must be positive"):
            DroneScenarioConfig(lens_spec="50mm_f4.0", altitude_m=-10.0)

    def test_invalid_altitude_too_low(self):
        """Test validation: altitude too low."""
        with pytest.raises(ValueError, match="too low"):
            DroneScenarioConfig(lens_spec="50mm_f4.0", altitude_m=2.0)

    def test_invalid_altitude_too_high(self):
        """Test validation: altitude too high."""
        with pytest.raises(ValueError, match="exceeds typical drone limit"):
            DroneScenarioConfig(lens_spec="50mm_f4.0", altitude_m=600.0)

    def test_invalid_ground_speed_negative(self):
        """Test validation: negative ground speed."""
        with pytest.raises(ValueError, match="Ground speed must be non-negative"):
            DroneScenarioConfig(lens_spec="50mm_f4.0", altitude_m=50.0, ground_speed_mps=-5.0)

    def test_to_instrument_config(self):
        """Test conversion to CameraConfig."""
        scenario = DroneScenarioConfig(
            lens_spec="50mm_f4.0", sensor_spec="full_frame", altitude_m=50.0
        )

        config = scenario.to_instrument_config()

        assert isinstance(config, CameraConfig)
        assert config.focal_length == pytest.approx(50e-3)
        assert config.f_number == pytest.approx(4.0)
        assert config.object_distance == 50.0

    def test_get_info(self):
        """Test get_info dictionary."""
        scenario = DroneScenarioConfig(lens_spec="50mm_f4.0", altitude_m=50.0)

        info = scenario.get_info()

        assert "scenario_type" in info
        assert info["scenario_type"] == "drone"
        assert "altitude_m" in info
        assert info["altitude_m"] == 50.0
        assert "gsd_cm" in info


class TestDroneBuilder:
    """Test drone builder pattern."""

    def test_basic_builder(self):
        """Test basic builder usage."""
        scenario = (
            DroneBuilder().lens("50mm_f2.8").sensor("aps_c").altitude(100).ground_speed(10).build()
        )

        assert scenario.lens_spec.focal_length_mm == 50
        assert scenario.sensor_spec.name == "aps_c"
        assert scenario.altitude_m == 100
        assert scenario.ground_speed_mps == 10

    def test_builder_with_wavelength(self):
        """Test builder with wavelength."""
        scenario = DroneBuilder().lens("35mm_f4.0").wavelength_nm(488).build()

        assert scenario.wavelength == pytest.approx(488e-9)

    def test_builder_with_name_description(self):
        """Test builder with name and description."""
        scenario = (
            DroneBuilder()
            .lens("50mm_f4.0")
            .name("Custom Drone")
            .description("Test drone setup")
            .build()
        )

        assert scenario.name == "Custom Drone"
        assert scenario.description == "Test drone setup"


class TestPresets:
    """Test preset system."""

    def test_list_all_presets(self):
        """Test listing all presets."""
        all_presets = list_scenario_presets()

        assert len(all_presets) > 0
        assert "microscope_100x_oil" in all_presets
        assert "drone_50m_survey" in all_presets

    def test_list_microscope_presets(self):
        """Test listing microscope presets only."""
        microscope_presets = list_scenario_presets("microscope")

        assert len(microscope_presets) > 0
        assert all("microscope" in name for name in microscope_presets)
        assert "drone_50m_survey" not in microscope_presets

    def test_list_drone_presets(self):
        """Test listing drone presets only."""
        drone_presets = list_scenario_presets("drone")

        assert len(drone_presets) > 0
        assert all("drone" in name for name in drone_presets)
        assert "microscope_100x_oil" not in drone_presets

    def test_get_microscope_preset(self):
        """Test loading microscope preset."""
        scenario = get_scenario_preset("microscope_100x_oil")

        assert isinstance(scenario, MicroscopeScenarioConfig)
        assert scenario.objective_spec.magnification == 100
        assert scenario.objective_spec.numerical_aperture == pytest.approx(1.4)
        assert scenario.objective_spec.immersion_medium == "oil"

    def test_get_drone_preset(self):
        """Test loading drone preset."""
        scenario = get_scenario_preset("drone_50m_survey")

        assert isinstance(scenario, DroneScenarioConfig)
        assert scenario.altitude_m == 50.0

    def test_preset_has_description(self):
        """Test preset has description."""
        scenario = get_scenario_preset("microscope_100x_oil")

        assert scenario.description != ""
        assert len(scenario.description) > 10

    def test_invalid_preset_name(self):
        """Test error on invalid preset name."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_scenario_preset("nonexistent_preset")

    def test_get_preset_description(self):
        """Test getting preset description."""
        desc = get_preset_description("microscope_100x_oil")

        assert desc != ""
        assert len(desc) > 10
        assert "resolution" in desc.lower() or "cellular" in desc.lower()

    def test_get_preset_description_not_found(self):
        """Test getting description for non-existent preset."""
        desc = get_preset_description("nonexistent_preset")

        assert desc == ""


class TestIntegration:
    """Integration tests for scenario system."""

    def test_microscope_scenario_to_instrument_works(self):
        """Test microscope scenario converts to working instrument config."""
        scenario = get_scenario_preset("microscope_40x_air")

        # Convert to instrument config
        config = scenario.to_instrument_config()

        # Verify config is valid
        assert isinstance(config, MicroscopeConfig)
        config.validate()

    def test_drone_scenario_to_instrument_works(self):
        """Test drone scenario converts to working instrument config."""
        scenario = get_scenario_preset("drone_50m_survey")

        # Convert to instrument config
        config = scenario.to_instrument_config()

        # Verify config is valid
        assert isinstance(config, CameraConfig)
        config.validate()

    def test_all_microscope_presets_valid(self):
        """Test all microscope presets are valid."""
        microscope_presets = list_scenario_presets("microscope")

        for preset_name in microscope_presets:
            scenario = get_scenario_preset(preset_name)
            scenario.validate()
            config = scenario.to_instrument_config()
            config.validate()

    def test_all_drone_presets_valid(self):
        """Test all drone presets are valid."""
        drone_presets = list_scenario_presets("drone")

        for preset_name in drone_presets:
            scenario = get_scenario_preset(preset_name)
            scenario.validate()
            config = scenario.to_instrument_config()
            config.validate()
