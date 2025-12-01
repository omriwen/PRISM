"""
Custom Scenario Design with Builder Pattern
============================================

Demonstrates how to create custom optical configurations for specialized
imaging applications beyond the standard presets.

This example covers:
1. MicroscopeBuilder for custom objectives
2. DroneBuilder for custom missions
3. Validation and parameter exploration
4. Integration with SPIDS instruments

Example:
    $ uv run python examples/python_api/08_custom_scenario_builder.py

    # Quick mode (skip instrument creation)
    $ uv run python examples/python_api/08_custom_scenario_builder.py --quick

    # List available sensor and lens options
    $ uv run python examples/python_api/08_custom_scenario_builder.py --list-options

Expected Runtime: < 1 minute

References:
    - Builder API: spids.scenarios
    - Presets catalog: docs/references/scenario_preset_catalog.md
    - Resolution theory: docs/references/resolution_limits_theory.md
"""

from __future__ import annotations

import argparse
import sys

from loguru import logger

from prism.core.instruments import create_instrument
from prism.scenarios import (
    DroneBuilder,
    DroneScenarioConfig,
    LensSpec,
    MicroscopeBuilder,
    MicroscopeScenarioConfig,
    ObjectiveSpec,
    SensorSpec,
    get_scenario_preset,
    list_scenario_presets,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Custom Scenario Design with Builder Pattern",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (skip instrument creation)",
    )
    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List available sensor and lens options",
    )
    return parser.parse_args()


def setup_logging() -> None:
    """Configure loguru logging."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level="INFO",
    )


def list_options() -> None:
    """List available sensor and lens options."""
    logger.info("=" * 70)
    logger.info("Available Sensor Types")
    logger.info("=" * 70)

    sensors = ["full_frame", "aps_c", "micro_four_thirds", "1_inch", "1_2.3_inch"]
    for sensor_name in sensors:
        sensor = SensorSpec.from_name(sensor_name)
        logger.info(
            f"  {sensor_name:<20} "
            f"{sensor.width_mm:.1f}x{sensor.height_mm:.1f}mm  "
            f"{sensor.pixel_pitch_um:.1f}µm  "
            f"{sensor.megapixels:.1f}MP"
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info("Example Lens Specifications")
    logger.info("=" * 70)
    logger.info("  Format: <focal_length>mm_f<f_number>")
    logger.info("  Examples:")
    logger.info("    35mm_f2.8   - Wide angle, fast aperture")
    logger.info("    50mm_f4.0   - Standard, moderate aperture")
    logger.info("    85mm_f1.8   - Portrait/telephoto, fast aperture")
    logger.info("    200mm_f5.6  - Telephoto, smaller aperture")

    logger.info("")
    logger.info("=" * 70)
    logger.info("Available Microscope Presets (for reference)")
    logger.info("=" * 70)
    microscope_presets = list_scenario_presets("microscope")
    for preset in sorted(microscope_presets):
        scenario = get_scenario_preset(preset)
        logger.info(
            f"  {preset:<30} "
            f"Res={scenario.lateral_resolution_nm:>5.0f}nm  "
            f"FOV={scenario.field_of_view_um:>6.1f}µm"
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info("Available Drone Presets (for reference)")
    logger.info("=" * 70)
    drone_presets = list_scenario_presets("drone")
    for preset in sorted(drone_presets):
        scenario = get_scenario_preset(preset)
        logger.info(
            f"  {preset:<30} GSD={scenario.actual_gsd_cm:>5.1f}cm  Alt={scenario.altitude_m:>4.0f}m"
        )


def demo_microscope_builder(quick: bool = False) -> None:
    """Demonstrate custom microscope scenario creation.

    Example 1: Custom Spinning Disk Confocal
    - Non-standard objective (75x with silicone immersion)
    - Phase contrast illumination
    - Red wavelength (640nm) for specific fluorophore
    - High-resolution camera (4096 pixels, 3.45µm pitch)
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Example 1: Custom Microscope - Spinning Disk Confocal")
    logger.info("=" * 70)

    # Build custom microscope configuration
    custom_microscope = (
        MicroscopeBuilder()
        .objective("75x_1.3NA_oil")  # Custom 75x silicone immersion objective
        .illumination("phase")  # Phase contrast mode
        .wavelength_nm(640)  # Red illumination (e.g., for mCherry)
        .sensor_pixels(2048, 3.45)  # High-res sCMOS camera
        .name("Custom Spinning Disk Confocal")
        .description("75x silicone objective for deep tissue imaging")
        .build()
    )

    # Display computed parameters
    logger.info(f"Scenario: {custom_microscope.name}")
    logger.info(f"Objective: {custom_microscope.objective_spec}")
    logger.info(f"Illumination: {custom_microscope.illumination_mode}")
    logger.info(f"Wavelength: {custom_microscope.wavelength * 1e9:.0f} nm")
    logger.info("")
    logger.info("Computed Parameters:")
    logger.info(f"  Lateral Resolution: {custom_microscope.lateral_resolution_nm:.1f} nm")
    logger.info(f"  Axial Resolution: {custom_microscope.axial_resolution_um:.2f} µm")
    logger.info(f"  Field of View: {custom_microscope.field_of_view_um:.1f} µm")

    if not quick:
        # Convert to SPIDS instrument and verify
        logger.info("")
        logger.info("Converting to SPIDS instrument...")
        instrument_config = custom_microscope.to_instrument_config()
        microscope = create_instrument(instrument_config)
        logger.success(f"Instrument created: {type(microscope).__name__}")
        logger.info(f"  Wavelength: {instrument_config.wavelength * 1e9:.0f} nm")
        logger.info(f"  NA: {instrument_config.numerical_aperture:.2f}")
        logger.info(f"  Magnification: {instrument_config.magnification:.0f}x")


def demo_microscope_comparison() -> None:
    """Compare custom microscope with standard preset.

    Shows how custom configurations relate to standard presets.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Example 2: Custom vs Standard Microscope Comparison")
    logger.info("=" * 70)

    # Standard 100x oil preset
    standard = get_scenario_preset("microscope_100x_oil")

    # Custom 100x with blue light for DAPI
    custom = (
        MicroscopeBuilder()
        .objective("100x_1.4NA_oil")
        .illumination("brightfield")
        .wavelength_nm(405)  # UV/blue for DAPI
        .sensor_pixels(2048, 6.5)  # Standard sCMOS
        .name("Custom 100x DAPI")
        .build()
    )

    logger.info(f"{'Parameter':<25} {'Standard (550nm)':<20} {'Custom (405nm)':<20}")
    logger.info("-" * 65)
    logger.info(
        f"{'Wavelength':<25} {standard.wavelength * 1e9:<20.0f} {custom.wavelength * 1e9:<20.0f}"
    )
    logger.info(
        f"{'Lateral Resolution (nm)':<25} "
        f"{standard.lateral_resolution_nm:<20.1f} "
        f"{custom.lateral_resolution_nm:<20.1f}"
    )
    logger.info(
        f"{'Axial Resolution (µm)':<25} "
        f"{standard.axial_resolution_um:<20.2f} "
        f"{custom.axial_resolution_um:<20.2f}"
    )
    logger.info(
        f"{'Field of View (µm)':<25} "
        f"{standard.field_of_view_um:<20.1f} "
        f"{custom.field_of_view_um:<20.1f}"
    )

    improvement = (
        (standard.lateral_resolution_nm - custom.lateral_resolution_nm)
        / standard.lateral_resolution_nm
        * 100
    )
    logger.info("")
    logger.success(f"Resolution improvement with 405nm: {improvement:.1f}%")


def demo_drone_builder(quick: bool = False) -> None:
    """Demonstrate custom drone scenario creation.

    Example 3: High-altitude corridor inspection
    - Telephoto lens (85mm f/1.8) for detail at altitude
    - Micro Four Thirds sensor
    - High altitude (150m) for utility corridor mapping
    - Moderate flight speed (15 m/s)
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Example 3: Custom Drone - High-altitude Corridor Inspection")
    logger.info("=" * 70)

    # Build custom drone configuration
    custom_drone = (
        DroneBuilder()
        .lens("85mm_f1.8")  # Telephoto lens
        .sensor("micro_four_thirds")  # Micro Four Thirds sensor
        .altitude(150)  # High altitude for corridor mapping
        .ground_speed(15)  # Moderate flight speed
        .name("High-altitude Corridor Inspection")
        .description("Telephoto configuration for power line inspection")
        .build()
    )

    # Display computed parameters
    logger.info(f"Scenario: {custom_drone.name}")
    logger.info(f"Lens: {custom_drone._lens}")
    logger.info(f"Sensor: {custom_drone._sensor.name} ({custom_drone._sensor.megapixels:.1f} MP)")
    logger.info(f"Altitude: {custom_drone.altitude_m:.0f} m")
    logger.info(f"Ground Speed: {custom_drone.ground_speed_mps:.1f} m/s")
    logger.info("")
    logger.info("Computed Parameters:")
    logger.info(f"  Ground Sampling Distance: {custom_drone.actual_gsd_cm:.2f} cm")
    logger.info(f"  Swath Width: {custom_drone.swath_width_m:.1f} m")
    logger.info(f"  Fresnel Number: {custom_drone.fresnel_number:.1f}")
    logger.info(f"  Motion Blur: {custom_drone.motion_blur_pixels:.2f} pixels")

    if not quick:
        # Convert to SPIDS instrument and verify
        logger.info("")
        logger.info("Converting to SPIDS instrument...")
        instrument_config = custom_drone.to_instrument_config()
        camera = create_instrument(instrument_config)
        logger.success(f"Instrument created: {type(camera).__name__}")
        logger.info(f"  Focal Length: {instrument_config.focal_length * 1e3:.0f} mm")
        logger.info(f"  f-number: f/{instrument_config.f_number:.1f}")
        logger.info(f"  Object Distance: {instrument_config.object_distance:.0f} m")


def demo_drone_mission_planning() -> None:
    """Demonstrate drone mission planning with builder.

    Shows how to plan missions for different altitude/GSD requirements.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Example 4: Drone Mission Planning - GSD vs Coverage Tradeoff")
    logger.info("=" * 70)

    # Common sensor and lens for comparison
    sensor = "1_inch"  # DJI-style sensor
    lens = "24mm_f2.8"  # Standard wide lens

    altitudes = [30, 50, 80, 100, 120]

    logger.info(f"Sensor: {sensor}, Lens: {lens}")
    logger.info("")
    logger.info(f"{'Altitude (m)':<15} {'GSD (cm)':<12} {'Swath (m)':<12} {'Fresnel #':<12}")
    logger.info("-" * 55)

    for alt in altitudes:
        drone = DroneBuilder().lens(lens).sensor(sensor).altitude(alt).ground_speed(10).build()
        logger.info(
            f"{alt:<15} "
            f"{drone.actual_gsd_cm:<12.2f} "
            f"{drone.swath_width_m:<12.1f} "
            f"{drone.fresnel_number:<12.1f}"
        )

    logger.info("")
    logger.info("Interpretation:")
    logger.info("  - Lower altitude = finer GSD (more detail)")
    logger.info("  - Higher altitude = wider swath (more coverage)")
    logger.info("  - Fresnel # > 10 indicates near-field (geometric optics)")


def demo_direct_configuration() -> None:
    """Demonstrate direct configuration without builder.

    For advanced users who prefer explicit configuration.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Example 5: Direct Configuration (Advanced)")
    logger.info("=" * 70)

    # Direct microscope configuration
    microscope = MicroscopeScenarioConfig(
        objective_spec="60x_1.2NA_water",
        illumination_mode="dic",
        wavelength=488e-9,  # Blue laser
        sensor_pixel_size=6.5e-6,
        n_pixels=2048,
    )

    logger.info("Direct MicroscopeScenarioConfig:")
    logger.info(f"  Objective: {microscope.objective_spec}")
    logger.info(f"  Resolution: {microscope.lateral_resolution_nm:.1f} nm")
    logger.info(f"  FOV: {microscope.field_of_view_um:.1f} µm")

    # Direct drone configuration
    drone = DroneScenarioConfig(
        lens_spec="35mm_f2.8",
        sensor_spec="aps_c",
        altitude_m=75,
        ground_speed_mps=12,
    )

    logger.info("")
    logger.info("Direct DroneScenarioConfig:")
    logger.info(f"  Lens: {drone._lens}")
    logger.info(f"  GSD: {drone.actual_gsd_cm:.2f} cm")
    logger.info(f"  Swath: {drone.swath_width_m:.1f} m")


def demo_custom_specs() -> None:
    """Demonstrate using custom ObjectiveSpec and LensSpec directly.

    For creating non-standard configurations.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("Example 6: Custom Specs (ObjectiveSpec, LensSpec)")
    logger.info("=" * 70)

    # Create custom objective spec
    custom_objective = ObjectiveSpec(
        magnification=150,  # Very high magnification
        numerical_aperture=1.45,  # High NA TIRF objective
        immersion_medium="oil",
    )

    microscope = (
        MicroscopeBuilder()
        .objective(custom_objective)  # Pass ObjectiveSpec directly
        .illumination("brightfield")
        .wavelength_nm(561)  # Green/yellow for GFP
        .sensor_pixels(4096, 3.45)  # High-res camera
        .name("TIRF 150x Custom")
        .build()
    )

    logger.info("Custom 150x TIRF Objective:")
    logger.info(f"  Magnification: {custom_objective.magnification}x")
    logger.info(f"  NA: {custom_objective.numerical_aperture}")
    logger.info(f"  Resolution: {microscope.lateral_resolution_nm:.1f} nm")
    logger.info(f"  FOV: {microscope.field_of_view_um:.1f} µm")

    # Create custom lens spec
    custom_lens = LensSpec(
        focal_length_mm=135,  # Portrait telephoto
        f_number=2.0,  # Fast aperture
    )

    drone = (
        DroneBuilder()
        .lens(custom_lens)  # Pass LensSpec directly
        .sensor("full_frame")
        .altitude(200)
        .ground_speed(5)  # Slow for sharp images
        .name("Telephoto Survey")
        .build()
    )

    logger.info("")
    logger.info("Custom 135mm f/2 Telephoto Drone:")
    logger.info(f"  Focal Length: {custom_lens.focal_length_mm} mm")
    logger.info(f"  Aperture: f/{custom_lens.f_number}")
    logger.info(f"  Aperture Diameter: {custom_lens.aperture_diameter_mm:.1f} mm")
    logger.info(f"  GSD at 200m: {drone.actual_gsd_cm:.2f} cm")


def main() -> int:
    """Main entry point for custom scenario builder demonstration.

    Returns
    -------
    int
        Exit code (0 for success)
    """
    setup_logging()
    args = parse_args()

    if args.list_options:
        list_options()
        return 0

    logger.info("=" * 70)
    logger.info("SPIDS Custom Scenario Builder Examples")
    logger.info("=" * 70)

    # Run all demonstrations
    demo_microscope_builder(quick=args.quick)
    demo_microscope_comparison()
    demo_drone_builder(quick=args.quick)
    demo_drone_mission_planning()
    demo_direct_configuration()
    demo_custom_specs()

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.success("Custom Scenario Builder Examples Complete!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Key takeaways:")
    logger.info("  1. Use MicroscopeBuilder/DroneBuilder for fluent configuration")
    logger.info("  2. Builders automatically validate parameters")
    logger.info("  3. Custom ObjectiveSpec/LensSpec for non-standard setups")
    logger.info("  4. Call to_instrument_config() to integrate with SPIDS")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  - Try --list-options to see available sensors/lenses")
    logger.info("  - Create your own custom configuration")
    logger.info("  - Use with 06_microscope_reconstruction.py or 07_drone_mapping.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
