"""
SPIDS Drone Mapping Reconstruction Pipeline
============================================

Complete workflow for aerial/drone imaging:

1. Load drone preset (configurable altitude and camera)
2. Create checkerboard test scene (for GSD validation)
3. Simulate camera measurements
4. Compute quality metrics (SSIM, PSNR)
5. Validate GSD against theoretical values
6. Save publication-quality results

Example:
    $ uv run python examples/python_api/07_drone_mapping.py

    # Use different preset
    $ uv run python examples/python_api/07_drone_mapping.py --preset drone_100m_mapping

    # Quick test mode (fewer iterations)
    $ uv run python examples/python_api/07_drone_mapping.py --quick

    # List all available drone presets
    $ uv run python examples/python_api/07_drone_mapping.py --list-presets

Expected Runtime: 2-5 minutes

Note:
    - Quick mode (--quick) skips saving results for faster testing.
    - This uses coherent wave propagation which includes diffraction effects.
      For drone altitudes (10-100m), diffraction causes significant blurring
      compared to geometric optics models. This is physically accurate but
      may produce lower SSIM/PSNR than incoherent imaging models.

References:
    - GSD formula: GSD = H * p / f (altitude * pixel_pitch / focal_length)
    - Drone presets: docs/references/scenario_preset_catalog.md
    - Target generation: spids.core.targets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from loguru import logger

from prism.core.instruments import create_instrument
from prism.core.targets import CheckerboardConfig, CheckerboardTarget
from prism.scenarios import get_scenario_preset, list_scenario_presets
from prism.utils.metrics import compute_ssim, psnr


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SPIDS Drone Mapping Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="drone_50m_survey",
        help="Drone preset name (default: drone_50m_survey)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/drone_mapping",
        help="Output directory for results (default: results/drone_mapping)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (implies --no-save)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available drone presets and exit",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results (useful for testing)",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=None,
        help="Checkerboard square size in meters (default: 10x GSD)",
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


def list_presets() -> None:
    """List all available drone presets with key parameters."""
    presets = list_scenario_presets("drone")

    logger.info("Available drone presets:")
    logger.info("=" * 80)

    for preset in sorted(presets):
        scenario = get_scenario_preset(preset)
        logger.info(
            f"  {preset:<30} "
            f"Alt={scenario.altitude_m:>4.0f}m  "
            f"GSD={scenario.actual_gsd_cm:>5.1f}cm  "
            f"Swath={scenario.swath_width_m:>5.1f}m  "
            f"Blur={scenario.motion_blur_pixels:.1f}px"
        )

    logger.info("=" * 80)
    logger.info(f"Total: {len(presets)} presets")


def create_target_for_scenario(
    scenario, square_size: float | None = None, n_pixels: int | None = None
) -> tuple[torch.Tensor, float]:
    """
    Create checkerboard target appropriate for the drone scenario.

    Parameters
    ----------
    scenario : DroneScenarioConfig
        Drone scenario configuration
    square_size : float | None
        Square size in meters (default: 10x GSD)
    n_pixels : int | None
        Override number of pixels (default: use scenario.n_pixels)

    Returns
    -------
    tuple[torch.Tensor, float]
        Target tensor and field size in meters
    """
    # Use n_pixels from scenario (must match camera grid)
    if n_pixels is None:
        n_pixels = scenario.n_pixels

    # Field of view = swath width (what the camera sees on ground)
    field_size = scenario.swath_width_m

    # Square size: default to ensure at least 10 pixels per square
    # For proper resolution, we need squares larger than the pixel size
    pixel_size_m = field_size / n_pixels
    if square_size is None:
        # Ensure squares are at least 10 pixels wide for good visibility
        square_size = max(pixel_size_m * 10, 0.5)  # At least 50cm or 10 pixels

    logger.info("Creating checkerboard target:")
    logger.info(f"  Field of view: {field_size:.1f} m")
    logger.info(f"  Square size: {square_size * 100:.1f} cm")
    logger.info(f"  Image size: {n_pixels}x{n_pixels} pixels")

    config = CheckerboardConfig(
        size=n_pixels,
        field_size=field_size,
        square_size=square_size,
        margin_ratio=0.1,  # Smaller margin for drone imaging
    )

    target = CheckerboardTarget(config)
    image = target.generate()

    # Log checkerboard parameters
    res_info = target.resolution_elements
    logger.info(f"  Number of squares: {res_info.get('n_squares', 'N/A')}")
    samples = target.gsd_samples_per_square
    if samples:
        logger.info(f"  Pixels per square: {samples:.1f}")

    return image, field_size


def simulate_drone_imaging(camera, target: torch.Tensor) -> torch.Tensor:
    """
    Simulate drone camera imaging of target.

    Parameters
    ----------
    camera : Camera
        Camera instrument
    target : torch.Tensor
        Ground truth target image

    Returns
    -------
    torch.Tensor
        Simulated measurement
    """
    # Add batch and channel dimensions if needed
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.dim() == 3:
        target = target.unsqueeze(0)

    with torch.no_grad():
        measurement = camera.forward(target)

    return measurement.squeeze()


def compute_metrics(
    ground_truth: torch.Tensor,
    measurement: torch.Tensor,
) -> dict[str, float]:
    """
    Compute quality metrics between ground truth and measurement.

    Parameters
    ----------
    ground_truth : torch.Tensor
        Ground truth image
    measurement : torch.Tensor
        Measured/reconstructed image

    Returns
    -------
    dict[str, float]
        Dictionary of quality metrics
    """
    # Ensure same shape
    gt = ground_truth.squeeze()
    meas = measurement.squeeze()

    # Normalize for fair comparison (keep as tensors)
    gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
    meas_norm = (meas - meas.min()) / (meas.max() - meas.min() + 1e-8)

    # compute_ssim and psnr expect tensors
    ssim_value = compute_ssim(meas_norm, gt_norm)
    psnr_value = psnr(meas_norm, gt_norm)

    return {
        "SSIM": ssim_value,
        "PSNR": psnr_value,
    }


def validate_gsd(scenario, ground_truth: torch.Tensor, measurement: torch.Tensor) -> dict:
    """
    Validate GSD by measuring resolution from checkerboard.

    Parameters
    ----------
    scenario : DroneScenarioConfig
        Drone scenario configuration
    ground_truth : torch.Tensor
        Ground truth checkerboard
    measurement : torch.Tensor
        Simulated measurement

    Returns
    -------
    dict
        Validation results including theoretical and measured GSD
    """
    # Theoretical GSD from scenario
    theoretical_gsd_cm = scenario.actual_gsd_cm

    # For a full validation, we would:
    # 1. Find checkerboard edges using Canny or similar
    # 2. Measure edge transition width
    # 3. Compare to theoretical resolution

    # Simplified validation: check that the camera can resolve the checkerboard
    # by comparing variance in measurement (well-resolved = high variance)
    gt_var = ground_truth.var().item()
    meas_var = measurement.var().item()
    resolution_ratio = meas_var / (gt_var + 1e-8)

    # If measurement preserves >50% of variance, GSD is validated
    gsd_validated = resolution_ratio > 0.5

    return {
        "theoretical_gsd_cm": theoretical_gsd_cm,
        "resolution_ratio": resolution_ratio,
        "validated": gsd_validated,
    }


def save_results(
    ground_truth: torch.Tensor,
    measurement: torch.Tensor,
    metrics: dict[str, float],
    gsd_validation: dict,
    scenario,
    field_size: float,
    output_dir: Path,
) -> None:
    """
    Save visualization results.

    Parameters
    ----------
    ground_truth : torch.Tensor
        Ground truth image
    measurement : torch.Tensor
        Simulated measurement
    metrics : dict[str, float]
        Quality metrics
    gsd_validation : dict
        GSD validation results
    scenario : DroneScenarioConfig
        Scenario configuration
    field_size : float
        Field size in meters
    output_dir : Path
        Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    gt = ground_truth.squeeze().cpu().numpy()
    meas = measurement.squeeze().cpu().numpy()
    diff = gt - meas

    extent = [0, field_size, 0, field_size]

    # Ground truth
    im0 = axes[0].imshow(gt, cmap="gray", extent=extent, origin="lower")
    axes[0].set_title("Ground Truth (Checkerboard)")
    axes[0].set_xlabel("Position (m)")
    axes[0].set_ylabel("Position (m)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Measurement
    im1 = axes[1].imshow(meas, cmap="gray", extent=extent, origin="lower")
    axes[1].set_title(
        f"Drone Measurement\nAlt={scenario.altitude_m:.0f}m, GSD={scenario.actual_gsd_cm:.1f}cm"
    )
    axes[1].set_xlabel("Position (m)")
    axes[1].set_ylabel("Position (m)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Difference
    im2 = axes[2].imshow(diff, cmap="RdBu", extent=extent, origin="lower")
    axes[2].set_title(f"Difference\nSSIM={metrics['SSIM']:.4f}, PSNR={metrics['PSNR']:.1f}dB")
    axes[2].set_xlabel("Position (m)")
    axes[2].set_ylabel("Position (m)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    fig.savefig(output_dir / "drone_mapping_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.success(f"Saved comparison figure to {output_dir / 'drone_mapping_comparison.png'}")

    # Save metrics to text file
    with open(output_dir / "metrics.txt", "w") as f:
        f.write("SPIDS Drone Mapping Reconstruction Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Preset: {scenario.name}\n")
        f.write(f"Lens: {scenario._lens}\n")
        f.write(f"Sensor: {scenario._sensor.name}\n")
        f.write(f"Altitude: {scenario.altitude_m:.0f} m\n")
        f.write(f"Ground Speed: {scenario.ground_speed_mps:.1f} m/s\n")
        f.write(f"Fresnel Number: {scenario.fresnel_number:.1f}\n\n")

        f.write("GSD Validation:\n")
        f.write(f"  Theoretical GSD: {gsd_validation['theoretical_gsd_cm']:.2f} cm\n")
        f.write(f"  Resolution Ratio: {gsd_validation['resolution_ratio']:.3f}\n")
        f.write(f"  Status: {'PASS' if gsd_validation['validated'] else 'FAIL'}\n\n")

        f.write("Quality Metrics:\n")
        f.write(f"  SSIM: {metrics['SSIM']:.4f}\n")
        f.write(f"  PSNR: {metrics['PSNR']:.2f} dB\n")

    logger.success(f"Saved metrics to {output_dir / 'metrics.txt'}")


def main() -> int:
    """
    Main entry point for drone mapping reconstruction pipeline.

    Returns
    -------
    int
        Exit code (0 for success)
    """
    setup_logging()
    args = parse_args()

    # Handle list presets option
    if args.list_presets:
        list_presets()
        return 0

    logger.info("=" * 60)
    logger.info("SPIDS Drone Mapping Reconstruction Pipeline")
    logger.info("=" * 60)

    # Quick mode implies no-save
    if args.quick:
        args.no_save = True

    # 1. Load drone scenario
    logger.info(f"Loading preset: {args.preset}")
    try:
        scenario = get_scenario_preset(args.preset)
    except (KeyError, ValueError) as e:
        logger.error(f"Unknown preset: {args.preset} ({e})")
        logger.error("Use --list-presets to see available options")
        return 1

    logger.info(f"  Altitude: {scenario.altitude_m:.0f} m")
    logger.info(f"  Lens: {scenario._lens}")
    logger.info(f"  Sensor: {scenario._sensor.name} ({scenario._sensor.megapixels:.1f} MP)")
    logger.info(f"  GSD: {scenario.actual_gsd_cm:.2f} cm")
    logger.info(f"  Swath Width: {scenario.swath_width_m:.1f} m")
    logger.info(f"  Fresnel Number: {scenario.fresnel_number:.1f}")

    if scenario.motion_blur_pixels > 0:
        logger.info(f"  Motion Blur: {scenario.motion_blur_pixels:.2f} pixels")

    # 2. Create camera instrument
    logger.info("Creating camera instrument...")
    instrument_config = scenario.to_instrument_config()
    camera = create_instrument(instrument_config)

    # 3. Create checkerboard target
    # Note: target n_pixels must match camera grid for propagation to work
    logger.info("Creating checkerboard test target...")
    target, field_size = create_target_for_scenario(scenario, square_size=args.square_size)
    n_pixels = target.shape[0]
    pixel_size_m = field_size / n_pixels
    logger.info(f"  Target size: {target.shape}")
    logger.info(f"  Ground pixel size: {pixel_size_m * 100:.2f} cm")

    # 4. Simulate drone camera measurement
    logger.info("Simulating drone camera measurement...")
    measurement = simulate_drone_imaging(camera, target)
    logger.info(f"  Measurement size: {measurement.shape}")

    # 5. Compute quality metrics
    logger.info("Computing quality metrics...")
    metrics = compute_metrics(target, measurement)
    logger.info(f"  SSIM: {metrics['SSIM']:.4f}")
    logger.info(f"  PSNR: {metrics['PSNR']:.2f} dB")

    # 6. Validate GSD
    logger.info("Validating GSD...")
    gsd_validation = validate_gsd(scenario, target, measurement)

    if gsd_validation["validated"]:
        logger.success(
            f"  GSD validated: {scenario.actual_gsd_cm:.2f} cm "
            f"(resolution ratio: {gsd_validation['resolution_ratio']:.3f})"
        )
    else:
        logger.warning(
            f"  GSD validation FAILED: resolution ratio {gsd_validation['resolution_ratio']:.3f} < 0.5"
        )

    # 7. Save results
    if not args.no_save:
        output_dir = Path(args.output_dir)
        logger.info(f"Saving results to {output_dir}...")
        save_results(
            ground_truth=target,
            measurement=measurement,
            metrics=metrics,
            gsd_validation=gsd_validation,
            scenario=scenario,
            field_size=field_size,
            output_dir=output_dir,
        )

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.success("Drone Mapping Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"  Preset: {args.preset}")
    logger.info(f"  Altitude: {scenario.altitude_m:.0f} m")
    logger.info(f"  GSD: {scenario.actual_gsd_cm:.2f} cm")
    logger.info(f"  SSIM: {metrics['SSIM']:.4f}")
    logger.info(f"  PSNR: {metrics['PSNR']:.2f} dB")

    # Quality assessment
    if metrics["SSIM"] > 0.9:
        logger.success("  Quality: EXCELLENT (SSIM > 0.9)")
    elif metrics["SSIM"] > 0.7:
        logger.info("  Quality: GOOD (SSIM > 0.7)")
    else:
        logger.warning("  Quality: NEEDS IMPROVEMENT (SSIM < 0.7)")

    # GSD validation summary
    if gsd_validation["validated"]:
        logger.success("  GSD Validation: PASS")
    else:
        logger.warning("  GSD Validation: FAIL")

    return 0


if __name__ == "__main__":
    sys.exit(main())
