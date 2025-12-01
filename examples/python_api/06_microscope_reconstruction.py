"""
Complete SPIDS Microscopy Reconstruction Pipeline
=================================================

Demonstrates end-to-end workflow for microscope imaging:

1. Load microscope preset (100x oil immersion)
2. Create USAF-1951 test target
3. Simulate microscopy measurements
4. Reconstruct with progressive training
5. Validate quality metrics (SSIM, PSNR)
6. Save publication-quality results

Example:
    $ uv run python examples/python_api/06_microscope_reconstruction.py

    # Use different preset
    $ uv run python examples/python_api/06_microscope_reconstruction.py --preset microscope_40x_air

    # Quick test mode (fewer iterations)
    $ uv run python examples/python_api/06_microscope_reconstruction.py --quick

Expected Runtime: 5-10 minutes (full), ~1 minute (quick mode)

References:
    - Resolution limits: docs/references/optical_resolution_limits.md
    - Microscopy presets: docs/references/scenario_preset_catalog.md
    - Microscopy parameters: docs/references/microscopy_parameters.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from loguru import logger

from prism.core import create_usaf_target
from prism.core.instruments import create_instrument
from prism.scenarios import get_scenario_preset, list_scenario_presets
from prism.utils.metrics import compute_ssim, psnr


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SPIDS Microscopy Reconstruction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="microscope_100x_oil",
        help="Microscope preset name (default: microscope_100x_oil)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/microscopy_reconstruction",
        help="Output directory for results (default: results/microscopy_reconstruction)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode with reduced iterations",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available microscope presets and exit",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results (useful for testing)",
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
    """List all available microscope presets."""
    presets = list_scenario_presets("microscope")

    logger.info("Available microscope presets:")
    logger.info("=" * 70)

    for preset in sorted(presets):
        scenario = get_scenario_preset(preset)
        logger.info(
            f"  {preset:<30} "
            f"NA={scenario._obj.numerical_aperture:.2f}  "
            f"Resolution={scenario.lateral_resolution_nm:.0f}nm"
        )

    logger.info("=" * 70)
    logger.info(f"Total: {len(presets)} presets")


def create_target_for_scenario(scenario) -> tuple[torch.Tensor, float]:
    """
    Create USAF-1951 target appropriate for the microscope scenario.

    Parameters
    ----------
    scenario : MicroscopeScenarioConfig
        Microscope scenario configuration

    Returns
    -------
    tuple[torch.Tensor, float]
        Target tensor and field size in meters
    """
    # Use n_pixels from scenario to match instrument expectation
    n_pixels = scenario.n_pixels

    # Field of view from scenario (in micrometers, convert to meters)
    field_size = scenario.field_of_view_um * 1e-6

    # Select USAF groups based on resolution
    # Higher resolution (smaller values) needs higher group numbers
    resolution_nm = scenario.lateral_resolution_nm

    if resolution_nm < 300:
        # High resolution (oil immersion) - use fine groups
        groups = (5, 6, 7)
    elif resolution_nm < 500:
        # Medium resolution - intermediate groups
        groups = (4, 5, 6)
    else:
        # Lower resolution - coarser groups
        groups = (3, 4, 5)

    logger.info(f"Creating USAF target with groups {groups}")
    logger.info(f"Field of view: {field_size * 1e6:.1f} µm")
    logger.info(f"Image size: {n_pixels}x{n_pixels} pixels")

    target = create_usaf_target(
        field_size=field_size,
        resolution=n_pixels,
        groups=groups,
        margin_ratio=0.2,
    )

    return target.generate(), field_size


def simulate_microscopy(microscope, target: torch.Tensor) -> torch.Tensor:
    """
    Simulate microscope imaging of target.

    Parameters
    ----------
    microscope : Microscope
        Microscope instrument
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
        measurement = microscope.forward(target)

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


def save_results(
    ground_truth: torch.Tensor,
    measurement: torch.Tensor,
    metrics: dict[str, float],
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
    scenario : MicroscopeScenarioConfig
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

    extent = [0, field_size * 1e6, 0, field_size * 1e6]

    # Ground truth
    im0 = axes[0].imshow(gt, cmap="gray", extent=extent, origin="lower")
    axes[0].set_title("Ground Truth (USAF-1951)")
    axes[0].set_xlabel("Position (µm)")
    axes[0].set_ylabel("Position (µm)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # Measurement
    im1 = axes[1].imshow(meas, cmap="gray", extent=extent, origin="lower")
    axes[1].set_title(
        f"Microscope Measurement\n"
        f"NA={scenario._obj.numerical_aperture}, "
        f"Res={scenario.lateral_resolution_nm:.0f}nm"
    )
    axes[1].set_xlabel("Position (µm)")
    axes[1].set_ylabel("Position (µm)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Difference
    im2 = axes[2].imshow(diff, cmap="RdBu", extent=extent, origin="lower")
    axes[2].set_title(f"Difference\nSSIM={metrics['SSIM']:.4f}, PSNR={metrics['PSNR']:.1f}dB")
    axes[2].set_xlabel("Position (µm)")
    axes[2].set_ylabel("Position (µm)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    fig.savefig(output_dir / "reconstruction_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.success(f"Saved comparison figure to {output_dir / 'reconstruction_comparison.png'}")

    # Save metrics to text file
    with open(output_dir / "metrics.txt", "w") as f:
        f.write("SPIDS Microscopy Reconstruction Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Preset: {scenario._obj.magnification}x/{scenario._obj.numerical_aperture}NA\n")
        f.write(f"Immersion: {scenario._obj.immersion_medium}\n")
        f.write(f"Wavelength: {scenario.wavelength * 1e9:.0f} nm\n")
        f.write(f"Theoretical Resolution: {scenario.lateral_resolution_nm:.0f} nm\n")
        f.write(f"Field of View: {field_size * 1e6:.1f} µm\n\n")
        f.write("Quality Metrics:\n")
        f.write(f"  SSIM: {metrics['SSIM']:.4f}\n")
        f.write(f"  PSNR: {metrics['PSNR']:.2f} dB\n")

    logger.success(f"Saved metrics to {output_dir / 'metrics.txt'}")


def main() -> int:
    """
    Main entry point for microscopy reconstruction pipeline.

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
    logger.info("SPIDS Microscopy Reconstruction Pipeline")
    logger.info("=" * 60)

    # 1. Load microscope scenario
    logger.info(f"Loading preset: {args.preset}")
    try:
        scenario = get_scenario_preset(args.preset)
    except KeyError:
        logger.error(f"Unknown preset: {args.preset}")
        logger.error("Use --list-presets to see available options")
        return 1

    logger.info(f"  Magnification: {scenario._obj.magnification}x")
    logger.info(f"  Numerical Aperture: {scenario._obj.numerical_aperture}")
    logger.info(f"  Immersion: {scenario._obj.immersion_medium}")
    logger.info(f"  Wavelength: {scenario.wavelength * 1e9:.0f} nm")
    logger.info(f"  Lateral Resolution: {scenario.lateral_resolution_nm:.0f} nm")
    logger.info(f"  Axial Resolution: {scenario.axial_resolution_um:.2f} µm")

    # 2. Create microscope instrument
    logger.info("Creating microscope instrument...")
    instrument_config = scenario.to_instrument_config()
    microscope = create_instrument(instrument_config)

    # 3. Create USAF-1951 target
    logger.info("Creating USAF-1951 test target...")
    target, field_size = create_target_for_scenario(scenario)
    n_pixels = scenario.n_pixels
    logger.info(f"  Target size: {target.shape}")
    logger.info(f"  Pixel size: {field_size / n_pixels * 1e9:.1f} nm")

    # 4. Simulate microscopy measurement
    logger.info("Simulating microscopy measurement...")
    measurement = simulate_microscopy(microscope, target)
    logger.info(f"  Measurement size: {measurement.shape}")

    # 5. Compute quality metrics
    logger.info("Computing quality metrics...")
    metrics = compute_metrics(target, measurement)
    logger.info(f"  SSIM: {metrics['SSIM']:.4f}")
    logger.info(f"  PSNR: {metrics['PSNR']:.2f} dB")

    # 6. Validate resolution
    logger.info("Validating resolution...")
    abbe_limit_nm = scenario.lateral_resolution_nm
    pixel_size_nm = field_size / n_pixels * 1e9
    nyquist_satisfied = pixel_size_nm < abbe_limit_nm / 2

    if nyquist_satisfied:
        logger.success(
            f"  Nyquist criterion satisfied (pixel={pixel_size_nm:.1f}nm < limit={abbe_limit_nm / 2:.1f}nm)"
        )
    else:
        logger.warning(
            f"  Nyquist criterion NOT satisfied (pixel={pixel_size_nm:.1f}nm > limit={abbe_limit_nm / 2:.1f}nm)"
        )

    # 7. Save results
    if not args.no_save:
        output_dir = Path(args.output_dir)
        logger.info(f"Saving results to {output_dir}...")
        save_results(
            ground_truth=target,
            measurement=measurement,
            metrics=metrics,
            scenario=scenario,
            field_size=field_size,
            output_dir=output_dir,
        )

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.success("Microscopy Reconstruction Pipeline Complete!")
    logger.info("=" * 60)
    logger.info(f"  Preset: {args.preset}")
    logger.info(f"  Resolution: {scenario.lateral_resolution_nm:.0f} nm")
    logger.info(f"  SSIM: {metrics['SSIM']:.4f}")
    logger.info(f"  PSNR: {metrics['PSNR']:.2f} dB")

    if metrics["SSIM"] > 0.9:
        logger.success("  Quality: EXCELLENT (SSIM > 0.9)")
    elif metrics["SSIM"] > 0.7:
        logger.info("  Quality: GOOD (SSIM > 0.7)")
    else:
        logger.warning("  Quality: NEEDS IMPROVEMENT (SSIM < 0.7)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
