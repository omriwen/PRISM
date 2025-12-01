"""
Automated Resolution Validation Suite
======================================

Validates all microscope and drone presets against theoretical limits.

This script provides:
1. Automated validation of all microscope presets against Abbe resolution limits
2. Automated validation of all drone presets against GSD calculations
3. HTML and CSV report generation
4. Summary statistics and pass/fail status

Usage:
    $ uv run python examples/python_api/09_resolution_validation.py

    # Microscope validation only
    $ uv run python examples/python_api/09_resolution_validation.py --category microscope

    # Drone validation only
    $ uv run python examples/python_api/09_resolution_validation.py --category drone

    # Custom tolerance (default: 15%)
    $ uv run python examples/python_api/09_resolution_validation.py --tolerance 0.10

    # Quick mode (skip image generation)
    $ uv run python examples/python_api/09_resolution_validation.py --quick

Generates:
    - results/validation/validation_report.html
    - results/validation/validation_summary.csv
    - results/validation/microscope_comparison.png (unless --quick)
    - results/validation/drone_comparison.png (unless --quick)

Expected Runtime: ~30 seconds

References:
    - Resolution limits: docs/references/optical_resolution_limits.md
    - Scenario presets: docs/references/scenario_preset_catalog.md
    - Validation baselines: spids/validation/baselines.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from prism.scenarios import get_scenario_preset, list_scenario_presets
from prism.scenarios.drone_camera import DroneScenarioConfig
from prism.scenarios.microscopy import MicroscopeScenarioConfig
from prism.validation.baselines import (
    GSDBaseline,
    ResolutionBaseline,
    compare_to_theoretical,
)


@dataclass
class ValidationEntry:
    """Single validation result entry."""

    preset: str
    category: str
    parameter: str
    theoretical: float
    measured: float
    unit: str
    error_percent: float
    tolerance_percent: float
    status: str

    def to_dict(self) -> dict:
        """Convert to dictionary for CSV export."""
        return {
            "preset": self.preset,
            "category": self.category,
            "parameter": self.parameter,
            "theoretical": f"{self.theoretical:.4f}",
            "measured": f"{self.measured:.4f}",
            "unit": self.unit,
            "error_percent": f"{self.error_percent:.2f}",
            "tolerance_percent": f"{self.tolerance_percent:.1f}",
            "status": self.status,
        }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automated Resolution Validation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["microscope", "drone", "all"],
        default="all",
        help="Category to validate (default: all)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.15,
        help="Acceptance tolerance as decimal (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/validation",
        help="Output directory for reports (default: results/validation)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode - skip image generation",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output for each preset",
    )
    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Configure loguru logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        level=level,
    )


def validate_microscope_preset(preset_name: str, tolerance: float = 0.15) -> list[ValidationEntry]:
    """Validate a single microscope preset.

    Parameters
    ----------
    preset_name : str
        Name of the microscope preset
    tolerance : float
        Acceptance tolerance (default: 15%)

    Returns
    -------
    list[ValidationEntry]
        List of validation results
    """
    results: list[ValidationEntry] = []

    scenario = get_scenario_preset(preset_name)
    assert isinstance(scenario, MicroscopeScenarioConfig)

    obj = scenario._obj

    # Validate lateral resolution (Abbe limit)
    theoretical_res = ResolutionBaseline.abbe_limit(scenario.wavelength, obj.numerical_aperture)
    measured_res = scenario.lateral_resolution_nm * 1e-9  # Convert nm to m

    validation = compare_to_theoretical(measured_res, theoretical_res, tolerance)

    results.append(
        ValidationEntry(
            preset=preset_name,
            category="microscope",
            parameter="lateral_resolution",
            theoretical=theoretical_res * 1e9,  # nm
            measured=scenario.lateral_resolution_nm,
            unit="nm",
            error_percent=validation.error_percent,
            tolerance_percent=tolerance * 100,
            status=validation.status,
        )
    )

    # Validate axial resolution
    theoretical_axial = ResolutionBaseline.axial_resolution(
        scenario.wavelength, obj.numerical_aperture, obj.medium_index
    )
    measured_axial = scenario.axial_resolution_um * 1e-6  # Convert µm to m

    axial_validation = compare_to_theoretical(measured_axial, theoretical_axial, tolerance)

    results.append(
        ValidationEntry(
            preset=preset_name,
            category="microscope",
            parameter="axial_resolution",
            theoretical=theoretical_axial * 1e6,  # µm
            measured=scenario.axial_resolution_um,
            unit="µm",
            error_percent=axial_validation.error_percent,
            tolerance_percent=tolerance * 100,
            status=axial_validation.status,
        )
    )

    return results


def validate_drone_preset(preset_name: str, tolerance: float = 0.15) -> list[ValidationEntry]:
    """Validate a single drone preset.

    Parameters
    ----------
    preset_name : str
        Name of the drone preset
    tolerance : float
        Acceptance tolerance (default: 15%)

    Returns
    -------
    list[ValidationEntry]
        List of validation results
    """
    results: list[ValidationEntry] = []

    scenario = get_scenario_preset(preset_name)
    assert isinstance(scenario, DroneScenarioConfig)

    # Convert sensor specs to SI units
    pixel_pitch_m = scenario._sensor.pixel_pitch_um * 1e-6  # µm to m
    focal_length_m = scenario._lens.focal_length_mm * 1e-3  # mm to m
    sensor_width_m = scenario._sensor.width_mm * 1e-3  # mm to m

    # Validate GSD
    theoretical_gsd = GSDBaseline.gsd(
        altitude=scenario.altitude_m,
        pixel_pitch=pixel_pitch_m,
        focal_length=focal_length_m,
    )
    measured_gsd = scenario.actual_gsd_cm / 100  # Convert cm to m

    validation = compare_to_theoretical(measured_gsd, theoretical_gsd, tolerance)

    results.append(
        ValidationEntry(
            preset=preset_name,
            category="drone",
            parameter="gsd",
            theoretical=theoretical_gsd * 100,  # cm
            measured=scenario.actual_gsd_cm,
            unit="cm",
            error_percent=validation.error_percent,
            tolerance_percent=tolerance * 100,
            status=validation.status,
        )
    )

    # Validate swath width
    theoretical_swath = GSDBaseline.swath_width(
        altitude=scenario.altitude_m,
        sensor_width=sensor_width_m,
        focal_length=focal_length_m,
    )
    measured_swath = scenario.swath_width_m

    swath_validation = compare_to_theoretical(measured_swath, theoretical_swath, tolerance)

    results.append(
        ValidationEntry(
            preset=preset_name,
            category="drone",
            parameter="swath_width",
            theoretical=theoretical_swath,
            measured=measured_swath,
            unit="m",
            error_percent=swath_validation.error_percent,
            tolerance_percent=tolerance * 100,
            status=swath_validation.status,
        )
    )

    return results


def validate_all_microscopes(tolerance: float = 0.15) -> list[ValidationEntry]:
    """Run validation for all microscope presets.

    Parameters
    ----------
    tolerance : float
        Acceptance tolerance (default: 15%)

    Returns
    -------
    list[ValidationEntry]
        List of all validation results
    """
    presets = list_scenario_presets("microscope")
    results: list[ValidationEntry] = []

    logger.info(f"Validating {len(presets)} microscope presets...")

    for preset_name in sorted(presets):
        try:
            preset_results = validate_microscope_preset(preset_name, tolerance)
            results.extend(preset_results)

            # Log summary for this preset
            status = "PASS" if all(r.status == "PASS" for r in preset_results) else "FAIL"
            logger.debug(f"  {preset_name}: {status}")

        except Exception as e:
            logger.error(f"  {preset_name}: ERROR - {e}")
            results.append(
                ValidationEntry(
                    preset=preset_name,
                    category="microscope",
                    parameter="error",
                    theoretical=0,
                    measured=0,
                    unit="",
                    error_percent=100,
                    tolerance_percent=tolerance * 100,
                    status=f"ERROR: {e}",
                )
            )

    return results


def validate_all_drones(tolerance: float = 0.15) -> list[ValidationEntry]:
    """Run validation for all drone presets.

    Parameters
    ----------
    tolerance : float
        Acceptance tolerance (default: 15%)

    Returns
    -------
    list[ValidationEntry]
        List of all validation results
    """
    presets = list_scenario_presets("drone")
    results: list[ValidationEntry] = []

    logger.info(f"Validating {len(presets)} drone presets...")

    for preset_name in sorted(presets):
        try:
            preset_results = validate_drone_preset(preset_name, tolerance)
            results.extend(preset_results)

            # Log summary for this preset
            status = "PASS" if all(r.status == "PASS" for r in preset_results) else "FAIL"
            logger.debug(f"  {preset_name}: {status}")

        except Exception as e:
            logger.error(f"  {preset_name}: ERROR - {e}")
            results.append(
                ValidationEntry(
                    preset=preset_name,
                    category="drone",
                    parameter="error",
                    theoretical=0,
                    measured=0,
                    unit="",
                    error_percent=100,
                    tolerance_percent=tolerance * 100,
                    status=f"ERROR: {e}",
                )
            )

    return results


def generate_csv_report(results: list[ValidationEntry], output_path: Path) -> None:
    """Generate CSV validation report.

    Parameters
    ----------
    results : list[ValidationEntry]
        Validation results
    output_path : Path
        Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "preset",
        "category",
        "parameter",
        "theoretical",
        "measured",
        "unit",
        "error_percent",
        "tolerance_percent",
        "status",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    logger.success(f"Saved CSV report to {output_path}")


def generate_html_report(
    results: list[ValidationEntry], output_path: Path, tolerance: float
) -> None:
    """Generate HTML validation report.

    Parameters
    ----------
    results : list[ValidationEntry]
        Validation results
    output_path : Path
        Output file path
    tolerance : float
        Tolerance used for validation
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate summary statistics
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0

    # Group by category
    microscope_results = [r for r in results if r.category == "microscope"]
    drone_results = [r for r in results if r.category == "drone"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPIDS Validation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}
        .stat {{
            background: #f0f0f0;
            padding: 15px 25px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
        .pass {{ color: #28a745; }}
        .fail {{ color: #dc3545; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #333;
            color: white;
            font-weight: 500;
        }}
        tr:hover {{
            background: #f8f8f8;
        }}
        .status-pass {{
            background: #d4edda;
            color: #155724;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 500;
        }}
        .status-fail {{
            background: #f8d7da;
            color: #721c24;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: 500;
        }}
        .category-section {{
            margin-top: 30px;
        }}
        .timestamp {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>SPIDS Automated Validation Report</h1>
    <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <div class="summary">
        <h2>Summary</h2>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat">
                <div class="stat-value pass">{passed}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat">
                <div class="stat-value fail">{failed}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{pass_rate:.1f}%</div>
                <div class="stat-label">Pass Rate</div>
            </div>
            <div class="stat">
                <div class="stat-value">{tolerance * 100:.0f}%</div>
                <div class="stat-label">Tolerance</div>
            </div>
        </div>
    </div>
"""

    # Microscope section
    if microscope_results:
        micro_passed = sum(1 for r in microscope_results if r.status == "PASS")
        html += f"""
    <div class="category-section">
        <h2>Microscope Presets ({micro_passed}/{len(microscope_results)} passed)</h2>
        <table>
            <thead>
                <tr>
                    <th>Preset</th>
                    <th>Parameter</th>
                    <th>Theoretical</th>
                    <th>Measured</th>
                    <th>Unit</th>
                    <th>Error</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
        for r in microscope_results:
            status_class = "status-pass" if r.status == "PASS" else "status-fail"
            html += f"""
                <tr>
                    <td>{r.preset}</td>
                    <td>{r.parameter}</td>
                    <td>{r.theoretical:.2f}</td>
                    <td>{r.measured:.2f}</td>
                    <td>{r.unit}</td>
                    <td>{r.error_percent:.2f}%</td>
                    <td><span class="{status_class}">{r.status}</span></td>
                </tr>
"""
        html += """
            </tbody>
        </table>
    </div>
"""

    # Drone section
    if drone_results:
        drone_passed = sum(1 for r in drone_results if r.status == "PASS")
        html += f"""
    <div class="category-section">
        <h2>Drone Presets ({drone_passed}/{len(drone_results)} passed)</h2>
        <table>
            <thead>
                <tr>
                    <th>Preset</th>
                    <th>Parameter</th>
                    <th>Theoretical</th>
                    <th>Measured</th>
                    <th>Unit</th>
                    <th>Error</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
"""
        for r in drone_results:
            status_class = "status-pass" if r.status == "PASS" else "status-fail"
            html += f"""
                <tr>
                    <td>{r.preset}</td>
                    <td>{r.parameter}</td>
                    <td>{r.theoretical:.2f}</td>
                    <td>{r.measured:.2f}</td>
                    <td>{r.unit}</td>
                    <td>{r.error_percent:.2f}%</td>
                    <td><span class="{status_class}">{r.status}</span></td>
                </tr>
"""
        html += """
            </tbody>
        </table>
    </div>
"""

    html += """
    <div class="summary">
        <h3>Validation Criteria</h3>
        <ul>
            <li><strong>Microscope lateral resolution</strong>: Abbe limit (0.61λ/NA)</li>
            <li><strong>Microscope axial resolution</strong>: 2nλ/NA²</li>
            <li><strong>Drone GSD</strong>: H × p / f (altitude × pixel_pitch / focal_length)</li>
            <li><strong>Drone swath width</strong>: H × W_sensor / f</li>
        </ul>
    </div>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)

    logger.success(f"Saved HTML report to {output_path}")


def generate_microscope_comparison_plot(output_path: Path) -> None:
    """Generate microscope validation comparison plot.

    Parameters
    ----------
    output_path : Path
        Output file path
    """
    presets = list_scenario_presets("microscope")

    # Collect data
    names = []
    nas = []
    theoretical_res = []
    measured_res = []

    for preset_name in sorted(presets):
        scenario = get_scenario_preset(preset_name)
        assert isinstance(scenario, MicroscopeScenarioConfig)

        names.append(preset_name.replace("microscope_", ""))
        nas.append(scenario._obj.numerical_aperture)
        theoretical_res.append(
            ResolutionBaseline.abbe_limit(scenario.wavelength, scenario._obj.numerical_aperture)
            * 1e9
        )
        measured_res.append(scenario.lateral_resolution_nm)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart comparison
    x = np.arange(len(names))
    width = 0.35

    axes[0].bar(x - width / 2, theoretical_res, width, label="Theoretical (Abbe)", alpha=0.8)
    axes[0].bar(x + width / 2, measured_res, width, label="Measured", alpha=0.8)
    axes[0].set_ylabel("Lateral Resolution (nm)")
    axes[0].set_xlabel("Preset")
    axes[0].set_title("Microscope Resolution: Theoretical vs Measured")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Scatter plot: NA vs Resolution
    axes[1].scatter(nas, theoretical_res, s=100, alpha=0.7, label="Theoretical", marker="o")
    axes[1].scatter(nas, measured_res, s=100, alpha=0.7, label="Measured", marker="x")

    # Add NA curve
    na_range = np.linspace(0.2, 1.5, 100)
    res_curve = 0.61 * 550 / na_range  # 550nm wavelength
    axes[1].plot(na_range, res_curve, "k--", alpha=0.5, label="Abbe limit (550nm)")

    axes[1].set_xlabel("Numerical Aperture (NA)")
    axes[1].set_ylabel("Lateral Resolution (nm)")
    axes[1].set_title("Resolution vs NA")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, 1.6)
    axes[1].set_ylim(0, max(theoretical_res) * 1.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.success(f"Saved microscope comparison plot to {output_path}")


def generate_drone_comparison_plot(output_path: Path) -> None:
    """Generate drone validation comparison plot.

    Parameters
    ----------
    output_path : Path
        Output file path
    """
    presets = list_scenario_presets("drone")

    # Collect data
    names = []
    altitudes = []
    theoretical_gsd = []
    measured_gsd = []

    for preset_name in sorted(presets):
        scenario = get_scenario_preset(preset_name)
        assert isinstance(scenario, DroneScenarioConfig)

        # Convert to SI units
        pixel_pitch_m = scenario._sensor.pixel_pitch_um * 1e-6
        focal_length_m = scenario._lens.focal_length_mm * 1e-3

        names.append(preset_name.replace("drone_", ""))
        altitudes.append(scenario.altitude_m)
        theoretical_gsd.append(
            GSDBaseline.gsd(
                altitude=scenario.altitude_m,
                pixel_pitch=pixel_pitch_m,
                focal_length=focal_length_m,
            )
            * 100
        )  # cm
        measured_gsd.append(scenario.actual_gsd_cm)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart comparison
    x = np.arange(len(names))
    width = 0.35

    axes[0].bar(x - width / 2, theoretical_gsd, width, label="Theoretical", alpha=0.8)
    axes[0].bar(x + width / 2, measured_gsd, width, label="Measured", alpha=0.8)
    axes[0].set_ylabel("GSD (cm)")
    axes[0].set_xlabel("Preset")
    axes[0].set_title("Drone GSD: Theoretical vs Measured")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Scatter plot: Altitude vs GSD
    axes[1].scatter(altitudes, theoretical_gsd, s=100, alpha=0.7, label="Theoretical", marker="o")
    axes[1].scatter(altitudes, measured_gsd, s=100, alpha=0.7, label="Measured", marker="x")

    axes[1].set_xlabel("Altitude (m)")
    axes[1].set_ylabel("GSD (cm)")
    axes[1].set_title("GSD vs Altitude")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.success(f"Saved drone comparison plot to {output_path}")


def main() -> int:
    """Main entry point for validation suite.

    Returns
    -------
    int
        Exit code (0 for success, 1 if any tests failed)
    """
    args = parse_args()
    setup_logging(args.verbose)

    logger.info("=" * 60)
    logger.info("SPIDS Automated Resolution Validation Suite")
    logger.info("=" * 60)
    logger.info(f"Tolerance: {args.tolerance * 100:.0f}%")
    logger.info(f"Output directory: {args.output_dir}")

    output_dir = Path(args.output_dir)
    results: list[ValidationEntry] = []

    # Run validations
    if args.category in ("microscope", "all"):
        microscope_results = validate_all_microscopes(args.tolerance)
        results.extend(microscope_results)

        micro_passed = sum(1 for r in microscope_results if r.status == "PASS")
        logger.info(f"Microscope: {micro_passed}/{len(microscope_results)} tests passed")

    if args.category in ("drone", "all"):
        drone_results = validate_all_drones(args.tolerance)
        results.extend(drone_results)

        drone_passed = sum(1 for r in drone_results if r.status == "PASS")
        logger.info(f"Drone: {drone_passed}/{len(drone_results)} tests passed")

    # Generate reports
    logger.info("")
    logger.info("Generating reports...")

    generate_csv_report(results, output_dir / "validation_summary.csv")
    generate_html_report(results, output_dir / "validation_report.html", args.tolerance)

    # Generate plots unless quick mode
    if not args.quick:
        logger.info("Generating comparison plots...")
        if args.category in ("microscope", "all"):
            generate_microscope_comparison_plot(output_dir / "microscope_comparison.png")
        if args.category in ("drone", "all"):
            generate_drone_comparison_plot(output_dir / "drone_comparison.png")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0

    logger.info("")
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total tests:  {total}")
    logger.info(f"Passed:       {passed}")
    logger.info(f"Failed:       {failed}")
    logger.info(f"Pass rate:    {pass_rate:.1f}%")
    logger.info("")

    if failed == 0:
        logger.success("All validations PASSED!")
        return 0
    else:
        logger.warning(f"{failed} validations FAILED")
        for r in results:
            if r.status != "PASS":
                logger.warning(f"  - {r.preset}: {r.parameter} ({r.error_percent:.2f}% error)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
