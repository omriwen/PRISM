"""Example 4: Advanced Visualization

This example demonstrates creating publication-quality visualizations
and reports from PRISM results.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from prism.reporting.html_reporter import HTMLReporter
from prism.reporting.statistics import ExperimentStatistics
from prism.visualization.interactive import InteractivePlotter
from prism.visualization.publication import PublicationPlotter


# Load checkpoint from previous experiment
checkpoint_path = "results/basic_reconstruction/checkpoint.pt"
print(f"Loading checkpoint from {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path)
except FileNotFoundError:
    print("Checkpoint not found. Please run example 01_basic_reconstruction.py first.")
    print("Creating dummy data for demonstration...")

    # Create dummy data for demonstration
    checkpoint = {
        "reconstruction": np.random.rand(512, 512),
        "losses": np.exp(-np.linspace(0, 5, 100)),
        "ssims": 1 - np.exp(-np.linspace(0, 5, 100)),
        "psnrs": 20 + 10 * (1 - np.exp(-np.linspace(0, 5, 100))),
        "config": {
            "image_size": 1024,
            "obj_size": 512,
            "n_samples": 100,
            "aperture_radius": 50,
            "snr": 40,
        },
    }

# Extract data
reconstruction = checkpoint["reconstruction"]
losses = checkpoint["losses"]
ssims = checkpoint["ssims"]
psnrs = checkpoint["psnrs"]
config = checkpoint["config"]

print("\nExperiment Configuration:")
print(f"  Samples: {config['n_samples']}")
print(f"  Aperture radius: {config['aperture_radius']} pixels")
print(f"  SNR: {config['snr']} dB")

# Create output directory
output_dir = Path("results/visualizations")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Publication-Quality Figures
print("\n1. Creating publication-quality figures...")
plotter = PublicationPlotter(figsize=(12, 8), dpi=300, use_latex=False)

# Training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(losses, linewidth=2)
axes[0].set_xlabel("Sample Index")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale("log")

axes[1].plot(ssims, linewidth=2, color="green")
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("SSIM")
axes[1].set_title("Structural Similarity")
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 1])

axes[2].plot(psnrs, linewidth=2, color="red")
axes[2].set_xlabel("Sample Index")
axes[2].set_ylabel("PSNR (dB)")
axes[2].set_title("Peak Signal-to-Noise Ratio")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
print(f"  Saved training curves to {output_dir / 'training_curves.png'}")

# Reconstruction visualization
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
im = ax.imshow(reconstruction, cmap="gray")
ax.set_title(
    f"Final Reconstruction\nSSIM: {ssims[-1]:.4f}, PSNR: {psnrs[-1]:.2f} dB",
    fontsize=14,
    fontweight="bold",
)
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(output_dir / "reconstruction.png", dpi=300, bbox_inches="tight")
print(f"  Saved reconstruction to {output_dir / 'reconstruction.png'}")

# 2. Interactive Visualizations
print("\n2. Creating interactive visualizations...")
interactive_plotter = InteractivePlotter()

# Interactive training curves
try:
    fig_interactive = interactive_plotter.plot_training_curves(
        losses=losses, ssims=ssims, psnrs=psnrs
    )
    fig_interactive.write_html(output_dir / "interactive_curves.html")
    print(f"  Saved interactive curves to {output_dir / 'interactive_curves.html'}")
except Exception as e:
    print(f"  Could not create interactive plot (plotly required): {e}")

# 3. Statistical Analysis
print("\n3. Generating statistical analysis...")
metrics_history = {"loss": losses, "ssim": ssims, "psnr": psnrs}

try:
    stats = ExperimentStatistics(metrics_history)
    summary = stats.generate_summary()

    print("\nExperiment Statistics:")
    print(f"  Final Loss: {summary['final_metrics']['loss']:.6f}")
    print(f"  Final SSIM: {summary['final_metrics']['ssim']:.4f}")
    print(f"  Final PSNR: {summary['final_metrics']['psnr']:.2f} dB")
    print(f"  Best SSIM: {summary['best_metrics']['ssim']:.4f}")
    print(f"  Best PSNR: {summary['best_metrics']['psnr']:.2f} dB")

    if "convergence_epoch" in summary and summary["convergence_epoch"] is not None:
        print(f"  Convergence at sample: {summary['convergence_epoch']}")

except Exception as e:
    print(f"  Could not generate statistics: {e}")

# 4. HTML Report
print("\n4. Generating HTML report...")
try:
    reporter = HTMLReporter("docs/templates/experiment_report.html")

    experiment_data = {
        "name": "PRISM Reconstruction Example",
        "config": config,
        "final_metrics": {"loss": losses[-1], "ssim": ssims[-1], "psnr": psnrs[-1]},
        "figures": {
            "reconstruction": plt.imread(output_dir / "reconstruction.png"),
            "training_curves": plt.imread(output_dir / "training_curves.png"),
        },
    }

    html_output = reporter.generate_report(experiment_data)

    with open(output_dir / "report.html", "w") as f:
        f.write(html_output)

    print(f"  Saved HTML report to {output_dir / 'report.html'}")

except Exception as e:
    print(f"  Could not generate HTML report: {e}")
    print("  (Template file may be missing)")

# 5. Multi-Panel Figure for Paper
print("\n5. Creating multi-panel figure for publication...")
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Panel A: Reconstruction
ax1 = fig.add_subplot(gs[0:2, 0:2])
im1 = ax1.imshow(reconstruction, cmap="gray")
ax1.set_title("A. Reconstruction", fontsize=16, fontweight="bold", loc="left")
ax1.axis("off")
plt.colorbar(im1, ax=ax1)

# Panel B: Training Loss
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(losses, linewidth=2, color="blue")
ax2.set_title("B. Training Loss", fontsize=12, fontweight="bold", loc="left")
ax2.set_xlabel("Sample Index", fontsize=10)
ax2.set_ylabel("Loss", fontsize=10)
ax2.set_yscale("log")
ax2.grid(True, alpha=0.3)

# Panel C: SSIM
ax3 = fig.add_subplot(gs[1, 2])
ax3.plot(ssims, linewidth=2, color="green")
ax3.set_title("C. SSIM", fontsize=12, fontweight="bold", loc="left")
ax3.set_xlabel("Sample Index", fontsize=10)
ax3.set_ylabel("SSIM", fontsize=10)
ax3.set_ylim([0, 1])
ax3.grid(True, alpha=0.3)

# Panel D: PSNR
ax4 = fig.add_subplot(gs[2, 2])
ax4.plot(psnrs, linewidth=2, color="red")
ax4.set_title("D. PSNR", fontsize=12, fontweight="bold", loc="left")
ax4.set_xlabel("Sample Index", fontsize=10)
ax4.set_ylabel("PSNR (dB)", fontsize=10)
ax4.grid(True, alpha=0.3)

# Panel E: Reconstruction Detail (center crop)
ax5 = fig.add_subplot(gs[2, 0])
center_size = 128
center = reconstruction.shape[0] // 2
detail = reconstruction[
    center - center_size // 2 : center + center_size // 2,
    center - center_size // 2 : center + center_size // 2,
]
ax5.imshow(detail, cmap="gray")
ax5.set_title("E. Detail View", fontsize=12, fontweight="bold", loc="left")
ax5.axis("off")

# Panel F: Histogram
ax6 = fig.add_subplot(gs[2, 1])
ax6.hist(reconstruction.flatten(), bins=50, alpha=0.7, color="gray", edgecolor="black")
ax6.set_title("F. Intensity Histogram", fontsize=12, fontweight="bold", loc="left")
ax6.set_xlabel("Intensity", fontsize=10)
ax6.set_ylabel("Count", fontsize=10)
ax6.grid(True, alpha=0.3)

plt.savefig(output_dir / "publication_figure.png", dpi=300, bbox_inches="tight")
print(f"  Saved publication figure to {output_dir / 'publication_figure.png'}")

print("\n" + "=" * 60)
print("All visualizations generated successfully!")
print(f"Output directory: {output_dir.absolute()}")
print("\nGenerated files:")
print("  - training_curves.png")
print("  - reconstruction.png")
print("  - publication_figure.png")
print("  - interactive_curves.html (if plotly available)")
print("  - report.html (if template available)")
print("\nDone!")
