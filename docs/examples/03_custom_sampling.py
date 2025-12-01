"""Example 3: Custom Sampling Patterns

This example demonstrates creating and using custom sampling patterns
for k-space coverage.
"""

import matplotlib.pyplot as plt
import numpy as np

from prism.utils.sampling import fermat_spiral_sample, random_sample, star_sample


# Built-in sampling patterns
print("Built-in Sampling Patterns")
print("=" * 60)

# 1. Fermat Spiral (Recommended - optimal coverage)
fermat_centers = fermat_spiral_sample(n_samples=100, r=300)
print(f"Fermat Spiral: {len(fermat_centers)} samples")
print("  Provides near-optimal k-space coverage")
print("  Samples distributed uniformly by area")

# 2. Random Sampling
random_centers = random_sample(n_samples=100, r=300, seed=42)
print(f"\nRandom Sampling: {len(random_centers)} samples")
print("  Simple but may have gaps in coverage")
print("  Seed ensures reproducibility")

# 3. Star Pattern (Radial lines)
star_centers = star_sample(n_lines=6, samples_per_line=20, r=300)
print(f"\nStar Pattern: {len(star_centers)} samples")
print(f"  {6} radial lines with {20} samples each")
print("  Good for testing radial symmetry")


# Custom sampling pattern: Concentric circles
def concentric_circles_sample(n_circles=5, samples_per_circle=20, r_max=300):
    """Create concentric circle sampling pattern.

    Args:
        n_circles: Number of concentric circles
        samples_per_circle: Number of samples per circle
        r_max: Maximum radius

    Returns:
        List of [y, x] center positions
    """
    centers = []

    for circle_idx in range(n_circles):
        # Radius increases linearly
        radius = r_max * (circle_idx + 1) / n_circles

        # Angular positions
        for angle_idx in range(samples_per_circle):
            angle = 2 * np.pi * angle_idx / samples_per_circle

            # Convert to Cartesian coordinates
            y = radius * np.sin(angle)
            x = radius * np.cos(angle)

            centers.append([y, x])

    return centers


# Create custom pattern
print("\n\nCustom Sampling Pattern: Concentric Circles")
print("=" * 60)
custom_centers = concentric_circles_sample(n_circles=5, samples_per_circle=20, r_max=300)
print(f"Concentric Circles: {len(custom_centers)} samples")
print("  5 circles with 20 samples each")
print("  Good for circular objects")

# Visualize all patterns
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

patterns = [
    (fermat_centers, "Fermat Spiral", axes[0, 0]),
    (random_centers, "Random", axes[0, 1]),
    (star_centers, "Star Pattern", axes[1, 0]),
    (custom_centers, "Concentric Circles", axes[1, 1]),
]

for centers, title, ax in patterns:
    # Extract y and x coordinates
    y_coords = [c[0] for c in centers]
    x_coords = [c[1] for c in centers]

    # Plot
    ax.scatter(x_coords, y_coords, alpha=0.6, s=30)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)

    # Draw circle showing max radius
    circle = plt.Circle(
        (0, 0), 300, fill=False, color="red", linestyle="--", alpha=0.3, label="ROI boundary"
    )
    ax.add_patch(circle)

plt.tight_layout()
plt.savefig("sampling_patterns.png", dpi=150, bbox_inches="tight")
print("\nSaved visualization to sampling_patterns.png")


# Advanced: Adaptive sampling based on current reconstruction quality
def adaptive_sample(current_ssim_map, n_samples=100, r_max=300):
    """Sample more densely in regions with poor reconstruction.

    Args:
        current_ssim_map: 2D array of local SSIM values
        n_samples: Total number of samples
        r_max: Maximum radius

    Returns:
        List of [y, x] center positions
    """
    # Lower SSIM = higher sampling priority
    sampling_density = 1.0 - current_ssim_map

    # Normalize to probability distribution
    sampling_density = sampling_density / sampling_density.sum()

    # Sample positions based on density
    centers = []
    # ... implementation details ...

    return centers


print("\n\nAdvanced Features")
print("=" * 60)
print("- Adaptive sampling based on reconstruction quality")
print("- Multi-scale sampling (coarse to fine)")
print("- Jittered grids for better coverage")
print("- Import custom patterns from files")

# Example: Saving/loading patterns
import json


# Save pattern to JSON
pattern_data = {
    "name": "concentric_circles",
    "n_samples": len(custom_centers),
    "centers": custom_centers,
    "parameters": {"n_circles": 5, "samples_per_circle": 20, "r_max": 300},
}

with open("custom_pattern.json", "w") as f:
    json.dump(pattern_data, f, indent=2)
print("\nSaved pattern to custom_pattern.json")

# Load pattern from JSON
with open("custom_pattern.json", "r") as f:
    loaded_pattern = json.load(f)
print(f"Loaded pattern: {loaded_pattern['name']}")
print(f"  Samples: {loaded_pattern['n_samples']}")

print("\nDone!")
