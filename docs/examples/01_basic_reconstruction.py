"""Example 1: Basic Progressive Reconstruction

This example demonstrates the core PRISM workflow:
1. Load an astronomical image
2. Create telescope and model
3. Progressive training with Fermat spiral sampling
4. Visualize and save results
"""

from pathlib import Path

import torch

from prism.core.instruments import Telescope, TelescopeConfig
from prism.core.measurement_system import MeasurementSystem
from prism.models.losses import LossAgg
from prism.models.networks import ProgressiveDecoder
from prism.utils.image import load_image
from prism.utils.metrics import compute_ssim, psnr
from prism.utils.sampling import fermat_spiral_sample
from prism.visualization.publication import PublicationPlotter


# Configuration
IMAGE_SIZE = 1024
OBJ_SIZE = 512
N_SAMPLES = 100
APERTURE_RADIUS = 50
SNR = 40
LEARNING_RATE = 0.001
LOSS_THRESHOLD = 0.001
MAX_EPOCHS_PER_SAMPLE = 1000

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load image (replace with your image path)
print("Loading image...")
image_path = "path/to/your/image.jpg"
image = load_image(image_path, size=IMAGE_SIZE).to(device)
print(f"Image shape: {image.shape}")

# Create components
print("\nInitializing components...")
config = TelescopeConfig(n_pixels=IMAGE_SIZE, aperture_radius_pixels=APERTURE_RADIUS, snr=SNR)
telescope = Telescope(config).to(device)
measurement_system = MeasurementSystem(telescope, obj_size=OBJ_SIZE).to(device)

model = ProgressiveDecoder(input_size=IMAGE_SIZE, output_size=OBJ_SIZE, use_bn=True, max_ch=256).to(
    device
)

criterion = LossAgg(loss_type="l1")
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Generate sampling pattern
print(f"\nGenerating {N_SAMPLES} Fermat spiral sample positions...")
centers = fermat_spiral_sample(n_samples=N_SAMPLES, r=300)

# Progressive training
print("\nStarting progressive training...")
print("=" * 60)

losses_history = []
ssims_history = []
psnrs_history = []

for sample_idx, center in enumerate(centers):
    print(f"\nSample {sample_idx + 1}/{N_SAMPLES}")
    print(f"Position: ({center[0]:.1f}, {center[1]:.1f})")

    # Generate measurement
    with torch.no_grad():
        current_rec = model()
    measurement = measurement_system.measure(image, current_rec, [center], add_noise=True)

    # Train to match measurement
    for epoch in range(MAX_EPOCHS_PER_SAMPLE):
        optimizer.zero_grad()
        output = model()
        loss_old, loss_new = criterion(output, measurement, measurement_system, [center])
        loss = loss_old + loss_new

        loss.backward()
        optimizer.step()

        # Check convergence
        if loss_old < LOSS_THRESHOLD and loss_new < LOSS_THRESHOLD:
            print(f"  Converged at epoch {epoch + 1}")
            print(
                f"  Loss: {loss.item():.6f} (old: {loss_old.item():.6f}, new: {loss_new.item():.6f})"
            )
            break

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch + 1}: Loss = {loss.item():.6f}")

    else:
        print(f"  Max epochs reached. Final loss: {loss.item():.6f}")

    # Add measurement to accumulated mask
    measurement_system.add_measurement([center])

    # Calculate metrics
    with torch.no_grad():
        reconstruction = model()
        # Convert to numpy for metrics
        rec_np = reconstruction.squeeze().cpu().numpy()
        img_np = image.squeeze().cpu().numpy()

        # Crop image to same size as reconstruction for comparison
        from prism.utils.image import crop_pad

        img_cropped = crop_pad(image, OBJ_SIZE).squeeze().cpu().numpy()

        ssim_val = compute_ssim(img_cropped, rec_np)
        psnr_val = psnr(img_cropped, rec_np)

        losses_history.append(loss.item())
        ssims_history.append(ssim_val)
        psnrs_history.append(psnr_val)

        print(f"  SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB")

print("\n" + "=" * 60)
print("Training complete!")

# Final metrics
print("\nFinal Metrics:")
print(f"  SSIM: {ssims_history[-1]:.4f}")
print(f"  PSNR: {psnrs_history[-1]:.2f} dB")
print(f"  Final Loss: {losses_history[-1]:.6f}")

# Visualize results
print("\nGenerating visualization...")
plotter = PublicationPlotter(figsize=(15, 5), dpi=150)

with torch.no_grad():
    final_reconstruction = model().squeeze().cpu().numpy()
    ground_truth = crop_pad(image, OBJ_SIZE).squeeze().cpu().numpy()
    static_measurement = telescope(image, [[0, 0]]).squeeze().cpu().numpy()

fig = plotter.plot_reconstruction_comparison(
    ground_truth=ground_truth, reconstruction=final_reconstruction, measurement=static_measurement
)

# Save results
output_dir = Path("results/basic_reconstruction")
output_dir.mkdir(parents=True, exist_ok=True)

fig.savefig(output_dir / "comparison.png", dpi=300, bbox_inches="tight")
print(f"Saved comparison figure to {output_dir / 'comparison.png'}")

# Save checkpoint
checkpoint = {
    "model_state_dict": model.state_dict(),
    "reconstruction": final_reconstruction,
    "losses": losses_history,
    "ssims": ssims_history,
    "psnrs": psnrs_history,
    "config": {
        "image_size": IMAGE_SIZE,
        "obj_size": OBJ_SIZE,
        "n_samples": N_SAMPLES,
        "aperture_radius": APERTURE_RADIUS,
        "snr": SNR,
    },
}
torch.save(checkpoint, output_dir / "checkpoint.pt")
print(f"Saved checkpoint to {output_dir / 'checkpoint.pt'}")

print("\nDone!")
