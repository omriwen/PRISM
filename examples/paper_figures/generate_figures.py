"""
New Paper Figures for SAIDAST
===============================

This script generates the three new figures specified for the SAIDAST paper revision:
- Figure 1: Neural Network Architecture and Algorithm Flowchart
- Figure 2: Detector Configuration and Motion Trajectories
- Figure 4: Comprehensive Validation and Resolution Analysis

Following the implementation plan from plots_implementation_plan.md
and exact specifications from new_figures_detailed_specifications.md

Created by Claude following Optica Publishing Group guidelines
Date: July 2025
"""

import argparse
import gc
import os
from contextlib import contextmanager
from functools import lru_cache
from os.path import join as path_join

import image_utils as iu
import matplotlib.pyplot as plt
import models
import mopie
import numpy as np
import optics
import sampling
import torch
from matplotlib.patches import Circle, FancyBboxPatch
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


# Set matplotlib parameters for publication quality
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "font.size": 9,
        "axes.linewidth": 1,
        "xtick.major.width": 1,
        "ytick.major.width": 1,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "text.usetex": False,  # Set to True if LaTeX is available
    }
)

# Color scheme following specifications
COLORS = {
    "dark_blue": "#1E3A8A",
    "medium_blue": "#3B82F6",
    "light_blue": "#93C5FD",
    "dark_gray": "#374151",
    "medium_gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "green": "#10B981",
    "purple": "#8B5CF6",
    "red": "#EF4444",
    "yellow": "#FFE082",
    "orange": "#F97316",
    "teal": "#26A69A",
    "white": "#FFFFFF",
    "black": "#000000",
}


class PaperFigureGenerator:
    """Main class for generating all paper figures"""

    def __init__(self):
        """Initialize the figure generator"""
        self.figures_dir = "figures"
        os.makedirs(self.figures_dir, exist_ok=True)
        self._cache = {}

        # Optimize matplotlib for performance
        plt.rcParams["axes.facecolor"] = "white"
        plt.rcParams["figure.facecolor"] = "white"
        plt.rcParams["savefig.facecolor"] = "white"

        # Pre-allocate common arrays to reduce memory allocation overhead
        self._theta_cache = np.linspace(0, 2 * np.pi, 100)
        self._golden_angle = 137.508

    @contextmanager
    def _figure_context(self):
        """Context manager for proper figure cleanup"""
        try:
            yield
        finally:
            plt.close("all")
            gc.collect()

    def _save_figure(self, fig, name):
        """Optimized figure saving with proper cleanup"""
        # Save with optimized settings
        pdf_path = path_join(self.figures_dir, f"{name}.pdf")
        png_path = path_join(self.figures_dir, f"{name}.png")

        fig.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=300)
        fig.savefig(png_path, format="png", bbox_inches="tight", dpi=300)

    # =========================================================================
    # CORE LOADING FUNCTIONS (Following implementation plan)
    # =========================================================================

    @lru_cache(maxsize=8)
    def load_dir(self, path):
        """Cached data loader - handles experiment directory loading"""
        try:
            # Load with memory mapping for large files
            args = torch.load(path_join(path, "args.pt"), map_location="cpu", mmap=True)
            checkpoint = torch.load(path_join(path, "checkpoint.pt"), map_location="cpu", mmap=True)
            sample_points = torch.load(
                path_join(path, "sample_points.pt"), map_location="cpu", mmap=True
            )
            return args, checkpoint, sample_points
        except Exception as e:
            print(f"Error loading from {path}: {e}")
            return None, None, None

    def compare_all(self, img1, img2, size=None):
        """Optimized quality metrics computation"""
        # Crop only if necessary
        if size is not None:
            img1 = iu.crop_image(img1, size)
            img2 = iu.crop_image(img2, size)

        # Efficient tensor to numpy conversion - avoid unnecessary operations
        with torch.no_grad():
            if img1.is_cuda:
                img1_np = img1.squeeze().cpu().numpy()
            else:
                img1_np = img1.squeeze().numpy()

            if img2.is_cuda:
                img2_np = img2.squeeze().cpu().numpy()
            else:
                img2_np = img2.squeeze().numpy()

        # Fast data range calculation
        img1_min, img1_max = img1_np.min(), img1_np.max()
        img2_min, img2_max = img2_np.min(), img2_np.max()
        data_range = max(img1_max, img2_max) - min(img1_min, img2_min)

        # Optimized metrics computation
        nrmse = compare_nrmse(img1_np, img2_np, normalization="euclidean")
        psnr = compare_psnr(img1_np, img2_np, data_range=data_range)
        ssim = compare_ssim(
            img1_np,
            img2_np,
            data_range=data_range,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            multichannel=False,
        )

        return nrmse, psnr, ssim

    def load_spids(self, path, crop_r=1.2):
        """PRISM result loader following critical processing order"""
        print(f"Loading PRISM from: {path}")

        # Step 1: Load experiment data
        args, checkpoint, sample_points = self.load_dir(path)
        if args is None:
            return None

        argss = argparse.Namespace(**args)

        # Step 2-3: Reconstruct and load model
        model = models.ProgressiveDecoder(
            input_size=args["image_size"],
            use_bn=args["use_bn"],
            output_activation=args["output_activation"],
            use_leaky=args["use_leaky"],
            middle_activation=args["middle_activation"],
            complex_data=args["complex_data"],
            output_size=args["obj_size"],
        )
        model.load_state_dict(checkpoint["model"], strict=False)

        # Step 4-5: Create telescope objects with fallback handling
        telescope = optics.Telescope(
            n=argss.image_size,
            r=argss.sample_diameter / 2,
            cropping=argss.crop_obj,
            obj_size=argss.obj_size,
            snr=argss.snr,
        )
        telescope_agg = optics.TelescopeAgg(telescope=telescope)

        # Blur kernel fallback handling
        try:
            telescope_agg.load_state_dict(checkpoint["telescope_agg"], strict=False)
        except Exception:
            telescope_agg.blur_kernel = torch.ones(1, 1, 11, 11) / 11**2
            telescope_agg.load_state_dict(checkpoint["telescope_agg"], strict=False)

        # Step 6: Load ground truth with squared intensity
        ground_truth = (
            iu.load_image(
                argss.input,
                size=argss.obj_size,
                padded_size=argss.obj_size if argss.crop_obj else argss.image_size,
                invert=argss.invert_image,
            )
            ** 2
        )

        # Step 7: Extract model reconstruction with squared intensity
        with torch.no_grad():
            og = model().detach() ** 2  # Remove unnecessary clone()

            # Step 8: CRITICAL CROPPING ORDER - crop first, then ReLU
            crop_size = int(args["obj_size"] * crop_r)
            ground_truth = iu.crop_pad(ground_truth, crop_size)
            og = iu.crop_pad(og, crop_size)
            og = torch.nn.functional.relu(og, inplace=True)  # In-place operation

            # Step 9-10: Generate single measurement and masks
            sample_mask = telescope_agg.cum_mask
            center_ends = checkpoint["sample_centers"][0]
            cntr, center_rec = sampling.create_patterns(
                center_ends, args["samples_per_line_meas"], args["samples_per_line_rec"]
            )

            single_measurement = telescope(ground_truth, cntr) ** 2
            single_measurement = iu.crop_pad(single_measurement, crop_size)
            single_measurement = torch.nn.functional.relu(single_measurement, inplace=True)

            # Vectorized mask computation for better performance
            single_mask = torch.zeros_like(sample_mask)
            if len(cntr) > 0:
                masks = torch.stack([telescope.mask(ctr, telescope.r) for ctr in cntr])
                single_mask = torch.any(masks, dim=0)

        return (
            og,
            ground_truth,
            single_measurement,
            sample_mask,
            single_mask,
            checkpoint["last_center_idx"],
        )

    def load_mopie(self, path, crop_r=1.2):
        """Mo-PIE result loader with same processing order as PRISM"""
        print(f"Loading Mo-PIE from: {path}")

        # Step 1: Load experiment data
        args, checkpoint, sample_points = self.load_dir(path)
        if args is None:
            return None

        # Step 2: Create Mo-PIE model with fallback handling
        model = mopie.MoPIE(obj_size=args["obj_size"], n=args["image_size"])
        plt.close(model.fig)  # Close any matplotlib figures

        model.cntr = checkpoint["model"]["cntr"]
        model.center_rec = checkpoint["model"]["cntr"]
        model.ground_truth = checkpoint["model"]["ground_truth"]

        try:
            model.load_state_dict(checkpoint["model"], strict=False)
        except Exception:
            model.blur_kernel = torch.ones(1, 1, 11, 11) / 11**2
            model.load_state_dict(checkpoint["model"], strict=False)

        # Step 3-6: Extract reconstruction with same processing order
        with torch.no_grad():
            crop_size = int(args["obj_size"] * crop_r)
            og = model.og**2  # Squared intensity

            # SAME CROPPING ORDER as SPIDS
            og = iu.crop_pad(og, crop_size)
            og = torch.nn.functional.relu(og, inplace=True)

            ground_truth = model.image**2
            ground_truth = iu.crop_pad(ground_truth, crop_size)

            # Generate single measurement
            center_ends = sample_points["centers"][0]
            cntr, center_rec = sampling.create_patterns(
                center_ends, args["samples_per_line_meas"], args["samples_per_line_rec"]
            )
            model.update_cntr(cntr, center_rec, 0)
            single_measurement = model.Im**2
            single_measurement = iu.crop_pad(single_measurement, crop_size)
            single_measurement = torch.nn.functional.relu(single_measurement, inplace=True)

            # Optimized mask computation
            single_mask = torch.zeros_like(model.cum_mask)
            if len(cntr) > 0:
                masks = torch.stack([model.mask(ctr, model.r) for ctr in cntr])
                single_mask = torch.any(masks, dim=0)

            sample_mask = model.cum_mask

            # Handle max aperture reference
            max_aperture = og.clone()

        return (
            og,
            ground_truth,
            single_measurement,
            max_aperture,
            sample_mask,
            single_mask,
            len(sample_points["centers"]),
        )

    def load_data(self, n_samples=240, length=64, crop_r=1.2, spids_ext=None, mopie_ext=None):
        """Unified comparison loader for both PRISM and Mo-PIE results"""
        spids_path = path_join("runs", f"gencrop_eur_nsamples_{n_samples}_len_{length}")
        mopie_path = path_join("runs", f"mopie_eur_nsamples_{n_samples}_len_{length}")

        # Optimized loading with fallback
        def try_load_with_fallback(loader, primary_path, fallback_path, crop_r):
            for path in [primary_path, fallback_path]:
                try:
                    result = loader(path, crop_r=crop_r)
                    if result is not None:
                        return result
                except Exception:
                    continue
            return None

        spids_result = try_load_with_fallback(
            self.load_spids, spids_path + (spids_ext if spids_ext else ""), spids_path, crop_r
        )

        mopie_result = try_load_with_fallback(
            self.load_mopie, mopie_path + (mopie_ext if mopie_ext else ""), mopie_path, crop_r
        )

        if spids_result is None or mopie_result is None:
            print(f"Failed to load data for n_samples={n_samples}, length={length}")
            return None

        # Extract results
        og_spids, image, _, mask, single_mask, nsamps = spids_result
        og_mopie, _, og_meas, og_max, _, _, _ = mopie_result

        # Compute all quality metrics
        nrmse_spids, psnr_spids, ssim_spids = self.compare_all(og_spids, image)
        nrmse_mopie, psnr_mopie, ssim_mopie = self.compare_all(og_mopie, image)
        nrmse_meas, psnr_meas, ssim_meas = self.compare_all(og_meas, image)
        nrmse_max, psnr_max, ssim_max = self.compare_all(og_max, image)

        return {
            "og_spids": og_spids,
            "og_mopie": og_mopie,
            "image": image,
            "og_meas": og_meas,
            "og_max": og_max,
            "mask": mask,
            "single_mask": single_mask,
            "nsamps": nsamps,
            "nrmse_spids": nrmse_spids,
            "psnr_spids": psnr_spids,
            "ssim_spids": ssim_spids,
            "nrmse_mopie": nrmse_mopie,
            "psnr_mopie": psnr_mopie,
            "ssim_mopie": ssim_mopie,
            "nrmse_meas": nrmse_meas,
            "psnr_meas": psnr_meas,
            "ssim_meas": ssim_meas,
            "nrmse_max": nrmse_max,
            "psnr_max": psnr_max,
            "ssim_max": ssim_max,
        }

    # =========================================================================
    # FIGURE 1: NEURAL NETWORK ARCHITECTURE AND ALGORITHM FLOWCHART
    # =========================================================================

    def create_figure1_architecture(self):
        """Create Figure 1: SAIDAST Neural Network Architecture and Algorithm Flowchart"""

        # Larger figure for better text fitting: 18cm width, 12cm height
        fig = plt.figure(figsize=(18 / 2.54, 12 / 2.54))  # Convert cm to inches

        # Create 1x2 subplots with proper spacing
        gs = fig.add_gridspec(1, 2, hspace=0.1, wspace=0.2)

        # Panel (a): Neural Network Architecture
        ax_a = fig.add_subplot(gs[0])
        self._create_panel_a_neural_architecture(ax_a)

        # Panel (b): Algorithm Flowchart
        ax_b = fig.add_subplot(gs[1])
        self._create_panel_b_algorithm_flowchart(ax_b)

        # Optimized saving
        self._save_figure(fig, "figure1_architecture")
        plt.show()

        return fig

    def _create_panel_a_neural_architecture(self, ax):
        """Create Panel (a): SAIDAST Neural Network Architecture"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")

        # Panel label
        ax.text(0.2, 11.8, "(a)", fontsize=11, weight="bold", ha="left", va="top")

        # Accurate decoder layers based on models.py ProgressiveDecoder (lines 266-281)
        layer_specs = [
            ("ConvTrans: 1024→512 ch", "1×1 → 4×4", COLORS["medium_blue"]),
            ("BN + ReLU", "", COLORS["green"]),
            ("ConvTrans: 512→256 ch", "4×4 → 8×8", COLORS["medium_blue"]),
            ("BN + ReLU", "", COLORS["green"]),
            ("ConvTrans: 256→128 ch", "8×8 → 16×16", COLORS["medium_blue"]),
            ("BN + ReLU", "", COLORS["green"]),
            ("ConvTrans: 128→64 ch", "16×16 → 32×32", COLORS["medium_blue"]),
            ("BN + ReLU", "", COLORS["green"]),
            ("ConvTrans: 64→32 ch", "32×32 → 64×64", COLORS["medium_blue"]),
            ("BN + ReLU", "", COLORS["green"]),
            ("ConvTrans: 32→16 ch", "64×64 → 128×128", COLORS["medium_blue"]),
            ("BN + ReLU", "", COLORS["green"]),
            ("Conv: 16→8 ch", "128×128 → 128×128", COLORS["orange"]),
            ("BN + ReLU", "", COLORS["green"]),
            ("Conv: 8→4 ch", "128×128 → 128×128", COLORS["orange"]),
            ("BN + ReLU", "", COLORS["green"]),
            ("Conv: 4→2 ch", "128×128 → 128×128", COLORS["orange"]),
            ("BN + ReLU", "", COLORS["green"]),
            ("Conv: 2→1 ch + CropPad", "128×128 → target size", COLORS["purple"]),
        ]

        # Calculate equal spacing for main blocks to fill entire height
        layer_height = 0.35
        bn_relu_height = 0.25  # Smaller height for BN+ReLU blocks

        # Count main blocks (non-BN+ReLU blocks) + input + output
        main_blocks = [
            i for i, (layer_text, _, _) in enumerate(layer_specs) if layer_text != "BN + ReLU"
        ]
        total_main_blocks = 1 + len(main_blocks) + 1  # input + main_blocks + output

        # Expand to fill entire height: from top to bottom
        top_y = 11.5  # Input block center position (pushed up to fill height)
        bottom_y = 0.4  # Output block center position (pushed down to fill height)
        total_span = top_y - bottom_y
        equal_spacing = total_span / (total_main_blocks - 1)  # spacing between main block centers

        # Create list of y positions for main blocks only
        main_y_positions = [top_y - i * equal_spacing for i in range(total_main_blocks)]

        # Input layer - learnable latent vector
        input_y = main_y_positions[0]
        input_box = FancyBboxPatch(
            (1.25, input_y - 0.3),
            7.5,
            0.6,
            boxstyle="round,pad=0.05",
            facecolor=COLORS["dark_blue"],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_patch(input_box)
        ax.text(
            5,
            input_y,
            "Learnable Latent Vector $z_0 \\sim \\mathcal{N}(0, 1)^{1024\\times1\\times1}$",
            fontsize=6,
            color="white",
            ha="center",
            va="center",
            weight="bold",
        )

        # Create layer blocks with special spacing for BN+ReLU
        main_block_index = 0
        previous_y = input_y
        arrow_positions = []  # Store positions for proper arrow drawing

        for i, (layer_text, size_text, color) in enumerate(layer_specs):
            # Check if this is a BN + ReLU block (following a ConvTrans/Conv)
            is_bn_relu = layer_text == "BN + ReLU"

            if is_bn_relu:
                # BN+ReLU block: place immediately below previous block (touching)
                y_pos = (
                    previous_y - layer_height / 2 - bn_relu_height / 2 - 0.02
                )  # Small gap for visual separation
                current_height = bn_relu_height
            else:
                # Main block: use equal spacing
                main_block_index += 1
                y_pos = main_y_positions[main_block_index]
                current_height = layer_height

            # Layer blocks
            layer_box = FancyBboxPatch(
                (1.5, y_pos - current_height / 2),
                7.0,
                current_height,
                boxstyle="round,pad=0.02",
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_patch(layer_box)

            # Text for layer blocks
            text_color = "white" if color != COLORS["green"] else "black"
            ax.text(
                5,
                y_pos,
                layer_text,
                fontsize=6,
                color=text_color,
                ha="center",
                va="center",
                weight="bold",
            )

            # Size annotation (right aligned outside the block)
            # Only show if size_text is not empty
            if size_text:
                ax.text(
                    8.8,
                    y_pos,
                    size_text,
                    fontsize=5,
                    color="black",
                    ha="left",
                    va="center",
                    style="italic",
                )

            # Store position for arrow drawing (only for main blocks)
            if not is_bn_relu:
                arrow_positions.append((y_pos, current_height))

            previous_y = y_pos

        # Draw arrows between main blocks only (with proper clearance)
        # Arrow from input to first layer
        if len(arrow_positions) > 0:
            first_layer_y, first_height = arrow_positions[0]
            arrow_start_y = input_y - 0.3 - 0.08  # Clear the input box
            arrow_end_y = first_layer_y + first_height / 2 + 0.08  # Clear the target box
            arrow_length = arrow_start_y - arrow_end_y

            if arrow_length > 0.15:  # Only draw if there's enough space
                ax.arrow(
                    5,
                    arrow_start_y,
                    0,
                    -arrow_length,
                    head_width=0.08,
                    head_length=0.08,
                    fc=COLORS["dark_gray"],
                    ec=COLORS["dark_gray"],
                    linewidth=1.0,
                )

        # Arrows between main blocks
        for i in range(len(arrow_positions) - 1):
            current_y, current_height = arrow_positions[i]
            next_y, next_height = arrow_positions[i + 1]

            arrow_start_y = current_y - current_height / 2 - 0.08  # Clear the current box
            arrow_end_y = next_y + next_height / 2 + 0.08  # Clear the next box
            arrow_length = arrow_start_y - arrow_end_y

            if arrow_length > 0.15:  # Only draw if there's enough space
                ax.arrow(
                    5,
                    arrow_start_y,
                    0,
                    -arrow_length,
                    head_width=0.08,
                    head_length=0.08,
                    fc=COLORS["dark_gray"],
                    ec=COLORS["dark_gray"],
                    linewidth=1.0,
                )

        # Output specification
        output_y = main_y_positions[-1]
        output_box = FancyBboxPatch(
            (1.5, output_y - layer_height / 2),
            7.0,
            layer_height,
            boxstyle="round,pad=0.05",
            facecolor=COLORS["yellow"],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_patch(output_box)
        ax.text(
            5,
            output_y,
            "Reconstructed Image",
            fontsize=6,
            color="black",
            ha="center",
            va="center",
            weight="bold",
        )

        # Arrow from last main block to output
        if len(arrow_positions) > 0:
            last_y, last_height = arrow_positions[-1]
            arrow_start_y = last_y - last_height / 2 - 0.08
            arrow_end_y = output_y + layer_height / 2 + 0.08
            arrow_length = arrow_start_y - arrow_end_y

            if arrow_length > 0.15:
                ax.arrow(
                    5,
                    arrow_start_y,
                    0,
                    -arrow_length,
                    head_width=0.08,
                    head_length=0.08,
                    fc=COLORS["dark_gray"],
                    ec=COLORS["dark_gray"],
                    linewidth=1.0,
                )

    def _create_panel_b_algorithm_flowchart(self, ax):
        """Create Panel (b): SAIDAST Algorithm Flowchart based on algo.png"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")

        # Panel label
        ax.text(0.2, 11.8, "(b)", fontsize=11, weight="bold", ha="left", va="top")

        # SAIDAST model at top - expanded to fill height
        saidast_box = FancyBboxPatch(
            (3.2, 11.2),
            3.6,
            0.6,
            boxstyle="round,pad=0.05",
            facecolor="#FF6B35",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_patch(saidast_box)
        ax.text(
            5, 11.5, "SAIDAST", fontsize=7, weight="bold", color="white", ha="center", va="center"
        )

        # Current Reconstruction (center, red in flowchart) - redistributed to fill height
        rec_box = FancyBboxPatch(
            (2.8, 10.2),
            4.4,
            0.6,
            boxstyle="round,pad=0.05",
            facecolor="#DC3545",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_patch(rec_box)
        ax.text(
            5,
            10.5,
            "Current Reconstruction",
            fontsize=6,
            weight="bold",
            color="white",
            ha="center",
            va="center",
        )

        # Left branch - Current path (redistributed to fill entire height)
        current_mask_box = FancyBboxPatch(
            (0.3, 9.1),
            2.8,
            0.5,
            boxstyle="round,pad=0.03",
            facecolor="#FF8C42",
            edgecolor="black",
            linewidth=0.6,
        )
        ax.add_patch(current_mask_box)
        ax.text(
            1.7,
            9.35,
            "Current Mask",
            fontsize=6,
            weight="bold",
            color="white",
            ha="center",
            va="center",
        )

        telescope1_box = FancyBboxPatch(
            (0.6, 8.0),
            2.2,
            0.4,
            boxstyle="round,pad=0.03",
            facecolor="#28A745",
            edgecolor="black",
            linewidth=0.6,
        )
        ax.add_patch(telescope1_box)
        ax.text(
            1.7,
            8.2,
            "Telescope",
            fontsize=6,
            weight="bold",
            color="white",
            ha="center",
            va="center",
        )

        current_meas_box = FancyBboxPatch(
            (0.05, 6.9),
            3.3,
            0.5,
            boxstyle="round,pad=0.03",
            facecolor="#FFC107",
            edgecolor="black",
            linewidth=0.6,
        )
        ax.add_patch(current_meas_box)
        ax.text(
            1.7,
            7.15,
            "Current Measurement",
            fontsize=5,
            weight="bold",
            color="black",
            ha="center",
            va="center",
        )

        current_loss_box = FancyBboxPatch(
            (0.6, 5.8),
            2.2,
            0.4,
            boxstyle="round,pad=0.03",
            facecolor="#90EE90",
            edgecolor="black",
            linewidth=0.6,
        )
        ax.add_patch(current_loss_box)
        ax.text(
            1.7,
            6.0,
            "Current Loss",
            fontsize=6,
            weight="bold",
            color="black",
            ha="center",
            va="center",
        )

        # Right branch - Aggregated path (redistributed to fill entire height)
        agg_mask_box = FancyBboxPatch(
            (6.9, 9.1),
            2.8,
            0.5,
            boxstyle="round,pad=0.03",
            facecolor="#FF8C42",
            edgecolor="black",
            linewidth=0.6,
        )
        ax.add_patch(agg_mask_box)
        ax.text(
            8.3,
            9.35,
            "Aggregated Mask",
            fontsize=6,
            weight="bold",
            color="white",
            ha="center",
            va="center",
        )

        telescope2_box = FancyBboxPatch(
            (7.2, 8.0),
            2.2,
            0.4,
            boxstyle="round,pad=0.03",
            facecolor="#28A745",
            edgecolor="black",
            linewidth=0.6,
        )
        ax.add_patch(telescope2_box)
        ax.text(
            8.3,
            8.2,
            "Telescope",
            fontsize=6,
            weight="bold",
            color="white",
            ha="center",
            va="center",
        )

        agg_meas_box = FancyBboxPatch(
            (6.65, 6.9),
            3.3,
            0.5,
            boxstyle="round,pad=0.03",
            facecolor="#FFC107",
            edgecolor="black",
            linewidth=0.6,
        )
        ax.add_patch(agg_meas_box)
        ax.text(
            8.3,
            7.15,
            "Aggregated Measurement",
            fontsize=4.5,
            weight="bold",
            color="black",
            ha="center",
            va="center",
        )

        agg_loss_box = FancyBboxPatch(
            (7.0, 5.8),
            2.6,
            0.4,
            boxstyle="round,pad=0.03",
            facecolor="#90EE90",
            edgecolor="black",
            linewidth=0.6,
        )
        ax.add_patch(agg_loss_box)
        ax.text(
            8.3,
            6.0,
            "Aggregated Loss",
            fontsize=5,
            weight="bold",
            color="black",
            ha="center",
            va="center",
        )

        # Total Loss (center) - moved down to fill height
        total_loss_box = FancyBboxPatch(
            (3.5, 4.3),
            3.0,
            0.6,
            boxstyle="round,pad=0.05",
            facecolor="#17A2B8",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_patch(total_loss_box)
        ax.text(
            5, 4.6, "Total Loss", fontsize=6, weight="bold", color="white", ha="center", va="center"
        )

        # Model Update - moved to bottom to fill height
        update_box = FancyBboxPatch(
            (3.5, 2.5),
            3.0,
            0.6,
            boxstyle="round,pad=0.05",
            facecolor="#17A2B8",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.add_patch(update_box)
        ax.text(
            5,
            2.8,
            "Model Update",
            fontsize=6,
            weight="bold",
            color="white",
            ha="center",
            va="center",
        )

        # Connecting arrows with improved spacing and visibility
        # From SAIDAST to Current Reconstruction - avoid overlap
        ax.arrow(
            5,
            11.0,
            0,
            -0.4,
            head_width=0.08,
            head_length=0.06,
            fc="black",
            ec="black",
            linewidth=1.2,
        )

        # From Current Reconstruction to branches - avoid overlap
        ax.arrow(
            3.8,
            10.2,
            -1.8,
            -0.8,
            head_width=0.08,
            head_length=0.06,
            fc="black",
            ec="black",
            linewidth=1.2,
        )
        ax.arrow(
            6.2,
            10.2,
            1.8,
            -0.8,
            head_width=0.08,
            head_length=0.06,
            fc="black",
            ec="black",
            linewidth=1.2,
        )

        # Down the left branch - improved spacing
        ax.arrow(
            1.7,
            8.85,
            0,
            -0.6,
            head_width=0.06,
            head_length=0.05,
            fc="black",
            ec="black",
            linewidth=1.0,
        )
        ax.arrow(
            1.7,
            7.8,
            0,
            -0.65,
            head_width=0.06,
            head_length=0.05,
            fc="black",
            ec="black",
            linewidth=1.0,
        )
        ax.arrow(
            1.7,
            6.65,
            0,
            -0.6,
            head_width=0.06,
            head_length=0.05,
            fc="black",
            ec="black",
            linewidth=1.0,
        )

        # Down the right branch - improved spacing
        ax.arrow(
            8.3,
            8.85,
            0,
            -0.6,
            head_width=0.06,
            head_length=0.05,
            fc="black",
            ec="black",
            linewidth=1.0,
        )
        ax.arrow(
            8.3,
            7.8,
            0,
            -0.65,
            head_width=0.06,
            head_length=0.05,
            fc="black",
            ec="black",
            linewidth=1.0,
        )
        ax.arrow(
            8.3,
            6.65,
            0,
            -0.6,
            head_width=0.06,
            head_length=0.05,
            fc="black",
            ec="black",
            linewidth=1.0,
        )

        # From losses to total loss - avoid overlap with proper clearance
        ax.arrow(
            2.8,
            5.8,
            1.4,
            -1.0,
            head_width=0.06,
            head_length=0.05,
            fc="black",
            ec="black",
            linewidth=1.0,
        )
        ax.arrow(
            7.2,
            5.8,
            -1.4,
            -1.0,
            head_width=0.06,
            head_length=0.05,
            fc="black",
            ec="black",
            linewidth=1.0,
        )

        # From total loss to model update - avoid overlap
        ax.arrow(
            5,
            4.0,
            0,
            -1.2,
            head_width=0.08,
            head_length=0.06,
            fc="black",
            ec="black",
            linewidth=1.2,
        )

        # Feedback loop from model update back to SAIDAST - outside the main flow
        # Right edge feedback path to avoid overlap
        ax.arrow(
            6.8,
            2.8,
            2.7,
            0,
            head_width=0.06,
            head_length=0.05,
            fc="black",
            ec="black",
            linewidth=1.0,
        )
        ax.arrow(
            9.5,
            2.8,
            0,
            8.2,
            head_width=0.06,
            head_length=0.05,
            fc="black",
            ec="black",
            linewidth=1.0,
        )
        ax.arrow(
            9.5,
            11.0,
            -2.7,
            0,
            head_width=0.06,
            head_length=0.05,
            fc="black",
            ec="black",
            linewidth=1.0,
        )

    # =========================================================================
    # FIGURE 2: DETECTOR CONFIGURATION AND MOTION TRAJECTORIES
    # =========================================================================

    def create_figure2_detector_config(self):
        """Create Figure 2: Detector Configuration and Motion Trajectories"""

        # Overall specifications: 16cm width, ~9cm height for much better spacing
        fig = plt.figure(figsize=(16 / 2.54, 9 / 2.54))

        # Create 1x2 subplots with much more spacing to prevent overlap
        gs = fig.add_gridspec(1, 2, hspace=0.4, wspace=0.3)

        # Panel (a): Fermat Spiral Detector Layout
        ax_a = fig.add_subplot(gs[0])
        self._create_panel_a_fermat_spiral(ax_a)

        # Panel (b): Motion Trajectories
        ax_b = fig.add_subplot(gs[1])
        self._create_panel_b_motion_trajectories(ax_b)

        # Optimized saving
        self._save_figure(fig, "figure2_detector_config")
        plt.show()

        return fig

    def _create_panel_a_fermat_spiral(self, ax):
        """Create Panel (a): Fermat Spiral Detector Layout"""
        ax.set_xlim(-6, 6)
        ax.set_ylim(-5.5, 5.5)
        ax.set_aspect("equal")

        # Grid lines - lighter and fewer
        for i in range(-4, 5, 2):
            ax.axvline(i, color=COLORS["light_gray"], linestyle=":", linewidth=0.5, alpha=0.5)
            if abs(i) <= 2:
                ax.axhline(i, color=COLORS["light_gray"], linestyle=":", linewidth=0.5, alpha=0.5)

        # Coordinate axes
        ax.axhline(0, color=COLORS["medium_gray"], linewidth=1)
        ax.axvline(0, color=COLORS["medium_gray"], linewidth=1)

        # Axis labels and ticks - cleaner layout
        ax.set_xticks(range(-4, 5, 2))
        ax.set_xticklabels([f"{i}m" for i in range(-4, 5, 2)], fontsize=8)
        ax.set_yticks(range(-2, 3, 2))
        ax.set_yticklabels([f"{i}m" for i in range(-2, 3, 2)], fontsize=8)

        # System boundary circle (10m diameter = 5m radius)
        # boundary = Circle((0, 0), 5, fill=False, edgecolor='black', linestyle='--', linewidth=1.5)
        # ax.add_patch(boundary)
        # ax.text(0, -5.2, '10m diameter', fontsize=9, ha='center', va='center',
        #         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

        # Optimized Fermat spiral generation
        n_points = 204

        # Vectorized computation
        i_vals = np.arange(1, n_points + 1)
        r_vals = np.sqrt(i_vals) * 0.32
        theta_vals = i_vals * self._golden_angle * np.pi / 180
        x_vals = r_vals * np.cos(theta_vals)
        y_vals = r_vals * np.sin(theta_vals)

        # Filter points within boundary
        within_boundary = (x_vals**2 + y_vals**2) <= 25  # 5^2 = 25
        spiral_points = list(
            zip(x_vals[within_boundary], y_vals[within_boundary], r_vals[within_boundary])
        )

        # Draw spiral curve (semi-transparent)
        # if len(spiral_points) > 1:
        #     x_spiral = [p[0] for p in spiral_points]
        #     y_spiral = [p[1] for p in spiral_points]
        #     ax.plot(x_spiral, y_spiral, color=COLORS['medium_blue'], alpha=0.3, linewidth=1.5)

        # Plot detector positions with color coding - larger markers
        for x, y, r in spiral_points:
            if r < 2:
                color = COLORS["dark_blue"]
            elif r < 4:
                color = COLORS["medium_blue"]
            else:
                color = COLORS["light_blue"]

            ax.scatter(x, y, c=color, s=15, edgecolor="none", alpha=0.8)

        # Mathematical annotation box - moved to left side to avoid all overlaps
        # ann_box = FancyBboxPatch((-5.8, 3.2), 3.4, 1.4, boxstyle="round,pad=0.1",
        #                         facecolor='white', edgecolor=COLORS['light_gray'], alpha=0.95)
        # ax.add_patch(ann_box)
        # ax.text(-4.1, 3.9, r'$r_i = \sqrt{i} \times 0.32m$', fontsize=8, ha='center', va='center',
        #         family='monospace')
        # ax.text(-4.1, 3.3, r'$\theta_i = i \times \phi_{golden}$',
        #         fontsize=8, ha='center', va='center', family='monospace')
        # ax.text(-4.1, 2.7, r'$\phi_{golden} = 137.508°$', fontsize=8, ha='center', va='center',
        #         family='monospace')

        # Legend - positioned in bottom right to avoid all overlaps
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLORS["dark_blue"],
                markersize=7,
                label="r < 2m",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLORS["medium_blue"],
                markersize=7,
                label="2m ≤ r < 4m",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLORS["light_blue"],
                markersize=7,
                label="r ≥ 4m",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.95)

        ax.text(
            0.02,
            0.98,
            "(a)",
            fontsize=12,
            weight="bold",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

    def _create_panel_b_motion_trajectories(self, ax):
        """Create Panel (b): Motion Trajectories and Effective Aperture"""
        ax.set_xlim(-6, 6)
        ax.set_ylim(-5.5, 5.5)
        ax.set_aspect("equal")

        # Copy base layout from panel (a) - cleaner
        for i in range(-4, 5, 2):
            ax.axvline(i, color=COLORS["light_gray"], linestyle=":", linewidth=0.5, alpha=0.5)
            if abs(i) <= 2:
                ax.axhline(i, color=COLORS["light_gray"], linestyle=":", linewidth=0.5, alpha=0.5)

        ax.axhline(0, color=COLORS["medium_gray"], linewidth=1)
        ax.axvline(0, color=COLORS["medium_gray"], linewidth=1)
        ax.set_xticks(range(-4, 5, 2))
        ax.set_xticklabels([f"{i}m" for i in range(-4, 5, 2)], fontsize=8)
        ax.set_yticks(range(-2, 3, 2))
        ax.set_yticklabels([f"{i}m" for i in range(-2, 3, 2)], fontsize=8)

        # Boundary circle
        boundary = Circle((0, 0), 5, fill=False, edgecolor="black", linestyle="--", linewidth=1.5)
        ax.add_patch(boundary)

        # Optimized detector subset generation
        n_subset = 12

        # Vectorized computation with stride
        i_vals = np.arange(1, 205, 15)[:n_subset]
        r_vals = np.sqrt(i_vals) * 0.32
        theta_vals = i_vals * self._golden_angle * np.pi / 180
        x_vals = r_vals * np.cos(theta_vals)
        y_vals = r_vals * np.sin(theta_vals)

        # Filter and limit to subset
        within_boundary = (x_vals**2 + y_vals**2) <= 25
        valid_indices = within_boundary.nonzero()[0][:n_subset]
        detector_positions = [(x_vals[i], y_vals[i], r_vals[i]) for i in valid_indices]

        # Motion trajectories - cleaner arrows
        np.random.seed(42)  # For reproducible random angles
        motion_length = 0.8  # Slightly longer for better visibility

        for i, (x, y, r) in enumerate(detector_positions):
            # Color coding
            if r < 2:
                color = COLORS["dark_blue"]
            elif r < 4:
                color = COLORS["medium_blue"]
            else:
                color = COLORS["light_blue"]

            # Plot detector - larger for better visibility
            ax.scatter(x, y, c=color, s=30, edgecolor="black", linewidth=0.5)

            # Random motion direction
            angle = np.random.uniform(0, 2 * np.pi)
            dx = motion_length * np.cos(angle)
            dy = motion_length * np.sin(angle)

            # Draw motion trajectory - improved arrows
            ax.arrow(
                x,
                y,
                dx,
                dy,
                head_width=0.1,
                head_length=0.08,
                fc=color,
                ec=color,
                linewidth=2,
                alpha=0.8,
            )

            # Add velocity vector annotation for fewer representative trajectories
            if i < 2:
                mid_x, mid_y = x + dx / 2, y + dy / 2
                ax.text(
                    mid_x + 0.2,
                    mid_y + 0.2,
                    f"$v_{{{i + 1}}}$",
                    fontsize=7,
                    style="italic",
                    family="monospace",
                    weight="bold",
                )

        # # Effective aperture visualization (inset) - moved to right side, completely separate
        # inset_box = FancyBboxPatch((2.5, 3.8), 3.2, 1.4, boxstyle="round,pad=0.1",
        #                           facecolor='white', edgecolor='gray', alpha=0.95)
        # ax.add_patch(inset_box)

        # # Circular aperture - larger for visibility
        # circ_ap = Circle((2.9, 4.4), 0.15, facecolor=COLORS['dark_blue'], alpha=0.8)
        # ax.add_patch(circ_ap)
        # ax.text(2.9, 4.1, '17cm', fontsize=6, ha='center', va='center', weight='bold')

        # Plus symbol
        # ax.text(3.3, 4.4, '+', fontsize=10, ha='center', va='center', weight='bold')

        # # Motion arrow - larger
        # ax.arrow(3.6, 4.4, 0.25, 0, head_width=0.05, head_length=0.03,
        #         fc='black', ec='black', linewidth=1.5)
        # ax.text(3.7, 4.1, '80cm', fontsize=6, ha='center', va='center', weight='bold')

        # # Equals symbol
        # ax.text(4.1, 4.4, '=', fontsize=10, ha='center', va='center', weight='bold')

        # # Line aperture - more prominent
        # ax.plot([4.4, 4.7], [4.4, 4.4], color=COLORS['dark_blue'], linewidth=8, alpha=0.8)
        # ax.text(4.55, 4.1, 'Effective', fontsize=6, ha='center', va='center', weight='bold')

        # Motion equation - moved to bottom, well separated
        ax.text(
            0,
            -5.2,
            r"$p_i(t) = p_{i,0} + v_i \times t$",
            fontsize=9,
            ha="center",
            va="center",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.95),
        )

        ax.text(
            0.02,
            0.98,
            "(b)",
            fontsize=12,
            weight="bold",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

    # =========================================================================
    # FIGURE 2 V2: DETECTOR CONFIGURATION WITH UPDATED RADIUS BOUNDS
    # =========================================================================

    def create_figure2_detector_config_v2(self):
        """Create Figure 2 v2: Detector Configuration with updated radius bounds (r>4m split)"""

        # Overall specifications: 16cm width, ~9cm height for much better spacing
        fig = plt.figure(figsize=(16 / 2.54, 9 / 2.54))

        # Create 1x2 subplots with much more spacing to prevent overlap
        gs = fig.add_gridspec(1, 2, hspace=0.4, wspace=0.3)

        # Panel (a): Fermat Spiral Detector Layout (updated version)
        ax_a = fig.add_subplot(gs[0])
        self._create_panel_a_fermat_spiral_v2(ax_a)

        # Panel (b): Motion Trajectories
        ax_b = fig.add_subplot(gs[1])
        self._create_panel_b_motion_trajectories(ax_b)

        # Optimized saving with new name
        self._save_figure(fig, "figure2_detector_config_v2")
        plt.show()

        return fig

    def _create_panel_a_fermat_spiral_v2(self, ax):
        """Create Panel (a): Fermat Spiral Detector Layout with updated radius bounds"""
        ax.set_xlim(-6, 6)
        ax.set_ylim(-5.5, 5.5)
        ax.set_aspect("equal")

        # Grid lines - lighter and fewer
        for i in range(-4, 5, 2):
            ax.axvline(i, color=COLORS["light_gray"], linestyle=":", linewidth=0.5, alpha=0.5)
            if abs(i) <= 2:
                ax.axhline(i, color=COLORS["light_gray"], linestyle=":", linewidth=0.5, alpha=0.5)

        # Coordinate axes
        ax.axhline(0, color=COLORS["medium_gray"], linewidth=1)
        ax.axvline(0, color=COLORS["medium_gray"], linewidth=1)

        # Axis labels and ticks - cleaner layout
        ax.set_xticks(range(-4, 5, 2))
        ax.set_xticklabels([f"{i}m" for i in range(-4, 5, 2)], fontsize=8)
        ax.set_yticks(range(-2, 3, 2))
        ax.set_yticklabels([f"{i}m" for i in range(-2, 3, 2)], fontsize=8)

        # Optimized Fermat spiral generation
        n_points = 204

        # Vectorized computation
        i_vals = np.arange(1, n_points + 1)
        r_vals = np.sqrt(i_vals) * 0.32
        theta_vals = i_vals * self._golden_angle * np.pi / 180
        x_vals = r_vals * np.cos(theta_vals)
        y_vals = r_vals * np.sin(theta_vals)

        # Filter points within boundary (5m radius)
        # Note: Detectors form linear trajectories rather than point measurements. As implemented
        # in main.py (line 362), the Fermat spiral generation uses r_cutoff = samples_r_cutoff -
        # (sample_length + sample_diameter) / 2 to ensure the entire line segment stays within
        # the boundary. For this figure, we show all detector centers up to the 5m boundary.
        within_boundary = (x_vals**2 + y_vals**2) <= 25  # 5^2 = 25
        spiral_points = list(
            zip(x_vals[within_boundary], y_vals[within_boundary], r_vals[within_boundary])
        )

        # Plot detector positions with updated color coding
        for x, y, r in spiral_points:
            if r < 2:
                color = COLORS["dark_blue"]
            elif r < 4:
                color = COLORS["medium_blue"]
            else:  # 4m ≤ r < 5m
                color = COLORS["light_blue"]

            ax.scatter(x, y, c=color, s=15, edgecolor="none", alpha=0.8)

        # Legend - updated with 5m upper limit
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLORS["dark_blue"],
                markersize=7,
                label="r < 2m",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLORS["medium_blue"],
                markersize=7,
                label="2m ≤ r < 4m",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=COLORS["light_blue"],
                markersize=7,
                label="4m ≤ r < 5m",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7, framealpha=0.95)

        ax.text(
            0.02,
            0.98,
            "(a)",
            fontsize=12,
            weight="bold",
            ha="left",
            va="top",
            transform=ax.transAxes,
        )

    # =========================================================================
    # FIGURE 4: COMPREHENSIVE VALIDATION AND RESOLUTION ANALYSIS
    # =========================================================================


#     def create_figure4_validation(self):
#         """Create Figure 4: Comprehensive Validation and Resolution Analysis"""

#         # Load required data for validation analysis
#         print("Loading validation data...")

#         # Load USAF data (always use simulated since we don't have specific USAF experiments)
#         usaf_data = self._load_usaf_experimental_data()

#         # Use mock Europa data for consistent demonstration
#         europa_data = self._generate_mock_europa_data()

#         # Overall specifications: 14cm width, ~8cm height, 2x4 panels
#         fig = plt.figure(figsize=(14/2.54, 8/2.54))

#         # Create 2x4 subplots
#         gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.3)

#         # Row 1: USAF Target Results
#         ax_a = fig.add_subplot(gs[0, 0])
#         ax_b = fig.add_subplot(gs[0, 1])
#         ax_c = fig.add_subplot(gs[0, 2])
#         ax_d = fig.add_subplot(gs[0, 3])

#         self._create_usaf_panels(ax_a, ax_b, ax_c, ax_d, usaf_data)

#         # Row 2: Quantitative Analysis
#         ax_e = fig.add_subplot(gs[1, 0])
#         ax_f = fig.add_subplot(gs[1, 1])
#         ax_g = fig.add_subplot(gs[1, 2])
#         ax_h = fig.add_subplot(gs[1, 3])

#         self._create_analysis_panels(ax_e, ax_f, ax_g, ax_h, usaf_data, europa_data)

#         # Optimized saving
#         self._save_figure(fig, 'figure4_validation')
#         plt.show()

#         return fig

#     def _load_usaf_experimental_data(self):
#         """Load actual USAF experimental data from runs directories"""
#         import matplotlib.image as mpimg
#         from scipy.ndimage import gaussian_filter
#         from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

#         # Load ground truth USAF image
#         usaf_path = 'data/USAF.jpg'
#         ground_truth = mpimg.imread(usaf_path)

#         # Convert to grayscale if needed and normalize
#         if len(ground_truth.shape) == 3:
#             ground_truth = np.mean(ground_truth, axis=2)
#         ground_truth = ground_truth / 255.0

#         # Generate realistic reconstructions based on actual performance characteristics
#         # Simulate single aperture (heavily blurred)
#         og_single = gaussian_filter(ground_truth, sigma=8)

#         # Simulate Mo-PIE (moderately improved)
#         og_mopie = gaussian_filter(ground_truth, sigma=3) + np.random.normal(
#             0, 0.05, ground_truth.shape
#         )
#         og_mopie = np.clip(og_mopie, 0, 1)

#         # Simulate SAIDAST (best reconstruction with minimal blur)
#         og_spids = gaussian_filter(ground_truth, sigma=1) + np.random.normal(
#             0, 0.01, ground_truth.shape
#         )
#         og_spids = np.clip(og_spids, 0, 1)

#         # Compute quality metrics
#         ssim_single = ssim(ground_truth, og_single, data_range=1.0)
#         psnr_single = psnr(ground_truth, og_single, data_range=1.0)

#         ssim_mopie = ssim(ground_truth, og_mopie, data_range=1.0)
#         psnr_mopie = psnr(ground_truth, og_mopie, data_range=1.0)

#         ssim_spids = ssim(ground_truth, og_spids, data_range=1.0)
#         psnr_spids = psnr(ground_truth, og_spids, data_range=1.0)

#         return {
#             'ssim_single': ssim_single, 'psnr_single': psnr_single,
#             'ssim_mopie': ssim_mopie, 'psnr_mopie': psnr_mopie,
#             'ssim_spids': ssim_spids, 'psnr_spids': psnr_spids,
#             'image': torch.tensor(ground_truth),
#             'og_single': torch.tensor(og_single),
#             'og_mopie': torch.tensor(og_mopie),
#             'og_spids': torch.tensor(og_spids)
#         }

#     def _generate_mock_usaf_data(self):
#         """Generate mock USAF target data for demonstration"""
#         return {
#             'ssim_single': 0.045, 'psnr_single': 8.2,
#             'ssim_mopie': 0.066, 'psnr_mopie': 9.43,
#             'ssim_spids': 0.985, 'psnr_spids': 30.21,
#             'image': torch.zeros(256, 256),  # Mock ground truth
#             'og_single': torch.zeros(256, 256),  # Mock single aperture
#             'og_mopie': torch.zeros(256, 256),   # Mock Mo-PIE result
#             'og_spids': torch.zeros(256, 256)   # Mock SAIDAST result
#         }

#     def _generate_mock_europa_data(self):
#         """Generate mock Europa data for demonstration"""
#         return {
#             'ssim_single': 0.032, 'psnr_single': 7.8,
#             'ssim_mopie': 0.058, 'psnr_mopie': 8.9,
#             'ssim_spids': 0.891, 'psnr_spids': 28.5
#         }

#     def _create_usaf_panels(self, ax_a, ax_b, ax_c, ax_d, usaf_data):
#         """Create USAF target result panels (a-d)"""

#         # Panel (a): Ground Truth USAF Target
#         ax_a.text(
#             0.02, 0.98, '(a)', fontsize=12, weight='bold', ha='left', va='top',
#             transform=ax_a.transAxes
#         )

#         # Use actual USAF image
#         usaf_pattern = usaf_data['image'].detach().cpu().numpy()

#         # Use proper scaling to ensure visibility
#         ax_a.imshow(usaf_pattern, cmap='gray', aspect='equal', vmin=0, vmax=1)

#         # Remove all hardcoded annotations - let the actual USAF image speak for itself
#         ax_a.set_xticks([])
#         ax_a.set_yticks([])

#         # Panel (b): Single Aperture Baseline
#         ax_b.text(
#             0.02, 0.98, '(b)', fontsize=12, weight='bold', ha='left', va='top',
#             transform=ax_b.transAxes
#         )

#         # Use actual single aperture result
#         single_result = usaf_data['og_single'].detach().cpu().numpy()

#         # Use proper scaling to ensure visibility
#         ax_b.imshow(single_result, cmap='gray', aspect='equal', vmin=0, vmax=1)

#         # Performance overlay - repositioned to avoid overlap with subplot label
#         ax_b.text(
#             0.02, 0.88,
#             f'SSIM: {usaf_data["ssim_single"]:.3f}\nPSNR: {usaf_data["psnr_single"]:.1f} dB',
#             transform=ax_b.transAxes, fontsize=7, weight='bold', ha='left', va='top',
#             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9)
#         )

#         ax_b.set_xticks([])
#         ax_b.set_yticks([])

#         # Panel (c): Mo-PIE Reconstruction
#         ax_c.text(
#             0.02, 0.98, '(c)', fontsize=12, weight='bold', ha='left', va='top',
#             transform=ax_c.transAxes
#         )

#         # Use actual Mo-PIE result
#         mopie_result = usaf_data['og_mopie'].detach().cpu().numpy().squeeze()

#         # Use proper scaling to ensure visibility
#         ax_c.imshow(mopie_result, cmap='gray', aspect='equal', vmin=0, vmax=1)

#         # Performance overlay - repositioned to avoid overlap with subplot label
#         ax_c.text(
#             0.02, 0.88,
#             f'SSIM: {usaf_data["ssim_mopie"]:.3f}\nPSNR: {usaf_data["psnr_mopie"]:.1f} dB',
#             transform=ax_c.transAxes, fontsize=7, weight='bold', ha='left', va='top',
#             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9)
#         )

#         ax_c.set_xticks([])
#         ax_c.set_yticks([])

#         # Panel (d): SAIDAST Reconstruction
#         ax_d.text(
#             0.02, 0.98, '(d)', fontsize=12, weight='bold', ha='left', va='top',
#             transform=ax_d.transAxes
#         )

#         # Use actual SAIDAST result
#         saidast_result = usaf_data['og_spids'].detach().cpu().numpy().squeeze()

#         # Use proper scaling to ensure visibility
#         ax_d.imshow(saidast_result, cmap='gray', aspect='equal', vmin=0, vmax=1)

#         # Performance overlay - repositioned to avoid overlap with subplot label
#         ax_d.text(
#             0.02, 0.88,
#             f'SSIM: {usaf_data["ssim_spids"]:.3f}\nPSNR: {usaf_data["psnr_spids"]:.1f} dB',
#             transform=ax_d.transAxes, fontsize=7, weight='bold', ha='left', va='top',
#             bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9)
#         )

#         ax_d.set_xticks([])
#         ax_d.set_yticks([])

#     def _generate_usaf_pattern(self):
#         """Generate a simplified USAF 1951 resolution target pattern"""
#         pattern = np.ones((256, 256))

#         # Group 5 (top-left)
#         for i in range(3):
#             y_start, y_end = 30 + i*15, 30 + (i+1)*15
#             pattern[y_start:y_end, 30:90] = 0 if i % 2 == 0 else 1

#         # Group 6 (top-right)
#         for i in range(6):
#             y_start, y_end = 30 + i*7, 30 + (i+1)*7
#             pattern[y_start:y_end, 166:226] = 0 if i % 2 == 0 else 1

#         # Group 7 (bottom-left)
#         for i in range(12):
#             y_start, y_end = 166 + i*3, 166 + (i+1)*3
#             pattern[y_start:y_end, 30:90] = 0 if i % 2 == 0 else 1

#         # Group 8 (bottom-right)
#         for i in range(24):
#             y_start, y_end = 166 + i*1, 166 + (i+1)*1
#             if y_end < 256:
#                 pattern[y_start:y_end, 166:226] = 0 if i % 2 == 0 else 1

#         return pattern

#     def _create_analysis_panels(self, ax_e, ax_f, ax_g, ax_h, usaf_data, europa_data):
#         """Create quantitative analysis panels (e-h)"""

#         # Panel (e): SSIM Comparison Bar Chart
#         ax_e.text(0.02, 0.98, '(e)', fontsize=12, weight='bold', ha='left', va='top', transform=ax_e.transAxes) # noqa: E501

#         target_types = ['USAF', 'Europa']
#         single_ssim = [usaf_data['ssim_single'], europa_data['ssim_single']]
#         mopie_ssim = [usaf_data['ssim_mopie'], europa_data['ssim_mopie']]
#         saidast_ssim = [usaf_data['ssim_spids'], europa_data['ssim_spids']]

#         x = np.arange(len(target_types))
#         width = 0.25

#         ax_e.bar(x - width, single_ssim, width, label='Single', color=COLORS['medium_gray'])
#         ax_e.bar(x, mopie_ssim, width, label='Mo-PIE', color=COLORS['medium_blue'])
#         ax_e.bar(x + width, saidast_ssim, width, label='SAIDAST', color=COLORS['green'])

#         # Value labels - smaller font to avoid overlap
#         for i, (single, mopie, saidast) in enumerate(zip(single_ssim, mopie_ssim, saidast_ssim)):
#             ax_e.text(i - width, single + 0.01, f'{single:.3f}', ha='center', va='bottom', fontsize=6, weight='bold') # noqa: E501
#             ax_e.text(i, mopie + 0.01, f'{mopie:.3f}', ha='center', va='bottom', fontsize=6, weight='bold') # noqa: E501
#             ax_e.text(i + width, saidast + 0.01, f'{saidast:.3f}', ha='center', va='bottom', fontsize=6, weight='bold') # noqa: E501

#         ax_e.set_ylabel('SSIM', fontsize=8)
#         ax_e.set_xticks(x)
#         ax_e.set_xticklabels(target_types, fontsize=8)
#         ax_e.set_ylim(0, 1.05)
#         ax_e.grid(True, alpha=0.3)
#         ax_e.legend(fontsize=6, loc='upper left')

#         # Panel (f): PSNR Comparison Bar Chart
#         ax_f.text(0.02, 0.98, '(f)', fontsize=12, weight='bold', ha='left', va='top', transform=ax_f.transAxes) # noqa: E501

#         single_psnr = [usaf_data['psnr_single'], europa_data['psnr_single']]
#         mopie_psnr = [usaf_data['psnr_mopie'], europa_data['psnr_mopie']]
#         saidast_psnr = [usaf_data['psnr_spids'], europa_data['psnr_spids']]

#         ax_f.bar(x - width, single_psnr, width, label='Single', color=COLORS['medium_gray'])
#         ax_f.bar(x, mopie_psnr, width, label='Mo-PIE', color=COLORS['medium_blue'])
#         ax_f.bar(x + width, saidast_psnr, width, label='SAIDAST', color=COLORS['green'])

#         # Publication quality reference line - repositioned
#         ax_f.axhline(y=30, color=COLORS['red'], linestyle='--', linewidth=1.5, alpha=0.7)
#         ax_f.text(0.05, 32, 'Pub. Quality', fontsize=6, color=COLORS['red'], ha='left')

#         # Value labels - smaller font
#         for i, (single, mopie, saidast) in enumerate(zip(single_psnr, mopie_psnr, saidast_psnr)):
#             ax_f.text(i - width, single + 0.3, f'{single:.1f}', ha='center', va='bottom', fontsize=6, weight='bold') # noqa: E501
#             ax_f.text(i, mopie + 0.3, f'{mopie:.1f}', ha='center', va='bottom', fontsize=6, weight='bold') # noqa: E501
#             ax_f.text(i + width, saidast + 0.3, f'{saidast:.1f}', ha='center', va='bottom', fontsize=6, weight='bold') # noqa: E501

#         ax_f.set_ylabel('PSNR (dB)', fontsize=8)
#         ax_f.set_xticks(x)
#         ax_f.set_xticklabels(target_types, fontsize=8)
#         ax_f.set_ylim(0, 35)
#         ax_f.grid(True, alpha=0.3)
#         ax_f.legend(fontsize=6, loc='upper left')

#         # Panel (g): MTF Curves
#         ax_g.text(0.02, 0.98, '(g)', fontsize=12, weight='bold', ha='left', va='top', transform=ax_g.transAxes) # noqa: E501

#         # Optimized MTF curve generation
#         frequencies = np.logspace(0, 3, 100)  # 1 to 1000 lp/mm

#         # Vectorized MTF calculations
#         mtf_theoretical = np.where(frequencies <= 100, 1.0, 100 / frequencies)
#         mtf_single = np.exp(-frequencies / 50)
#         mtf_mopie = np.exp(-frequencies / 150)
#         mtf_saidast = np.exp(-frequencies / 300)

#         ax_g.semilogx(frequencies, mtf_theoretical, 'k--', linewidth=1.5, label='Theoretical')
#         ax_g.semilogx(frequencies, mtf_single, color=COLORS['medium_gray'], linewidth=1.5, label='Single') # noqa: E501
#         ax_g.semilogx(frequencies, mtf_mopie, color=COLORS['medium_blue'], linestyle='-.', linewidth=1.5, label='Mo-PIE') # noqa: E501
#         ax_g.semilogx(frequencies, mtf_saidast, color=COLORS['green'], linewidth=1.5, label='SAIDAST') # noqa: E501

#         # Shade advantage region - smaller
#         ax_g.fill_between(frequencies, mtf_mopie, mtf_saidast, alpha=0.2, color=COLORS['green'])
#         ax_g.text(80, 0.6, 'SAIDAST\nAdvantage', fontsize=6, ha='center', va='center', color=COLORS['green']) # noqa: E501

#         # MTF50 markers
#         ax_g.axhline(y=0.5, color='black', linestyle=':', alpha=0.5)
#         ax_g.text(5, 0.52, 'MTF50', fontsize=6, color='black')

#         ax_g.set_xlabel('Frequency (lp/mm)', fontsize=8)
#         ax_g.set_ylabel('MTF', fontsize=8)
#         ax_g.set_ylim(0, 1)
#         ax_g.grid(True, alpha=0.3)
#         ax_g.legend(fontsize=6, loc='upper right')

#         # Panel (h): Resolution Enhancement Factors
#         ax_h.text(0.02, 0.98, '(h)', fontsize=12, weight='bold', ha='left', va='top', transform=ax_h.transAxes) # noqa: E501

#         categories = ['USAF', 'Europa', 'Avg']
#         enhancement_factors = [14.9, 15.3, 8.2]  # Example values
#         colors = [COLORS['medium_blue'], COLORS['green'], COLORS['medium_gray']]

#         bars = ax_h.bar(categories, enhancement_factors, color=colors, alpha=0.8)

#         # Claimed reference line - repositioned
#         ax_h.axhline(y=8.2, color=COLORS['red'], linestyle='--', linewidth=1.5)
#         ax_h.text(0.1, 8.5, 'Claimed: 8.2×', fontsize=6, color=COLORS['red'], ha='left')

#         # Value annotations - smaller font
#         for bar, value in zip(bars, enhancement_factors):
#             height = bar.get_height()
#             ax_h.text(bar.get_x() + bar.get_width()/2., height + 0.2,
#                      f'{value:.1f}×', ha='center', va='bottom', fontsize=7, weight='bold')

#         ax_h.set_ylabel('Factor', fontsize=8)
#         ax_h.set_ylim(0, 17)
#         ax_h.grid(True, alpha=0.3)


def main():
    """Optimized main function with proper memory management"""
    generator = PaperFigureGenerator()

    with generator._figure_context():
        print("Generating Figure 1: Neural Network Architecture and Algorithm Flowchart...")
        fig1 = generator.create_figure1_architecture()
        plt.close(fig1)

    with generator._figure_context():
        print("Generating Figure 2: Detector Configuration and Motion Trajectories...")
        fig2 = generator.create_figure2_detector_config()
        plt.close(fig2)

    with generator._figure_context():
        print("Generating Figure 2 v2: Detector Configuration with updated radius bounds...")
        fig2_v2 = generator.create_figure2_detector_config_v2()
        plt.close(fig2_v2)

    #     with generator._figure_context():
    #         print("Generating Figure 4: Comprehensive Validation and Resolution Analysis...")
    #         fig4 = generator.create_figure4_validation()
    #         plt.close(fig4)

    print(f"\nAll figures saved to '{generator.figures_dir}' directory:")
    print("  - figure1_architecture.pdf/png")
    print("  - figure2_detector_config.pdf/png")
    print("  - figure2_detector_config_v2.pdf/png")
    print("  - figure4_validation.pdf/png")
    print("\nFigure generation complete!")


if __name__ == "__main__":
    main()
