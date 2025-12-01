"""
Module: prism.core.algorithms.mopie
Purpose: Motion-aware Ptychographic Iterative Engine (Mo-PIE) implementation for Fourier ptychography
Dependencies: torch, prism.core.instruments, prism.utils.{image,transforms,metrics}, matplotlib, numpy, pickle

Main Classes:
    - MoPIE: Motion-aware phase retrieval algorithm using composition with Telescope

Description:
    Implements the Mo-PIE algorithm for iterative phase retrieval from multiple
    aperture measurements. Unlike deep learning approaches, Mo-PIE directly
    estimates the complex-valued object and probe through physics-based updates.

MoPIE Class (uses Telescope via composition):
    Core Properties:
    - obj (og): Complex-valued object estimate (Parameter)
    - probe (Pg): Complex-valued probe/aperture function (Parameter, optional)
    - ground_truth (image): True object for error calculation
    - cum_mask: Accumulated k-space coverage
    - measurements (Im): Cached intensity measurements
    - fix_probe: If True, probe is fixed as known aperture mask

    Algorithm Steps:
    1. Forward model:
        Phi = Og * Pr  (exit wave = object × probe in k-space)
        phi = IFFT(Phi)  (propagate to detector)
        Ir = |phi|²  (predicted intensity)

    2. Measurement constraint:
        Psi = FFT(sqrt(Im) * exp(i*angle(phi)))  (replace magnitude, keep phase)

    3. Updates:
        dOg = lr_obj * Pr* * (Psi - Phi) / max(|Pr|²)
        Og += dOg  (or sum over samples if parallel_update=True)

        dPg = lr_probe * Og* * (Psi - Phi) / max(|Og|²)  (if not fix_probe)

    Properties (computed on-the-fly):
    - Og: Object in k-space (FFT domain)
    - Pg: Probe at center (k-space)
    - Pr: Shifted probes at current sample positions
    - Phi: Exit wave (Og * Pr)
    - phi: Exit wave in real space
    - Psi: Updated exit wave after measurement constraint
    - Im: Measured intensity (cached, computed once per sample)
    - Ir: Reconstructed intensity
    - dOg: Object update gradient

    Key Methods:
    - update_obj(): Apply object update (parallel or serial)
    - update_probe(): Apply probe update (if not fix_probe)
    - update_step(): Combined object and probe update
    - update_cntr(center, center_rec, idx): Set current sample position
    - errors(): Compute RMSE, SSIM, PSNR vs ground truth
    - init_plot(): Create figure for visualization
    - update_plot(): Refresh visualization with current state

Implementation Notes:
    - Object constrained to support mask (zero outside)
    - Real vs complex: Can enforce real-valued if complex_data=False
    - Measurement caching: Im computed once per sample position
    - Single sample mode: Use only center of line for faster updates
    - Parallel update: Sum gradients from all line samples
    - Serial update: Apply gradients sequentially (slower but more stable)

    IMPORTANT: Fixed probe mode (fix_probe=True) is recommended as probe
    is typically known from aperture design. Non-fixed probe requires
    proper Fourier shift implementation.
"""

from __future__ import annotations

import pickle
from typing import Optional, Tuple

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.nn import functional as F

# Import from refactored prism package
from prism.core.instruments import Telescope, TelescopeConfig
from prism.models.noise import ShotNoise
from prism.utils.image import crop_pad
from prism.utils.metrics import compute_rmse, compute_ssim
from prism.utils.metrics import psnr as compute_psnr
from prism.utils.transforms import fft, ifft


matplotlib.use("Agg")


def replace_mag(tensor: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
    """
    Replace the magnitude of a complex tensor with the given magnitude.

    Args:
        tensor: Complex tensor whose magnitude will be replaced
        mag: New magnitude tensor

    Returns:
        Complex tensor with replaced magnitude but original phase
    """
    return tensor / tensor.abs() * mag


class MoPIE(nn.Module):
    """
    Motion-aware Ptychographic Iterative Engine (Mo-PIE) algorithm.

    Physics-based iterative phase retrieval algorithm that alternately updates
    object and probe estimates from multiple aperture measurements.

    Uses composition with Telescope instead of inheritance for cleaner separation
    of concerns. The telescope is used for aperture mask generation only.
    """

    def __init__(
        self,
        n: int = 256,
        r: float = 10,
        is_sum: bool = True,
        sum_pattern: Optional[torch.Tensor] = None,
        cropping: bool = True,
        obj_size: Optional[int] = None,
        snr: float = 100,
        telescope: Optional[Telescope] = None,
        req_grad: bool = False,
        fix_probe: bool = True,
        lr_obj: float = 1.0,
        lr_probe: float = 1.0,
        ground_truth: Optional[torch.Tensor] = None,
        complex_data: bool = False,
        parallel_update: bool = True,
        single_sample: bool = False,
        blur_image: bool = False,
    ):
        """
        Initialize Mo-PIE algorithm.

        Args:
            n: Image size in pixels
            r: Aperture radius in pixels
            is_sum: Whether to sum line samples
            sum_pattern: Custom summation pattern
            cropping: Whether to crop output to object size
            obj_size: Size of object region
            snr: Signal-to-noise ratio in dB (None for noiseless)
            telescope: Existing Telescope instance to use for mask generation
            req_grad: Whether to compute gradients (for gradient-based Mo-PIE)
            fix_probe: Keep probe fixed as known aperture
            lr_obj: Learning rate for object updates
            lr_probe: Learning rate for probe updates
            ground_truth: Ground truth image for metrics
            complex_data: Allow complex-valued reconstructions
            parallel_update: Sum gradients from all samples vs serial
            single_sample: Use only center sample of line
            blur_image: Apply blur to measurements
        """
        super().__init__()

        # Copy parameters from telescope if provided
        if telescope is not None:
            n = telescope.config.n_pixels
            r = int(telescope._telescope_config.aperture_radius_pixels)
            # Use defaults for is_sum, cropping since unified Telescope doesn't have these
            # They are measurement-specific, not telescope-specific
            obj_size = obj_size  # Keep provided value

        # Store telescope parameters as instance attributes
        self._n = n
        self._r = float(r)
        self._is_sum = is_sum
        self.sum_pattern = sum_pattern
        self.cropping = cropping
        self.obj_size = obj_size
        self._snr = snr
        self.blur_image = blur_image

        # Create telescope for mask generation using composition
        if telescope is not None:
            self.telescope = telescope
        else:
            config = TelescopeConfig(
                n_pixels=n,
                aperture_radius_pixels=r,
                snr=snr if snr is not None else None,
            )
            self.telescope = Telescope(config)

        # Create noise model
        self.noise: nn.Module = ShotNoise(snr) if snr is not None else nn.Identity()

        # Create blur kernel
        self.register_buffer("blur_kernel", torch.ones(1, 1, 15, 15) / 15**2)

        # Initialize object estimate (uniform initial guess)
        # obj_size is ensured to be int by this point
        assert obj_size is not None, "obj_size must be provided"
        obj = torch.ones((1, 1, obj_size, obj_size))
        obj = crop_pad(obj, n)

        # Object and probe as learnable parameters
        self.obj = nn.Parameter(obj.detach().clone(), requires_grad=req_grad)
        self.probe = nn.Parameter(obj.detach().clone(), requires_grad=req_grad and not fix_probe)

        # Register buffers (non-trainable state)
        if ground_truth is None:
            self.register_buffer("ground_truth", obj.detach().clone())
        else:
            self.register_buffer("ground_truth", ground_truth)

        self.register_buffer("support_mask", obj.detach().clone())  # Object support constraint
        self.register_buffer("req_grad", torch.tensor(req_grad))
        self.register_buffer("fix_probe", torch.tensor(fix_probe))
        self.register_buffer("lr_obj", torch.tensor(lr_obj))
        self.register_buffer("lr_probe", torch.tensor(lr_probe))
        self.register_buffer("center", torch.tensor([[0, 0]]))  # Current sample center
        self.register_buffer("cntr_rec", torch.tensor([[0, 0]]))  # Reconstruction center
        self.register_buffer("cum_mask", torch.zeros((n, n)))  # Accumulated k-space coverage
        self.register_buffer("complex_data", torch.tensor(complex_data))
        self.register_buffer("parallel_update", torch.tensor(parallel_update))
        self.register_buffer("single_sample", torch.tensor(single_sample))

        # Measurement caching state
        self.is_meas = False
        self.measurements = torch.tensor(0)
        self.center_idx = 0

        # Create visualization figure
        self.fig, self.axs = self.init_plot()

    # Type narrowing properties for instance attributes (formerly from Telescope)
    @property
    def n_int(self) -> int:
        """Image size as int."""
        return self._n

    @property
    def r_float(self) -> float:
        """Aperture radius as float."""
        return self._r

    @property
    def is_sum_bool(self) -> bool:
        """Sum mode as bool."""
        return self._is_sum

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio."""
        return self._snr

    @property
    def blur_kernel_tensor(self) -> Tensor:
        """Blur kernel as Tensor (type-safe accessor for buffer)."""
        assert isinstance(self.blur_kernel, Tensor)
        return self.blur_kernel

    def mask(self, center: list, r: Optional[float] = None) -> Tensor:
        """Generate aperture mask at specified center using telescope.

        Args:
            center: Center position [y, x]
            r: Aperture radius (uses default if None)

        Returns:
            Boolean mask tensor
        """
        return self.telescope.generate_aperture_mask(center, radius=r)

    # Type narrowing properties for buffer attributes
    @property
    def support_mask_tensor(self) -> Tensor:
        """Support mask as Tensor (type-safe accessor for buffer)."""
        assert isinstance(self.support_mask, Tensor)
        return self.support_mask

    @property
    def req_grad_bool(self) -> bool:
        """Requires grad flag as bool (type-safe accessor for buffer)."""
        return bool(self.req_grad.item())  # type: ignore[union-attr,operator]

    @property
    def fix_probe_bool(self) -> bool:
        """Fix probe flag as bool (type-safe accessor for buffer)."""
        return bool(self.fix_probe.item())  # type: ignore[union-attr,operator]

    @property
    def lr_obj_float(self) -> float:
        """Object learning rate as float (type-safe accessor for buffer)."""
        return float(self.lr_obj.item())  # type: ignore[union-attr,operator]

    @property
    def lr_probe_float(self) -> float:
        """Probe learning rate as float (type-safe accessor for buffer)."""
        return float(self.lr_probe.item())  # type: ignore[union-attr,operator]

    @property
    def cntr_tensor(self) -> Tensor:
        """Current center as Tensor (type-safe accessor for buffer)."""
        assert isinstance(self.center, Tensor)
        return self.center

    @property
    def cntr_rec_tensor(self) -> Tensor:
        """Reconstruction center as Tensor (type-safe accessor for buffer)."""
        assert isinstance(self.cntr_rec, Tensor)
        return self.cntr_rec

    @property
    def cum_mask_tensor(self) -> Tensor:
        """Cumulative mask as Tensor (type-safe accessor for buffer)."""
        assert isinstance(self.cum_mask, Tensor)
        return self.cum_mask

    @property
    def complex_data_bool(self) -> bool:
        """Complex data flag as bool (type-safe accessor for buffer)."""
        return bool(self.complex_data.item())  # type: ignore[union-attr,operator]

    @property
    def parallel_update_bool(self) -> bool:
        """Parallel update flag as bool (type-safe accessor for buffer)."""
        return bool(self.parallel_update.item())  # type: ignore[union-attr,operator]

    @property
    def single_sample_bool(self) -> bool:
        """Single sample flag as bool (type-safe accessor for buffer)."""
        return bool(self.single_sample.item())  # type: ignore[union-attr,operator]

    @property
    def ground_truth_tensor(self) -> Tensor:
        """Ground truth as Tensor (type-safe accessor for buffer)."""
        assert isinstance(self.ground_truth, Tensor)
        return self.ground_truth

    @property
    def og(self) -> torch.Tensor:
        """Object estimate in real space."""
        return self.obj

    @og.setter
    def og(self, value: torch.Tensor) -> None:
        """
        Set object estimate with constraints.

        Applies real-valued or magnitude constraints and support mask.
        """
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if value.device != self.obj.device:
            value = value.to(self.obj.device)

        # Apply data constraints
        if not self.complex_data_bool:
            # Real-valued constraint: clamp to [0, 1]
            value = torch.clamp(value.real, 0, 1)
        else:
            # Complex-valued: clamp magnitude to [0, 1]
            mag, ang = value.abs(), value.angle()
            mag = torch.clamp(mag, 0, 1)
            value = torch.polar(mag, ang)

        # Apply support constraint (zero outside object region)
        value = value * self.support_mask_tensor
        self.obj = nn.Parameter(value, requires_grad=self.req_grad_bool)

    @property
    def Og(self) -> torch.Tensor:
        """Object estimate in Fourier (k-space) domain."""
        return fft(self.og)

    @Og.setter
    def Og(self, value: torch.Tensor) -> None:
        """Set object in k-space (automatically transforms to real space)."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if value.device != self.obj.device:
            value = value.to(self.obj.device)
        self.og = ifft(value)

    @property
    def object_guess(self) -> torch.Tensor:
        """Object estimate in Fourier (k-space) domain (alias for Og)."""
        return self.Og

    @object_guess.setter
    def object_guess(self, value: torch.Tensor) -> None:
        """Set object in k-space (alias for Og)."""
        self.Og = value

    @property
    def Pg(self) -> torch.Tensor:
        """Probe estimate at center in k-space."""
        if self.fix_probe_bool:
            # Fixed probe: use known aperture mask
            return self.mask([0, 0], self.r_float).to(self.obj.dtype)
        else:
            # Learnable probe
            return self.probe

    @Pg.setter
    def Pg(self, value: torch.Tensor) -> None:
        """Set probe estimate."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if value.device != self.obj.device:
            value = value.to(self.obj.device)
        requires_grad = self.req_grad_bool and not self.fix_probe_bool
        self.probe = nn.Parameter(value, requires_grad=requires_grad)

    @property
    def probe_guess(self) -> torch.Tensor:
        """Probe estimate at center in k-space (alias for Pg)."""
        return self.Pg

    @probe_guess.setter
    def probe_guess(self, value: torch.Tensor) -> None:
        """Set probe estimate (alias for Pg)."""
        self.Pg = value

    @property
    def Pr(self) -> torch.Tensor:
        """
        Shifted probe functions at current sample positions.

        Returns stack of probes centered at each position in curr_center.
        Shape: [n_samples, 1, H, W]

        For fixed probe mode, generates aperture masks at each position.
        For non-fixed (learnable) probe mode, shifts the probe using torch.roll()
        to translate it in k-space to each sample position.
        """
        if self.fix_probe_bool:
            return (
                torch.stack([self.mask(center, self.r_float) for center in self.curr_center], dim=0)
                .unsqueeze(1)
                .to(device=self.obj.device, dtype=self.obj.dtype)
            )
        else:
            # Non-fixed probe: shift learnable probe to each sample position
            # Using torch.roll for circular shift in k-space (equivalent to
            # translation by integer pixel amounts)
            #
            # Note: self.Pg has shape (1, 1, H, W), we need to squeeze to (H, W) before
            # stacking so that the output matches fixed probe mode shape [n_samples, 1, H, W]
            probe_2d = self.Pg.squeeze()  # (H, W)
            shifted_probes = []
            for center in self.curr_center:
                # Convert center to integer pixel shifts
                shift_y = int(center[0].item())
                shift_x = int(center[1].item())

                # Roll the probe to the target position
                # Negative shifts because we want to move the probe center TO the position
                shifted = torch.roll(probe_2d, shifts=(-shift_y, -shift_x), dims=(-2, -1))
                shifted_probes.append(shifted)

            return torch.stack(shifted_probes, dim=0).unsqueeze(1).to(self.obj.dtype)

    @property
    def probe_recovered(self) -> torch.Tensor:
        """Shifted probe functions at current sample positions (alias for Pr)."""
        return self.Pr

    @property
    def n_Pr(self) -> int:
        """Number of probe positions."""
        return int(self.cntr_tensor.size(0))

    @property
    def image(self) -> torch.Tensor:
        """Ground truth image."""
        return self.ground_truth_tensor

    @property
    def image_f(self) -> torch.Tensor:
        """Ground truth in Fourier domain."""
        return fft(self.image)

    @property
    def Im(self) -> torch.Tensor:
        """
        Measured intensity (cached).

        Computes measurement from ground truth once per sample position
        and caches for efficiency.
        """
        with torch.no_grad():
            if self.measurements.ndim == 0:
                # First measurement: initialize cache
                self.is_meas = True
                output = self.forward(self.image)
                self.is_meas = False
                self.measurements = output.unsqueeze(0)
            elif self.measurements.size(0) == self.center_idx:
                # New sample position: compute and cache
                self.is_meas = True
                output = self.forward(self.image)
                self.is_meas = False
                self.measurements = torch.cat([self.measurements, output.unsqueeze(0)], dim=0)
            elif self.measurements.size(0) > self.center_idx:
                # Return cached measurement
                output = self.measurements[self.center_idx]
            else:
                raise ValueError("Invalid measurement index")

            return output

    @property
    def intensity_measured(self) -> torch.Tensor:
        """Measured intensity (cached) (alias for Im)."""
        return self.Im

    @property
    def Ir(self) -> torch.Tensor:
        """Reconstructed intensity from current object estimate."""
        return self.forward(self.og)

    @property
    def intensity_recovered(self) -> torch.Tensor:
        """Reconstructed intensity from current object estimate (alias for Ir)."""
        return self.Ir

    @property
    def Phi(self) -> torch.Tensor:
        """
        Exit wave in k-space.

        Product of object and probe in Fourier domain.
        """
        return self.Og * self.Pr

    @property
    def phase(self) -> torch.Tensor:
        """Exit wave in k-space (alias for Phi)."""
        return self.Phi

    @property
    def phi(self) -> torch.Tensor:
        """Exit wave in real space."""
        return ifft(self.Phi)

    @property
    def Psi(self) -> torch.Tensor:
        """
        Updated exit wave after applying measurement constraint.

        Replaces magnitude of exit wave with measured intensity,
        preserving phase information.
        """
        if self.single_sample:
            phi = self.phi
            return fft(self.Im * phi / phi.abs())
        else:
            return fft(self.Im * self.phi / self.Ir)

    @property
    def exit_wave(self) -> torch.Tensor:
        """Updated exit wave after applying measurement constraint (alias for Psi)."""
        return self.Psi

    @property
    def dOg(self) -> torch.Tensor:
        """
        Object update gradient in k-space.

        Computed from difference between measured and predicted exit waves,
        scaled by learning rate and normalized by probe power.
        """
        return (
            self.lr_obj_float * self.Pr.conj() / self.Pr.abs().pow(2).max() * (self.Psi - self.Phi)
        )

    def update_obj(self) -> None:
        """
        Apply object update step.

        Updates object estimate using gradient dOg. Supports three modes:
        - single_sample: Use only center sample of line
        - parallel_update: Sum gradients from all samples
        - serial: Apply gradients sequentially
        """
        if self.single_sample_bool:
            # Use only center sample
            self.Og = self.Og + self.dOg[self.n_Pr // 2].unsqueeze(0)
        else:
            if self.parallel_update_bool:
                # Parallel update: sum all gradients
                self.Og = self.Og + self.dOg.sum(dim=0, keepdim=True)
            else:
                # Serial update: apply sequentially
                for i in range(self.n_Pr):
                    self.Og = self.Og + self.dOg[i].unsqueeze(0)

    def update_probe(self) -> None:
        """
        Apply probe update step (if not fix_probe).

        Updates probe estimate using gradient similar to object update.
        """
        if not self.fix_probe_bool:
            self.Pg = self.Pg + self.lr_probe_float * self.Og.conj() / self.Og.abs().pow(
                2
            ).max() * (self.Psi - self.Phi)

    def update_step(self) -> None:
        """
        Perform one Mo-PIE iteration.

        Alternately updates object and probe estimates.
        """
        self.update_obj()
        self.update_probe()

    @property
    def curr_center(self) -> torch.Tensor:
        """Current center positions (measurement or reconstruction)."""
        if self.is_meas:
            return self.cntr_tensor
        else:
            return self.cntr_rec_tensor

    @curr_center.setter
    def curr_center(self, value: torch.Tensor) -> None:
        """Set current centers and update k-space coverage mask."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if value.device != self.obj.device:
            value = value.to(self.obj.device)
        self.add_mask(self.cntr_tensor)
        self.center = value

    @torch.no_grad()
    def add_mask(self, centers: Optional[torch.Tensor] = None, r: Optional[float] = None) -> None:
        """
        Update cumulative k-space coverage mask.

        Args:
            centers: Sample center positions
            r: Aperture radius
        """
        if centers is None:
            centers = self.cntr_tensor
        if r is None:
            r = self.r_float

        tmp_mask = self.cum_mask_tensor.clone().detach()
        target_device = tmp_mask.device
        for idx, center in enumerate(centers):
            if not self.single_sample_bool or idx == self.n_Pr // 2:
                mask = self.mask(center, r)
                # Ensure mask is on the same device as tmp_mask
                if mask.device != target_device:
                    mask = mask.to(target_device)
                tmp_mask = torch.logical_or(tmp_mask, mask)
        self.cum_mask = tmp_mask

    def update_cntr(self, center: torch.Tensor, center_rec: torch.Tensor, idx: int) -> None:
        """
        Update current sample position.

        Args:
            center: Measurement center positions
            center_rec: Reconstruction center positions
            idx: Sample index for measurement caching
        """
        self.curr_center = center
        self.center_rec = center_rec
        self.center_idx = idx

    def forward(  # type: ignore[override]
        self,
        tensor: torch.Tensor,
        centers: Optional[torch.Tensor] = None,
        r: Optional[float] = None,
        is_sum: Optional[bool] = None,
        sum_pattern: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward model: simulate telescope measurement.

        Applies probe masks in k-space, transforms to real space,
        takes magnitude, and optionally adds noise.

        Args:
            tensor: Input image
            centers: Sample center positions
            r: Aperture radius
            is_sum: Whether to sum line samples
            sum_pattern: Custom summation pattern

        Returns:
            Simulated measurement intensity
        """
        if centers is None:
            centers = self.cntr_tensor
        if is_sum is None:
            is_sum = self.is_sum_bool

        # Apply aperture masks in k-space
        tensor_f = fft(tensor).squeeze()
        masks = self.Pr

        # Transform to real space and compute intensity
        if self.snr is not None:
            # Add shot noise
            tensor_meas = torch.stack(
                [self.noise(ifft(tensor_f * pr.squeeze()).abs(), self.is_meas) for pr in masks],
                dim=0,
            ).unsqueeze(1)
        else:
            # No noise
            tensor_meas = torch.stack(
                [ifft(tensor_f * pr.squeeze()).abs() for pr in masks], dim=0
            ).unsqueeze(1)

        # Sum line samples if requested
        if is_sum:
            tensor_meas = tensor_meas.pow(2).mean(dim=0, keepdim=True).sqrt()

        # Apply blur if requested
        if self.blur_image:
            blur_kernel = self.blur_kernel_tensor
            tensor_meas = F.conv2d(tensor_meas, blur_kernel, padding=blur_kernel.size(-1) // 2)

        return tensor_meas

    def init_plot(self) -> Tuple[plt.Figure, np.ndarray]:
        """
        Initialize visualization figure.

        Returns:
            Figure and axes array for plotting
        """
        fig, axs = plt.subplots(2, 2, figsize=(5, 5))
        axs[1][1].imshow(
            crop_pad(self.image, self.obj_size).squeeze().abs().cpu().numpy(), cmap="gray"
        )
        axs[1][1].set_title("Target")
        return fig, axs

    def update_plot(self) -> None:
        """Update visualization with current reconstruction state."""
        # Clear all axes except target (bottom-right)
        for idx, ax in enumerate(self.axs.flatten()):
            if idx != 3:
                ax.cla()

        # Plot new measurement
        self.axs[0][0].imshow(
            crop_pad(self.Im, self.obj_size).squeeze().abs().detach().cpu().numpy(), cmap="gray"
        )

        # Plot current reconstruction
        self.axs[0][1].imshow(
            crop_pad(self.og, self.obj_size).squeeze().abs().detach().cpu().numpy(), cmap="gray"
        )

        # Plot k-space with coverage mask
        self.axs[1][0].imshow(self.image_f.abs().log10().squeeze().cpu().numpy(), cmap="gray")
        n = self.n_int
        zero_mask = np.zeros((n, n))
        cum_mask_np = self.cum_mask_tensor.cpu().squeeze().numpy()
        self.axs[1][0].imshow(np.stack([zero_mask, cum_mask_np, zero_mask, cum_mask_np], axis=2))

        # Overlay current sample positions
        centers_map = (
            torch.stack([self.mask(center, self.r_float) for center in self.curr_center], dim=0)
            .sum(dim=0)
            .squeeze()
            > 0
        )
        self.axs[1][0].imshow(
            np.stack(
                [centers_map.cpu().numpy(), zero_mask, zero_mask, centers_map.cpu().numpy()], axis=2
            )
        )

        # Set titles
        self.axs[0][0].set_title("New Measurement")
        self.axs[0][1].set_title("Current reconstruction")
        self.axs[1][0].set_title("K-space")

        plt.tight_layout()
        plt.show()
        plt.draw()
        plt.pause(0.1)
        plt.show()

    def errors(self) -> Tuple[float, float, float]:
        """
        Compute reconstruction error metrics.

        Returns:
            Tuple of (RMSE, SSIM, PSNR) comparing reconstruction to ground truth
        """
        rmse = compute_rmse(self.og.abs(), self.image.abs(), self.obj_size)
        ssim = compute_ssim(self.og.abs(), self.image.abs(), self.obj_size)
        psnr = compute_psnr(self.og.abs(), self.image.abs(), self.obj_size)
        return rmse, ssim, psnr

    @property
    def figure(self) -> plt.Figure:
        """
        Get a copy of the current figure.

        Returns:
            Deep copy of visualization figure
        """
        # Serialize and deserialize to create deep copy
        fig_data = pickle.dumps(self.fig)
        copied_fig: plt.Figure = pickle.loads(fig_data)  # type: ignore[assignment]
        return copied_fig
