"""
Module: mopie.py
Purpose: Motion-aware Ptychographic Iterative Engine (Mo-PIE) implementation for Fourier ptychography
Dependencies: torch, optics, image_utils, matplotlib, numpy, pickle
Main Classes:
    - Mo-PIE: Traditional phase retrieval algorithm extending Telescope class

Description:
    Implements the Mo-PIE algorithm for iterative phase retrieval from multiple
    aperture measurements. Unlike deep learning approaches, Mo-PIE directly
    estimates the complex-valued object and probe through physics-based updates.

Mo-PIE Class (extends Telescope):
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
    - update_cntr(cntr, center_rec, idx): Set current sample position
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

# %% Imports
import matplotlib
import torch
from torch import nn
from torch.nn import functional as F

import prism.core.optics as op
import prism.utils.image_utils as iu


matplotlib.use("Agg")
import pickle

import numpy as np
from matplotlib import pyplot as plt


# %% Utility functions
def replace_mag(tensor, mag):
    """
    Replace the magnitude of a complex tensor with the given magnitude
    :param tensor: The complex tensor
    :param mag: The magnitude tensor
    :return: The complex tensor with the given magnitude
    """
    return tensor / tensor.abs() * mag


# %% Mo-PIE class
class MoPIE(op.Telescope):
    def __init__(
        self,
        n=256,
        r=10,
        is_sum=True,
        sum_pattern=None,
        cropping=True,
        obj_size=None,
        snr=100,
        telescope=None,
        req_grad=False,
        fix_probe=True,
        lr_obj=1,
        lr_probe=1,
        ground_truth=None,
        complex_data=False,
        parallel_update=True,
        single_sample=False,
        blur_image=False,
    ):
        if telescope is not None:
            n = telescope.n
            r = telescope.r
            is_sum = telescope.is_sum
            sum_pattern = telescope.sum_pattern
            cropping = telescope.cropping
            obj_size = telescope.obj_size
            snr = telescope.snr
            blur_image = telescope.blur_image
        super().__init__(n, r, is_sum, sum_pattern, cropping, obj_size, snr, blur_image=blur_image)
        obj = torch.ones((1, 1, obj_size, obj_size))
        obj = iu.crop_pad(obj, n)
        self.obj = nn.Parameter(obj.detach().clone(), requires_grad=req_grad)
        self.probe = nn.Parameter(obj.detach().clone(), requires_grad=req_grad and not fix_probe)
        if ground_truth is None:
            self.register_buffer("ground_truth", obj.detach().clone())
        else:
            self.register_buffer("ground_truth", ground_truth)
        self.register_buffer("support_mask", obj.detach().clone())
        self.register_buffer("req_grad", torch.tensor(req_grad))
        self.register_buffer("fix_probe", torch.tensor(fix_probe))
        self.register_buffer("lr_obj", torch.tensor(lr_obj))
        self.register_buffer("lr_probe", torch.tensor(lr_probe))
        self.register_buffer("cntr", torch.tensor([[0, 0]]))
        self.register_buffer("cntr_rec", torch.tensor([[0, 0]]))
        self.register_buffer("cum_mask", torch.zeros((n, n)))
        self.register_buffer("complex_data", torch.tensor(complex_data))
        self.register_buffer("parallel_update", torch.tensor(parallel_update))
        self.register_buffer("single_sample", torch.tensor(single_sample))
        self.is_meas = False
        self.measurements = torch.tensor(0)
        self.center_idx = 0

        # Create the figure for plotting
        self.fig, self.axs = self.init_plot()

    @property
    def og(self):
        return self.obj

    @og.setter
    def og(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if value.device != self.obj.device:
            value = value.to(self.obj.device)
        if not self.complex_data:
            # value = value.abs()
            value = torch.clamp(value.real, 0, 1)
        else:
            mag, ang = value.abs(), value.angle()
            mag = torch.clamp(mag, 0, 1)
            value = torch.polar(mag, ang)
        value = value * self.support_mask
        self.obj = nn.Parameter(value, requires_grad=self.req_grad.item())

    @property
    def Og(self):
        return iu.fft(self.og)

    @Og.setter
    def Og(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if value.device != self.obj.device:
            value = value.to(self.obj.device)
        self.og = iu.ifft(value)

    @property
    def Pg(self):
        if self.fix_probe:
            return self.mask([0, 0], self.r).to(self.obj.dtype)
        else:
            return self.probe

    @Pg.setter
    def Pg(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if value.device != self.obj.device:
            value = value.to(self.obj.device)
        self.probe = nn.Parameter(
            value, requires_grad=(self.req_grad and not self.fix_probe).item()
        )

    @property
    def Pr(self):
        if self.fix_probe:
            return (
                torch.stack([self.mask(ctr, self.r) for ctr in self.curr_cntr], dim=0)
                .unsqueeze(1)
                .to(self.obj.dtype)
            )
        else:
            # TODO: Implement correct Fourier shift. Previous im_shift implementation was incorrect.
            raise NotImplementedError(
                "Non-fixed probe mode requires proper Fourier shift implementation"
            )

    @property
    def n_Pr(self):
        return self.cntr.size(0)

    @property
    def image(self):
        return self.ground_truth

    @property
    def image_f(self):
        return iu.fft(self.image)

    @property
    def Im(self):
        # Measured intensity
        with torch.no_grad():
            if self.measurements.ndim == 0:
                self.is_meas = True
                output = self.forward(self.image)
                # print(output.size())
                self.is_meas = False
                self.measurements = output.unsqueeze(0)
            elif self.measurements.size(0) == self.center_idx:
                self.is_meas = True
                output = self.forward(self.image)
                # print(output.size())
                self.is_meas = False
                self.measurements = torch.cat([self.measurements, output.unsqueeze(0)], dim=0)
            elif self.measurements.size(0) > self.center_idx:
                output = self.measurements[self.center_idx]
            else:
                raise ValueError("Invalid measurement index")

            return output

    @property
    def Ir(self):
        return self.forward(self.og)

    @property
    def Phi(self):
        return self.Og * self.Pr

    @property
    def phi(self):
        return iu.ifft(self.Phi)

    @property
    def Psi(self):
        if self.single_sample:
            phi = self.phi
            return iu.fft(self.Im * phi / phi.abs())
        else:
            return iu.fft(self.Im * self.phi / self.Ir)

    @property
    def dOg(self):
        return self.lr_obj * self.Pr.conj() / self.Pr.abs().pow(2).max() * (self.Psi - self.Phi)

    def update_obj(self):
        if self.single_sample:
            self.Og = self.Og + self.dOg[self.n_Pr // 2].unsqueeze(0)
        else:
            if self.parallel_update:
                # Parallel update
                self.Og = self.Og + self.dOg.sum(dim=0, keepdim=True)
            else:
                # Serial update
                for i in range(self.n_Pr):
                    self.Og = self.Og + self.dOg[i].unsqueeze(0)

    def update_probe(self):
        if not self.fix_probe:
            self.Pg = self.Pg + self.lr_probe * self.Og.conj() / self.Og.abs().pow(2).max() * (
                self.Psi - self.Phi
            )

    def update_step(self):
        self.update_obj()
        self.update_probe()

    @property
    def curr_cntr(self):
        if self.is_meas:
            return self.cntr
        else:
            return self.center_rec

    @curr_cntr.setter
    def curr_cntr(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        if value.device != self.obj.device:
            value = value.to(self.obj.device)
        self.add_mask(self.cntr)
        self.cntr = value

    @torch.no_grad()
    def add_mask(self, centers=None, r=None):
        if centers is None:
            centers = self.cntr
        if r is None:
            r = self.r
        tmp_mask = self.cum_mask.clone().detach()
        for idx, ctr in enumerate(centers):
            if not self.single_sample or idx == self.n_Pr // 2:
                mask = self.mask(ctr, r)
                tmp_mask = torch.logical_or(tmp_mask, mask)
        self.cum_mask = tmp_mask

    def update_cntr(self, cntr, center_rec, idx):
        self.curr_cntr = cntr
        self.center_rec = center_rec
        self.center_idx = idx

    def forward(self, tensor, centers=None, r=None, is_sum=None, sum_pattern=None):
        if centers is None:
            centers = self.cntr
        if is_sum is None:
            is_sum = self.is_sum
        tensor_f = iu.fft(tensor).squeeze()
        masks = self.Pr
        # In the future add "stupid" sum so that uncropped images can be summed as well
        if self.snr is not None:
            tensor_meas = torch.stack(
                [self.noise(iu.ifft(tensor_f * pr.squeeze()).abs(), self.is_meas) for pr in masks],
                dim=0,
            ).unsqueeze(1)
        else:
            tensor_meas = torch.stack(
                [iu.ifft(tensor_f * pr.squeeze()).abs() for pr in masks], dim=0
            ).unsqueeze(1)
        if is_sum:
            tensor_meas = tensor_meas.pow(2).mean(dim=0, keepdim=True).sqrt()
        if self.blur_image:
            tensor_meas = F.conv2d(
                tensor_meas, self.blur_kernel, padding=self.blur_kernel.size(-1) // 2
            )
        return tensor_meas

    def init_plot(self):
        fig, axs = plt.subplots(2, 2, figsize=(5, 5))
        axs[1][1].imshow(
            iu.crop_pad(self.image, self.obj_size).squeeze().abs().cpu().numpy(), cmap="gray"
        )
        axs[1][1].set_title("Target")
        return fig, axs

    def update_plot(self):
        for idx, ax in enumerate(self.axs.flatten()):
            if idx != 3:
                ax.cla()
        self.axs[0][0].imshow(
            iu.crop_pad(self.Im, self.obj_size).squeeze().abs().detach().cpu().numpy(), cmap="gray"
        )
        self.axs[0][1].imshow(
            iu.crop_pad(self.og, self.obj_size).squeeze().abs().detach().cpu().numpy(), cmap="gray"
        )
        self.axs[1][0].imshow(self.image_f.abs().log10().squeeze().cpu().numpy(), cmap="gray")
        zero_mask = np.zeros((self.n, self.n))
        self.axs[1][0].imshow(
            np.stack(
                [
                    zero_mask,
                    self.cum_mask.cpu().squeeze().numpy(),
                    zero_mask,
                    self.cum_mask.cpu().squeeze().numpy(),
                ],
                axis=2,
            )
        )
        centers_map = (
            torch.stack([self.mask(ctr, self.r) for ctr in self.curr_cntr], dim=0)
            .sum(dim=0)
            .squeeze()
            > 0
        )
        self.axs[1][0].imshow(
            np.stack(
                [centers_map.cpu().numpy(), zero_mask, zero_mask, centers_map.cpu().numpy()], axis=2
            )
        )

        self.axs[0][0].set_title("New Measurement")
        self.axs[0][1].set_title("Current reconstruction")
        self.axs[1][0].set_title("K-space")

        plt.tight_layout()
        plt.show()
        plt.draw()
        plt.pause(0.1)
        plt.show()

    def errors(self):
        rmse = iu.compare_rmse(self.og.abs(), self.image.abs(), self.obj_size)
        ssim = iu.ssim_skimage(self.og.abs(), self.image.abs(), self.obj_size)
        psnr = iu.psnr(self.og.abs(), self.image.abs(), self.obj_size)
        return rmse, ssim, psnr

    @property
    def figure(self):
        # Serialize the figure to a byte stream
        fig_data = pickle.dumps(self.fig)

        # Deserialize the byte stream to create a new figure
        copied_fig = pickle.loads(fig_data)
        return copied_fig
