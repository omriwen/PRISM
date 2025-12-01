"""Integration tests for microscope lens forward model.

These tests verify end-to-end integration of the unified forward model
with the Microscope class, including:

1. Default working distance configuration
2. Custom working distance and regime selection
3. Manual regime override via config
4. Realistic imaging scenarios (USAF target)

Note: Some tests are marked with pytest.skip until the unified forward
model integration is complete (Phase 3 tasks 3.2-3.4).
"""

import pytest
import torch

from prism.core.instruments.microscope import Microscope, MicroscopeConfig


class TestMicroscopeLensIntegration:
    """Integration tests for lens forward model with Microscope class."""

    @pytest.mark.skipif(
        not hasattr(Microscope, "forward_model"),
        reason="Unified forward model not yet integrated (Phase 3)",
    )
    def test_scenario_with_default_working_distance(self):
        """Default working distance should place object at focal plane.

        When working_distance is not specified (None), it should default
        to the objective focal length, resulting in auto-selection of
        the SIMPLIFIED forward model regime.
        """
        from prism.core.optics import ForwardModelRegime

        config = MicroscopeConfig(
            numerical_aperture=0.9,
            magnification=100.0,  # 100x for Nyquist-valid sampling
            wavelength=532e-9,
            n_pixels=128,
            pixel_size=6.5e-6,
            # working_distance=None (default) -> should be f_objective
        )
        mic = Microscope(config)

        # Should auto-select SIMPLIFIED since object at focal plane
        assert mic.forward_model.selected_regime == ForwardModelRegime.SIMPLIFIED, (
            f"Expected SIMPLIFIED with default working distance, "
            f"got {mic.forward_model.selected_regime}"
        )

        # Working distance should equal objective focal length
        expected_wd = config.tube_lens_focal / config.magnification
        assert mic.working_distance == pytest.approx(expected_wd), (
            f"Working distance {mic.working_distance * 1e3:.3f} mm should equal "
            f"f_objective {expected_wd * 1e3:.3f} mm"
        )

    @pytest.mark.skipif(
        not hasattr(Microscope, "forward_model"),
        reason="Unified forward model not yet integrated (Phase 3)",
    )
    def test_scenario_with_custom_working_distance(self):
        """Custom working distance should trigger FULL regime when defocused.

        When object is beyond the focal plane by more than the threshold,
        the FULL forward model regime should be auto-selected.
        """
        from prism.core.optics import ForwardModelRegime

        f_obj = 0.2 / 100.0  # 2mm for 100x

        config = MicroscopeConfig(
            numerical_aperture=0.9,
            magnification=100.0,
            wavelength=532e-9,
            n_pixels=128,
            pixel_size=6.5e-6,
            working_distance=f_obj * 1.5,  # 50% beyond focal plane
        )
        mic = Microscope(config)

        # Should auto-select FULL due to significant defocus (50% > 1% threshold)
        assert mic.forward_model.selected_regime == ForwardModelRegime.FULL, (
            f"Expected FULL regime for 50% defocus, got {mic.forward_model.selected_regime}"
        )

        # Defocus parameter should be approximately 0.5
        assert mic.forward_model.defocus_parameter == pytest.approx(0.5, rel=0.01), (
            f"Defocus parameter should be ~0.5, got {mic.forward_model.defocus_parameter}"
        )

    @pytest.mark.skipif(
        not hasattr(Microscope, "forward_model"),
        reason="Unified forward model not yet integrated (Phase 3)",
    )
    def test_manual_regime_override_to_full(self):
        """Manual regime override should force FULL model even at focus."""
        from prism.core.optics import ForwardModelRegime

        f_obj = 0.2 / 100.0

        config = MicroscopeConfig(
            numerical_aperture=0.9,
            magnification=100.0,
            wavelength=532e-9,
            n_pixels=128,
            pixel_size=6.5e-6,
            working_distance=f_obj,  # At focus (would auto-select SIMPLIFIED)
            forward_model_regime="full",  # Force FULL
        )
        mic = Microscope(config)

        assert mic.forward_model.selected_regime == ForwardModelRegime.FULL, (
            f"Manual override to FULL should be respected, got {mic.forward_model.selected_regime}"
        )

    @pytest.mark.skipif(
        not hasattr(Microscope, "forward_model"),
        reason="Unified forward model not yet integrated (Phase 3)",
    )
    def test_manual_regime_override_to_simplified(self):
        """Manual regime override should force SIMPLIFIED model even when defocused."""
        from prism.core.optics import ForwardModelRegime

        f_obj = 0.2 / 100.0

        config = MicroscopeConfig(
            numerical_aperture=0.9,
            magnification=100.0,
            wavelength=532e-9,
            n_pixels=128,
            pixel_size=6.5e-6,
            working_distance=f_obj * 2.0,  # 100% defocus (would auto-select FULL)
            forward_model_regime="simplified",  # Force SIMPLIFIED
        )
        mic = Microscope(config)

        assert mic.forward_model.selected_regime == ForwardModelRegime.SIMPLIFIED, (
            f"Manual override to SIMPLIFIED should be respected, "
            f"got {mic.forward_model.selected_regime}"
        )

    def test_usaf_target_imaging_produces_valid_output(self):
        """Test with realistic bar pattern (USAF-like target).

        This test verifies basic imaging functionality with a structured
        target, independent of the unified forward model.
        """
        config = MicroscopeConfig(
            numerical_aperture=0.9,
            magnification=100.0,
            wavelength=532e-9,
            n_pixels=256,
            pixel_size=6.5e-6,
        )
        mic = Microscope(config)

        # Create simple bar pattern (like USAF target elements)
        target = torch.zeros(256, 256, dtype=torch.complex64)
        for i in range(0, 256, 40):
            target[:, i : i + 20] = 1.0

        output = mic.forward(target)

        # Basic validity checks
        assert output.shape == (256, 256), "Output shape should match input"
        assert (output >= 0).all(), "Output intensity should be non-negative"
        assert output.max() > 0, "Output should have non-zero intensity"
        assert torch.isfinite(output).all(), "Output should be finite"

    def test_bar_pattern_contrast_decreases_near_resolution_limit(self):
        """Fine bars near resolution limit should show reduced contrast.

        This validates that the optical transfer function correctly
        attenuates high spatial frequencies.
        """
        # Low-NA microscope to make resolution limit observable
        config = MicroscopeConfig(
            numerical_aperture=0.25,
            magnification=20.0,  # 325nm object pixels, Nyquist OK for NA=0.25
            wavelength=532e-9,
            n_pixels=256,
            pixel_size=6.5e-6,
        )
        mic = Microscope(config)

        # Coarse bars (well above resolution limit)
        coarse_target = torch.zeros(256, 256, dtype=torch.complex64)
        for i in range(0, 256, 40):  # 40px period
            coarse_target[:, i : i + 20] = 1.0

        # Fine bars (approaching resolution limit)
        fine_target = torch.zeros(256, 256, dtype=torch.complex64)
        for i in range(0, 256, 8):  # 8px period
            fine_target[:, i : i + 4] = 1.0

        output_coarse = mic.forward(coarse_target)
        output_fine = mic.forward(fine_target)

        # Calculate contrast (max - min) / (max + min)
        def contrast(img):
            # Use central region to avoid edge effects
            roi = img[64:192, 64:192]
            return (roi.max() - roi.min()) / (roi.max() + roi.min() + 1e-8)

        contrast_coarse = contrast(output_coarse)
        contrast_fine = contrast(output_fine)

        assert contrast_fine < contrast_coarse, (
            f"Fine bars should have lower contrast than coarse: "
            f"fine={contrast_fine:.3f}, coarse={contrast_coarse:.3f}"
        )


class TestMicroscopeForwardModelInfo:
    """Tests for forward model information/diagnostics."""

    @pytest.mark.skipif(
        not hasattr(Microscope, "forward_model"),
        reason="Unified forward model not yet integrated (Phase 3)",
    )
    def test_get_info_returns_configuration(self):
        """get_info() should return dictionary with key parameters."""
        config = MicroscopeConfig(
            numerical_aperture=1.4,
            magnification=100.0,
            medium_index=1.515,
            wavelength=532e-9,
            n_pixels=128,
            pixel_size=6.5e-6,
        )
        mic = Microscope(config)

        info = mic.forward_model.get_info()

        assert isinstance(info, dict), "get_info() should return dict"
        assert "regime" in info, "Info should include regime"
        assert "defocus_parameter" in info, "Info should include defocus_parameter"
        assert "na" in info, "Info should include NA"
        assert "medium_index" in info, "Info should include medium_index"

    @pytest.mark.skipif(
        not hasattr(Microscope, "forward_model"),
        reason="Unified forward model not yet integrated (Phase 3)",
    )
    def test_forward_model_lazy_initialization(self):
        """Forward model should be lazily initialized on first access."""
        config = MicroscopeConfig(
            numerical_aperture=0.9,
            magnification=100.0,
            wavelength=532e-9,
            n_pixels=128,
            pixel_size=6.5e-6,
        )
        mic = Microscope(config)

        # Before accessing forward_model, _forward_model should be None
        assert mic._forward_model is None, "Forward model should not be created during __init__"

        # Access triggers initialization
        _ = mic.forward_model

        # Now should be initialized
        assert mic._forward_model is not None, (
            "Forward model should be initialized after first access"
        )


class TestMicroscopeLegacyCompatibility:
    """Tests for backward compatibility with legacy forward model."""

    @pytest.mark.skipif(
        not hasattr(Microscope, "_forward_legacy"),
        reason="Legacy forward method not yet added (Phase 3)",
    )
    def test_use_unified_model_false_uses_legacy(self):
        """use_unified_model=False should use legacy FFT-only implementation."""
        config = MicroscopeConfig(
            numerical_aperture=0.9,
            magnification=100.0,
            wavelength=532e-9,
            n_pixels=128,
            pixel_size=6.5e-6,
        )
        mic = Microscope(config)

        field = torch.zeros(128, 128, dtype=torch.complex64)
        field[64, 64] = 1.0

        # Should work without error
        output = mic.forward(field, use_unified_model=False)

        assert output.shape == (128, 128), "Legacy output shape should match"
        assert (output >= 0).all(), "Legacy output should be non-negative"

    @pytest.mark.skipif(
        not hasattr(Microscope, "forward_model"),
        reason="Unified forward model not yet integrated (Phase 3)",
    )
    def test_unified_and_legacy_match_at_focus(self):
        """At focal plane, unified and legacy models should give same result.

        When object is at the focal plane and SIMPLIFIED regime is selected,
        the unified forward model should produce identical results to the
        legacy FFT-only implementation.
        """
        f_obj = 0.2 / 100.0  # 2mm

        config = MicroscopeConfig(
            numerical_aperture=0.9,
            magnification=100.0,
            wavelength=532e-9,
            n_pixels=128,
            pixel_size=6.5e-6,
            working_distance=f_obj,  # Exactly at focus
        )
        mic = Microscope(config)

        # Point source
        field = torch.zeros(128, 128, dtype=torch.complex64)
        field[64, 64] = 1.0

        output_unified = mic.forward(field, use_unified_model=True)
        output_legacy = mic.forward(field, use_unified_model=False)

        # Should be identical
        torch.testing.assert_close(
            output_unified,
            output_legacy,
            rtol=1e-4,
            atol=1e-8,
            msg="Unified and legacy should match at focus",
        )


class TestMicroscopeDefocusThreshold:
    """Tests for defocus threshold configuration."""

    @pytest.mark.skipif(
        not hasattr(Microscope, "forward_model"),
        reason="Unified forward model not yet integrated (Phase 3)",
    )
    def test_custom_defocus_threshold(self):
        """Custom defocus threshold should be respected."""
        from prism.core.optics import ForwardModelRegime

        f_obj = 0.2 / 100.0

        # 5% defocus with 10% threshold -> should use SIMPLIFIED
        config = MicroscopeConfig(
            numerical_aperture=0.9,
            magnification=100.0,
            wavelength=532e-9,
            n_pixels=128,
            pixel_size=6.5e-6,
            working_distance=f_obj * 1.05,  # 5% defocus
            defocus_threshold=0.10,  # 10% threshold
        )
        mic = Microscope(config)

        # 5% < 10% threshold -> SIMPLIFIED
        assert mic.forward_model.selected_regime == ForwardModelRegime.SIMPLIFIED, (
            f"5% defocus with 10% threshold should use SIMPLIFIED, "
            f"got {mic.forward_model.selected_regime}"
        )

    @pytest.mark.skipif(
        not hasattr(Microscope, "forward_model"),
        reason="Unified forward model not yet integrated (Phase 3)",
    )
    def test_tight_defocus_threshold_triggers_full(self):
        """Small threshold should trigger FULL for minor defocus."""
        from prism.core.optics import ForwardModelRegime

        f_obj = 0.2 / 100.0

        # 0.5% defocus with 0.1% threshold -> should use FULL
        config = MicroscopeConfig(
            numerical_aperture=0.9,
            magnification=100.0,
            wavelength=532e-9,
            n_pixels=128,
            pixel_size=6.5e-6,
            working_distance=f_obj * 1.005,  # 0.5% defocus
            defocus_threshold=0.001,  # 0.1% threshold
        )
        mic = Microscope(config)

        # 0.5% > 0.1% threshold -> FULL
        assert mic.forward_model.selected_regime == ForwardModelRegime.FULL, (
            f"0.5% defocus with 0.1% threshold should use FULL, "
            f"got {mic.forward_model.selected_regime}"
        )
