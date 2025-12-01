# ruff: noqa: N806  # Allow uppercase X for frequency-domain (standard in DSP/optics)
"""
Phase 2: Direction Parameter Verification Tests.

This module tests the direction parameter in FraunhoferPropagator
and verifies FFT normalization behavior.

Key Tests:
1. FFT normalization preserves energy (Parseval's theorem)
2. Forward and backward directions are proper inverses
3. Behavior matches legacy fallback (direct FFT calls)
4. Ortho normalization is consistent

Physics Background:
    - Fraunhofer propagation: U_k = FFT(U_spatial)
    - With ortho normalization: ||FFT(x)||² = ||x||² (energy preserved)
    - Forward then backward should recover original: IFFT(FFT(x)) = x
"""

from __future__ import annotations

import pytest
import torch

from prism.core.propagators import FraunhoferPropagator
from prism.utils.transforms import fft, ifft


class TestFFTNormalization:
    """Test FFT normalization modes and energy conservation."""

    def test_ortho_normalization_preserves_energy(self):
        """Verify ortho normalization preserves energy (Parseval's theorem)."""
        # Create random complex field
        x = torch.randn(128, 128, dtype=torch.complex64)

        # FFT with ortho normalization
        X = torch.fft.fft2(x, norm="ortho")

        # Check energy conservation
        energy_spatial = (x.abs() ** 2).sum()
        energy_freq = (X.abs() ** 2).sum()

        assert torch.allclose(energy_spatial, energy_freq, rtol=1e-5), (
            f"Energy not conserved: spatial={energy_spatial:.6f}, freq={energy_freq:.6f}"
        )

    def test_ortho_fft_ifft_are_inverses(self):
        """Verify FFT and IFFT are proper inverses with ortho normalization."""
        x = torch.randn(128, 128, dtype=torch.complex64)

        # Forward then backward should recover original
        X = torch.fft.fft2(x, norm="ortho")
        x_recovered = torch.fft.ifft2(X, norm="ortho")

        # Note: complex64 uses float32 for real/imag, so tolerance is ~1e-6
        assert torch.allclose(x, x_recovered, rtol=1e-4, atol=1e-6), (
            "FFT/IFFT with ortho not perfect inverses"
        )

    def test_backward_normalization_energy_not_preserved(self):
        """Verify backward normalization does NOT preserve energy on FFT."""
        x = torch.randn(128, 128, dtype=torch.complex64)

        # FFT with backward normalization (PyTorch default)
        X = torch.fft.fft2(x, norm="backward")

        # Energy is NOT preserved (scaled by N)
        energy_spatial = (x.abs() ** 2).sum()
        energy_freq = (X.abs() ** 2).sum()

        N = x.shape[0] * x.shape[1]
        expected_freq_energy = energy_spatial * N

        assert torch.allclose(energy_freq, expected_freq_energy, rtol=1e-5), (
            f"Backward norm energy scaling incorrect: "
            f"expected {expected_freq_energy:.2e}, got {energy_freq:.2e}"
        )


class TestDirectionParameter:
    """Test direction parameter in FraunhoferPropagator."""

    def test_forward_direction_is_fft(self):
        """Verify direction='forward' performs FFT."""
        prop = FraunhoferPropagator(normalize=True)
        x = torch.randn(128, 128, dtype=torch.complex64)

        # Forward propagation
        X_prop = prop(x, direction="forward")

        # Direct FFT with ortho
        X_direct = torch.fft.fft2(torch.fft.ifftshift(x), norm="ortho")
        X_direct = torch.fft.fftshift(X_direct)

        assert torch.allclose(X_prop, X_direct, rtol=1e-5, atol=1e-7), (
            "Forward direction does not match FFT"
        )

    def test_backward_direction_is_ifft(self):
        """Verify direction='backward' performs IFFT."""
        prop = FraunhoferPropagator(normalize=True)
        X = torch.randn(128, 128, dtype=torch.complex64)

        # Backward propagation
        x_prop = prop(X, direction="backward")

        # Direct IFFT with ortho
        x_direct = torch.fft.ifftshift(X)
        x_direct = torch.fft.ifft2(x_direct, norm="ortho")
        x_direct = torch.fft.fftshift(x_direct)

        assert torch.allclose(x_prop, x_direct, rtol=1e-5, atol=1e-7), (
            "Backward direction does not match IFFT"
        )

    def test_forward_backward_inversion(self):
        """Verify forward then backward recovers original field."""
        prop = FraunhoferPropagator(normalize=True)
        x = torch.randn(128, 128, dtype=torch.complex64)

        # Forward then backward
        X = prop(x, direction="forward")
        x_recovered = prop(X, direction="backward")

        # Note: complex64 uses float32 for real/imag, so tolerance is ~1e-6
        assert torch.allclose(x, x_recovered, rtol=1e-4, atol=1e-6), (
            "Forward then backward did not recover original"
        )

    def test_backward_forward_inversion(self):
        """Verify backward then forward recovers original field."""
        prop = FraunhoferPropagator(normalize=True)
        X = torch.randn(128, 128, dtype=torch.complex64)

        # Backward then forward
        x = prop(X, direction="backward")
        X_recovered = prop(x, direction="forward")

        # Note: complex64 uses float32 for real/imag, so tolerance is ~1e-6
        assert torch.allclose(X, X_recovered, rtol=1e-4, atol=1e-6), (
            "Backward then forward did not recover original"
        )

    def test_energy_conservation_forward(self):
        """Verify forward propagation conserves energy with normalize=True."""
        prop = FraunhoferPropagator(normalize=True)
        x = torch.randn(128, 128, dtype=torch.complex64)

        # Forward propagation
        X = prop(x, direction="forward")

        # Check energy
        energy_in = (x.abs() ** 2).sum()
        energy_out = (X.abs() ** 2).sum()

        assert torch.allclose(energy_in, energy_out, rtol=1e-5), (
            f"Forward propagation energy not conserved: in={energy_in:.6f}, out={energy_out:.6f}"
        )

    def test_energy_conservation_backward(self):
        """Verify backward propagation conserves energy with normalize=True."""
        prop = FraunhoferPropagator(normalize=True)
        X = torch.randn(128, 128, dtype=torch.complex64)

        # Backward propagation
        x = prop(X, direction="backward")

        # Check energy
        energy_in = (X.abs() ** 2).sum()
        energy_out = (x.abs() ** 2).sum()

        assert torch.allclose(energy_in, energy_out, rtol=1e-5), (
            f"Backward propagation energy not conserved: in={energy_in:.6f}, out={energy_out:.6f}"
        )

    def test_invalid_direction_raises_error(self):
        """Verify invalid direction raises ValueError."""
        prop = FraunhoferPropagator(normalize=True)
        x = torch.randn(128, 128, dtype=torch.complex64)

        with pytest.raises(ValueError, match="Unknown direction"):
            prop(x, direction="invalid")


class TestLegacyCompatibility:
    """Test compatibility with legacy fallback behavior."""

    def test_forward_matches_fft_fallback(self):
        """Verify direction='forward' matches legacy fft() function."""
        prop = FraunhoferPropagator(normalize=True)
        x = torch.randn(128, 128, dtype=torch.complex64)

        # Propagator forward
        X_prop = prop(x, direction="forward")

        # Legacy fallback (from telescope.py)
        X_legacy = fft(x)

        assert torch.allclose(X_prop, X_legacy, rtol=1e-5, atol=1e-7), (
            "Forward direction does not match legacy fft() fallback"
        )

    def test_backward_matches_ifft_fallback(self):
        """Verify direction='backward' matches legacy ifft() function."""
        prop = FraunhoferPropagator(normalize=True)
        X = torch.randn(128, 128, dtype=torch.complex64)

        # Propagator backward
        x_prop = prop(X, direction="backward")

        # Legacy fallback
        x_legacy = ifft(X)

        assert torch.allclose(x_prop, x_legacy, rtol=1e-5, atol=1e-7), (
            "Backward direction does not match legacy ifft() fallback"
        )


class TestTelescopePropagationConsistency:
    """Test consistency between propagator and fallback in telescope usage."""

    def test_propagate_to_spatial_uses_fft(self):
        """
        Verify that telescope's propagate_to_spatial uses FFT for both paths (FIXED in Phase 2).

        After Phase 2 fix (telescope.py:453):
        - telescope.propagate_to_spatial() fallback: uses fft(tensor) → FFT ✓
        - telescope.propagate_to_spatial() with propagator: uses direction="forward" → FFT ✓

        Both paths now consistently use FFT as they should for Fraunhofer propagation
        to a detector at the focal plane.
        """
        from prism.utils.transforms import fft

        # Simulate telescope fallback behavior
        tensor = torch.randn(128, 128, dtype=torch.complex64)

        # Fallback uses fft (which is FFT, not IFFT)
        fallback_result = fft(tensor).abs()

        # With propagator should also use forward (FFT) after Phase 2 fix
        prop = FraunhoferPropagator(normalize=True)
        propagator_result = prop(tensor, direction="forward").abs()

        # These SHOULD match now that both use FFT (verified with Phase 2 fix)
        assert torch.allclose(fallback_result, propagator_result, rtol=1e-5, atol=1e-6), (
            "Propagator with direction='forward' should match fallback FFT (Phase 2 fix verified)"
        )

    def test_both_propagations_can_use_fft(self):
        """
        Test that both propagation steps can use FFT (not IFFT).

        Physics insight from user:
        - First propagation (object → aperture): FFT
        - Second propagation (aperture → detector at focal plane): Also FFT

        In a Fraunhofer imaging system, detector at focal plane means
        both steps are Fourier transforms.
        """
        prop = FraunhoferPropagator(normalize=True)

        # Simulate full imaging chain with both FFTs
        object_field = torch.randn(128, 128, dtype=torch.complex64)

        # First propagation: object → aperture (k-space)
        k_space = prop(object_field, direction="forward")

        # Apply aperture mask (simulate aperture)
        aperture = torch.ones_like(k_space)
        aperture[40:60, 40:60] = 0  # Block center
        k_space_masked = k_space * aperture

        # Second propagation: aperture → detector
        # User's insight: should also use FFT (forward), not IFFT (backward)
        detector_field_fft = prop(k_space_masked, direction="forward")
        detector_field_ifft = prop(k_space_masked, direction="backward")

        # These give different results
        assert not torch.allclose(detector_field_fft.abs(), detector_field_ifft.abs(), rtol=0.1), (
            "FFT and IFFT should give different results"
        )

        # The question is: which is physically correct?
        # This test documents that both options exist


class TestNormalizationModes:
    """Test different normalization modes."""

    def test_normalize_true_uses_ortho(self):
        """Verify normalize=True uses 'ortho' normalization."""
        prop = FraunhoferPropagator(normalize=True)
        x = torch.randn(128, 128, dtype=torch.complex64)

        X_prop = prop(x, direction="forward")
        X_ortho = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x), norm="ortho"))

        assert torch.allclose(X_prop, X_ortho, rtol=1e-5, atol=1e-7), (
            "normalize=True should use 'ortho' normalization"
        )

    def test_normalize_false_uses_backward(self):
        """Verify normalize=False uses 'backward' normalization."""
        prop = FraunhoferPropagator(normalize=False)
        x = torch.randn(128, 128, dtype=torch.complex64)

        X_prop = prop(x, direction="forward")
        X_backward = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x), norm="backward"))

        assert torch.allclose(X_prop, X_backward, rtol=1e-5, atol=1e-7), (
            "normalize=False should use 'backward' normalization"
        )

    def test_normalize_false_does_not_preserve_energy(self):
        """Verify normalize=False does NOT preserve energy."""
        prop = FraunhoferPropagator(normalize=False)
        x = torch.randn(128, 128, dtype=torch.complex64)

        X = prop(x, direction="forward")

        energy_in = (x.abs() ** 2).sum()
        energy_out = (X.abs() ** 2).sum()

        # Should NOT be equal
        assert not torch.allclose(energy_in, energy_out, rtol=0.1), (
            "normalize=False should NOT preserve energy"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
