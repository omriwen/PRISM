"""
Unit tests for the Mo-PIE (Motion-aware Ptychographic Iterative Engine) algorithm.

Tests cover:
- Basic Mo-PIE initialization
- Fixed probe mode (default)
- Non-fixed (learnable) probe mode with Fourier shift implementation
"""

from __future__ import annotations

import torch

from prism.core.algorithms.mopie import MoPIE


class TestMoPIEInitialization:
    """Tests for MoPIE class initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic MoPIE initialization with default parameters."""
        mopie = MoPIE(n=64, r=10.0, obj_size=32)

        assert mopie.n_int == 64
        assert mopie.r_float == 10.0
        assert mopie.obj_size == 32
        assert mopie.fix_probe_bool is True  # Default is fixed probe

    def test_initialization_with_fix_probe_false(self) -> None:
        """Test MoPIE initialization with learnable probe."""
        mopie = MoPIE(n=64, r=10.0, obj_size=32, fix_probe=False)

        assert mopie.fix_probe_bool is False
        assert mopie.probe is not None


class TestFixedProbeMode:
    """Tests for fixed probe mode (default)."""

    def test_fixed_probe_pr_generation(self) -> None:
        """Test that Pr generates masks at sample positions for fixed probe."""
        mopie = MoPIE(n=64, r=10.0, obj_size=32, fix_probe=True)

        # Set up sample positions by directly setting the cntr_rec buffer
        # (curr_center uses cntr_rec when is_meas=False)
        centers = torch.tensor([[0, 0], [5, 5], [-5, 5]])
        mopie.cntr_rec = centers

        # Get shifted probes
        shifted_probes = mopie.Pr

        # Should have one probe per center
        assert shifted_probes.shape[0] == 3
        assert shifted_probes.shape[1] == 1  # Channel dimension
        assert shifted_probes.shape[2] == 64  # Height
        assert shifted_probes.shape[3] == 64  # Width


class TestNonFixedProbeMode:
    """Tests for non-fixed (learnable) probe mode with Fourier shift."""

    def test_nonfixed_probe_pr_generation(self) -> None:
        """Test that Pr uses Fourier shift for non-fixed probe."""
        mopie = MoPIE(n=64, r=10.0, obj_size=32, fix_probe=False)

        # Set up sample positions by directly setting the cntr_rec buffer
        centers = torch.tensor([[0, 0], [5, 5], [-5, 5]])
        mopie.cntr_rec = centers

        # Get shifted probes - should not raise NotImplementedError
        shifted_probes = mopie.Pr

        # Should have one probe per center
        assert shifted_probes.shape[0] == 3
        assert shifted_probes.shape[1] == 1  # Channel dimension
        assert shifted_probes.shape[2] == 64  # Height
        assert shifted_probes.shape[3] == 64  # Width

    def test_nonfixed_probe_shift_correctness(self) -> None:
        """Test that probe shift works correctly."""
        mopie = MoPIE(n=64, r=10.0, obj_size=32, fix_probe=False)

        # Set a known probe pattern (note: Pg has shape from obj which is (1,1,n,n))
        probe = torch.zeros((1, 1, 64, 64))
        probe[0, 0, 32, 32] = 1.0  # Delta at center
        mopie.Pg = probe

        # Test shift by (0, 0) - should remain at center
        mopie.cntr_rec = torch.tensor([[0, 0]])
        shifted_probes = mopie.Pr
        assert shifted_probes[0, 0, 32, 32] == 1.0  # Should still be at center

        # Test shift by (5, 3) - delta should move
        mopie.cntr_rec = torch.tensor([[5, 3]])
        shifted_probes = mopie.Pr
        # After rolling by (-5, -3), the peak should be at (32-5, 32-3) = (27, 29)
        assert shifted_probes[0, 0, 27, 29] == 1.0

    def test_nonfixed_probe_multiple_positions(self) -> None:
        """Test shifting probe to multiple positions simultaneously."""
        mopie = MoPIE(n=64, r=10.0, obj_size=32, fix_probe=False)

        # Set a known probe pattern
        probe = torch.zeros((1, 1, 64, 64))
        probe[0, 0, 32, 32] = 1.0  # Delta at center
        mopie.Pg = probe

        # Multiple shift positions
        centers = torch.tensor([[0, 0], [10, 0], [0, 10], [10, 10]])
        mopie.cntr_rec = centers
        shifted_probes = mopie.Pr

        # Check each shifted probe
        assert shifted_probes.shape[0] == 4
        assert shifted_probes[0, 0, 32, 32] == 1.0  # No shift
        assert shifted_probes[1, 0, 22, 32] == 1.0  # Shift by (10, 0) -> center at (22, 32)
        assert shifted_probes[2, 0, 32, 22] == 1.0  # Shift by (0, 10) -> center at (32, 22)
        assert shifted_probes[3, 0, 22, 22] == 1.0  # Shift by (10, 10) -> center at (22, 22)

    def test_nonfixed_probe_dtype_preservation(self) -> None:
        """Test that probe dtype is preserved during shift."""
        mopie = MoPIE(n=64, r=10.0, obj_size=32, fix_probe=False)

        # Ensure probe is complex
        complex_probe = torch.randn(1, 1, 64, 64, dtype=torch.cfloat)
        mopie.Pg = complex_probe

        mopie.cntr_rec = torch.tensor([[5, 5]])
        shifted_probes = mopie.Pr

        # Output should match object dtype
        assert shifted_probes.dtype == mopie.obj.dtype


class TestMoPIEUpdate:
    """Tests for MoPIE update operations."""

    def test_update_obj_runs_without_error(self) -> None:
        """Test that object update can run (basic smoke test)."""
        # Create a minimal MoPIE setup
        mopie = MoPIE(n=64, r=10.0, obj_size=32, fix_probe=True)

        # Set up ground truth and sample position
        mopie.ground_truth = torch.rand(1, 1, 64, 64)
        mopie.center = torch.tensor([[0, 0]])

        # Run update - should not raise
        mopie.update_obj()

    def test_update_step_fixed_probe(self) -> None:
        """Test combined update step with fixed probe."""
        mopie = MoPIE(n=64, r=10.0, obj_size=32, fix_probe=True)

        mopie.ground_truth = torch.rand(1, 1, 64, 64)
        mopie.center = torch.tensor([[0, 0]])

        # Should run without error
        mopie.update_step()

    def test_update_step_nonfixed_probe(self) -> None:
        """Test combined update step with non-fixed probe."""
        mopie = MoPIE(n=64, r=10.0, obj_size=32, fix_probe=False)

        mopie.ground_truth = torch.rand(1, 1, 64, 64)
        mopie.center = torch.tensor([[0, 0]])

        # Should run without error (probe update is no-op when fix_probe=True,
        # but here fix_probe=False so it should update)
        mopie.update_step()


class TestMoPIEProperties:
    """Tests for MoPIE computed properties."""

    def test_og_property(self) -> None:
        """Test object getter property."""
        mopie = MoPIE(n=64, r=10.0, obj_size=32)
        assert mopie.og is mopie.obj

    def test_og_setter_constraints(self) -> None:
        """Test object setter applies constraints."""
        mopie = MoPIE(n=64, r=10.0, obj_size=32, complex_data=False)

        # Set value outside [0, 1]
        mopie.og = torch.full((1, 1, 64, 64), fill_value=2.0)

        # Should be clamped to [0, 1]
        assert mopie.og.max() <= 1.0

    def test_og_fourier_transform(self) -> None:
        """Test Og property returns FFT of object."""
        mopie = MoPIE(n=64, r=10.0, obj_size=32)

        # Get k-space representation
        obj_kspace = mopie.Og

        assert obj_kspace.shape == mopie.og.shape
        assert obj_kspace.is_complex()
