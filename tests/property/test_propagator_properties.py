"""
Property-based tests for Propagators using hypothesis.

These tests verify physical properties that should hold for optical propagation:
- Energy conservation
- Reversibility
- Unitarity
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from prism.core.grid import Grid
from prism.core.propagators import (
    AngularSpectrumPropagator,
    FraunhoferPropagator,
    FresnelPropagator,
)


@given(
    n=st.integers(min_value=64, max_value=256),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
)
@settings(max_examples=30, deadline=None)
def test_fraunhofer_energy_conservation(n, wavelength):
    """Property: Fraunhofer propagation should conserve energy (unitarity)."""
    # Create propagator
    prop = FraunhoferPropagator(normalize=True)

    # Create random complex field
    field = torch.randn(1, 1, n, n, dtype=torch.complex64)

    # Propagate forward
    propagated = prop(field, direction="forward")

    # Energy before and after
    energy_before = (field.abs() ** 2).sum()
    energy_after = (propagated.abs() ** 2).sum()

    # Should be conserved with orthonormal FFT
    assert torch.allclose(energy_before, energy_after, rtol=1e-4)


@given(
    n=st.integers(min_value=64, max_value=256),
)
@settings(max_examples=30, deadline=None)
def test_fraunhofer_reversibility(n):
    """Property: Fraunhofer forward then backward should return to original."""
    prop = FraunhoferPropagator(normalize=True)

    # Create random complex field
    field = torch.randn(1, 1, n, n, dtype=torch.complex64)

    # Forward then backward
    forward = prop(field, direction="forward")
    back = prop(forward, direction="backward")

    # Should return to original (allow for numerical precision)
    assert torch.allclose(field, back, rtol=1e-4, atol=1e-6)


@given(
    n=st.integers(min_value=64, max_value=128),
    dx=st.floats(min_value=1e-6, max_value=1e-4),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
    distance=st.floats(min_value=0.01, max_value=1.0),
)
@settings(max_examples=30, deadline=None)
def test_fresnel_energy_conservation(n, dx, wavelength, distance):
    """Property: Fresnel propagation should approximately conserve energy."""
    # Skip if parameters are invalid
    assume(distance > wavelength * 100)  # Ensure reasonable distance

    # Calculate Fresnel number
    fov = n * dx
    fresnel_number = fov**2 / (wavelength * distance)

    # Only test in valid Fresnel regime (0.01 < F < 100)
    assume(0.01 < fresnel_number < 100)

    # Create propagator
    dxf = 1.0 / (n * dx)
    prop = FresnelPropagator(
        dx=dx,
        dxf=dxf,
        wavelength=wavelength,
        obj_distance=distance,
        image_size=n,
    )

    # Create random complex field
    field = torch.randn(1, 1, n, n, dtype=torch.complex64)

    # Propagate
    propagated = prop(field)

    # Energy before and after
    energy_before = (field.abs() ** 2).sum()
    energy_after = (propagated.abs() ** 2).sum()

    # Should be approximately conserved (Fresnel is approximate)
    # Allow larger tolerance than Fraunhofer due to approximations
    assert torch.allclose(energy_before, energy_after, rtol=0.1)


@given(
    n=st.integers(min_value=64, max_value=128),
    dx=st.floats(min_value=1e-6, max_value=1e-4),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
    distance=st.floats(min_value=0.001, max_value=0.1),
)
@settings(max_examples=30, deadline=None)
def test_angular_spectrum_energy_conservation(n, dx, wavelength, distance):
    """Property: Angular spectrum should conserve energy (exact method)."""
    # Skip if parameters are invalid
    assume(distance > wavelength)

    # Create grid and propagator
    grid = Grid(nx=n, dx=dx, wavelength=wavelength)
    prop = AngularSpectrumPropagator(grid, distance=distance)

    # Create random complex field
    field = torch.randn(1, 1, n, n, dtype=torch.complex64)

    # Propagate
    propagated = prop(field)

    # Energy before and after (excluding evanescent waves)
    energy_before = (field.abs() ** 2).sum()
    energy_after = (propagated.abs() ** 2).sum()

    # Should be well conserved (within numerical precision)
    assert torch.allclose(energy_before, energy_after, rtol=1e-3)


@given(
    n=st.integers(min_value=64, max_value=128),
    dx=st.floats(min_value=1e-6, max_value=1e-4),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
    distance=st.floats(min_value=0.001, max_value=0.1),
)
@settings(max_examples=30, deadline=None)
def test_angular_spectrum_reversibility(n, dx, wavelength, distance):
    """Property: Angular spectrum forward then backward should return to original."""
    # Skip if parameters are invalid
    assume(distance > wavelength)

    # Create grid and propagators for forward and backward
    grid = Grid(nx=n, dx=dx, wavelength=wavelength)
    prop_forward = AngularSpectrumPropagator(grid, distance=distance)
    prop_backward = AngularSpectrumPropagator(grid, distance=-distance)

    # Create random complex field
    field = torch.randn(1, 1, n, n, dtype=torch.complex64)

    # Forward then backward
    forward = prop_forward(field)
    back = prop_backward(forward)

    # Should return to original (within tolerance)
    assert torch.allclose(field, back, rtol=1e-3, atol=1e-6)


@given(
    n=st.integers(min_value=64, max_value=128),
    dx=st.floats(min_value=1e-6, max_value=1e-4),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
)
@settings(max_examples=30, deadline=None)
def test_angular_spectrum_evanescent_filtering(n, dx, wavelength):
    """Property: Angular spectrum should filter evanescent waves."""
    # Create grid and propagator
    grid = Grid(nx=n, dx=dx, wavelength=wavelength)
    prop = AngularSpectrumPropagator(grid, distance=0.01)

    # The propagator should have a diffraction limit mask
    diff_limit = prop.diff_limit_tensor

    # Check that some frequencies are filtered (for typical parameters)
    # At least the DC component should propagate
    assert diff_limit.any()

    # If Nyquist frequency is too high, some should be filtered
    nyquist = 1.0 / (2.0 * dx)
    if nyquist * wavelength > 1.0:
        # Evanescent waves present
        assert not diff_limit.all()


@given(
    n=st.integers(min_value=64, max_value=128),
    dx=st.floats(min_value=1e-6, max_value=1e-4),
)
@settings(max_examples=30, deadline=None)
def test_fraunhofer_linearity(n, dx):
    """Property: Fraunhofer propagation should be linear."""
    prop = FraunhoferPropagator(normalize=True)

    # Create two random fields
    field1 = torch.randn(1, 1, n, n, dtype=torch.complex64)
    field2 = torch.randn(1, 1, n, n, dtype=torch.complex64)

    # Propagate individually
    prop1 = prop(field1, direction="forward")
    prop2 = prop(field2, direction="forward")

    # Propagate sum
    prop_sum = prop(field1 + field2, direction="forward")

    # Should satisfy linearity: prop(f1 + f2) = prop(f1) + prop(f2)
    # Allow for numerical precision in complex FFT
    assert torch.allclose(prop_sum, prop1 + prop2, rtol=1e-4, atol=1e-5)


@given(
    n=st.integers(min_value=64, max_value=128),
    dx=st.floats(min_value=1e-6, max_value=1e-4),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
    alpha=st.floats(min_value=0.1, max_value=10.0),
)
@settings(max_examples=30, deadline=None)
def test_fresnel_linearity(n, dx, wavelength, alpha):
    """Property: Fresnel propagation should be linear."""
    # Fixed distance for this test
    distance = 0.1

    # Create propagator
    dxf = 1.0 / (n * dx)
    prop = FresnelPropagator(
        dx=dx,
        dxf=dxf,
        wavelength=wavelength,
        obj_distance=distance,
        image_size=n,
    )

    # Create random field
    field = torch.randn(1, 1, n, n, dtype=torch.complex64)

    # Propagate scaled field
    prop_scaled = prop(alpha * field)

    # Should equal scaled propagation
    prop_field = prop(field)

    # Allow for numerical precision in complex arithmetic
    assert torch.allclose(prop_scaled, alpha * prop_field, rtol=1e-4, atol=1e-5)


@given(
    n=st.integers(min_value=64, max_value=128),
    dx=st.floats(min_value=1e-6, max_value=1e-4),
    wavelength=st.floats(min_value=400e-9, max_value=700e-9),
)
@settings(max_examples=30, deadline=None)
def test_angular_spectrum_linearity(n, dx, wavelength):
    """Property: Angular spectrum propagation should be linear."""
    # Create grid and propagator
    grid = Grid(nx=n, dx=dx, wavelength=wavelength)
    prop = AngularSpectrumPropagator(grid, distance=0.01)

    # Create two random fields
    field1 = torch.randn(1, 1, n, n, dtype=torch.complex64)
    field2 = torch.randn(1, 1, n, n, dtype=torch.complex64)

    # Propagate individually
    prop1 = prop(field1)
    prop2 = prop(field2)

    # Propagate sum
    prop_sum = prop(field1 + field2)

    # Should satisfy linearity (allow for numerical precision)
    assert torch.allclose(prop_sum, prop1 + prop2, rtol=1e-4, atol=1e-5)


@given(
    n=st.integers(min_value=64, max_value=128),
)
@settings(max_examples=30, deadline=None)
def test_fraunhofer_dc_preservation(n):
    """Property: Fraunhofer should preserve DC component correctly."""
    prop = FraunhoferPropagator(normalize=True)

    # Create constant field (DC only)
    field = torch.ones(1, 1, n, n, dtype=torch.complex64)

    # Propagate forward
    propagated = prop(field, direction="forward")

    # DC component should be at center after fftshift
    # For ortho normalization, DC = sum(field) / sqrt(N)
    dc_expected = field.sum() / (n * n) ** 0.5

    # Get center value (DC component)
    center = n // 2
    dc_actual = propagated[0, 0, center, center]

    # Should match expected DC value
    assert torch.allclose(dc_actual, dc_expected, rtol=1e-5)


# Validation tests
def test_fresnel_invalid_parameters():
    """Test that invalid Fresnel parameters raise errors."""
    # Negative wavelength
    with pytest.raises(ValueError, match="wavelength must be positive"):
        FresnelPropagator(
            dx=1e-5,
            dxf=1.0 / (256 * 1e-5),
            wavelength=-520e-9,
            obj_distance=0.1,
            image_size=256,
        )

    # Distance too small (< wavelength)
    with pytest.raises(ValueError, match="obj_distance.*too small"):
        FresnelPropagator(
            dx=1e-5,
            dxf=1.0 / (256 * 1e-5),
            wavelength=520e-9,
            obj_distance=100e-9,  # Less than wavelength
            image_size=256,
        )


def test_angular_spectrum_invalid_parameters():
    """Test that invalid Angular Spectrum parameters raise errors."""
    # Negative wavelength (Grid validates this, not AngularSpectrum directly)
    with pytest.raises(ValueError, match="wavelength must be positive"):
        grid = Grid(nx=256, dx=1e-5, wavelength=-520e-9)
        AngularSpectrumPropagator(grid)


def test_fraunhofer_invalid_direction():
    """Test that invalid direction raises error."""
    prop = FraunhoferPropagator()
    field = torch.randn(64, 64, dtype=torch.complex64)

    with pytest.raises(ValueError, match="Unknown direction"):
        prop(field, direction="invalid")
