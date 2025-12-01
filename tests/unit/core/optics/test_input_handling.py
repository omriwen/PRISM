"""Unit tests for input handling module."""

import warnings

import pytest
import torch

from prism.core.optics.input_handling import (
    InputMode,
    convert_to_complex_field,
    detect_input_mode,
    prepare_field,
    validate_fov_consistency,
)


class TestDetectInputMode:
    """Tests for detect_input_mode()."""

    def test_complex_tensor_returns_complex(self):
        field = torch.randn(64, 64, dtype=torch.complex64)
        assert detect_input_mode(field) == InputMode.COMPLEX

    def test_nonnegative_real_returns_intensity_with_warning(self):
        field = torch.rand(64, 64)  # [0, 1)
        with pytest.warns(UserWarning, match="Auto-detected input as INTENSITY"):
            mode = detect_input_mode(field)
        assert mode == InputMode.INTENSITY

    def test_negative_values_raise_error(self):
        field = torch.randn(64, 64)  # Has negatives
        with pytest.raises(ValueError, match="negative values"):
            detect_input_mode(field)


class TestConvertToComplexField:
    """Tests for convert_to_complex_field()."""

    def test_intensity_applies_sqrt(self):
        intensity = torch.tensor([[0.0, 0.25, 1.0]])
        result = convert_to_complex_field(intensity, InputMode.INTENSITY)
        expected = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.complex64)
        torch.testing.assert_close(result, expected)

    def test_amplitude_no_sqrt(self):
        amplitude = torch.tensor([[0.0, 0.5, 1.0]])
        result = convert_to_complex_field(amplitude, InputMode.AMPLITUDE)
        expected = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.complex64)
        torch.testing.assert_close(result, expected)

    def test_complex_passthrough(self):
        field = torch.randn(64, 64, dtype=torch.complex64)
        result = convert_to_complex_field(field, InputMode.COMPLEX)
        torch.testing.assert_close(result, field)

    def test_complex_mode_with_real_field(self):
        """Test that COMPLEX mode converts real tensors to complex."""
        field = torch.tensor([[1.0, 2.0, 3.0]])
        result = convert_to_complex_field(field, InputMode.COMPLEX)
        expected = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.complex64)
        torch.testing.assert_close(result, expected)
        assert result.is_complex()

    def test_unknown_mode_raises(self):
        """Test that an invalid input mode raises ValueError."""
        field = torch.tensor([[1.0, 2.0, 3.0]])
        # Create a mock enum value that's not valid
        with pytest.raises(ValueError, match="Unknown input mode"):
            convert_to_complex_field(field, "invalid_mode")


class TestPrepareField:
    """Tests for prepare_field()."""

    def test_shape_mismatch_raises(self):
        field = torch.rand(32, 32)
        with pytest.raises(ValueError, match="don't match"):
            prepare_field(field, expected_shape=(64, 64))

    def test_correct_shape_passes(self):
        field = torch.rand(64, 64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = prepare_field(field, expected_shape=(64, 64))
        assert result.shape == (64, 64)
        assert result.is_complex()


class TestValidateFovConsistency:
    """Tests for validate_fov_consistency()."""

    def test_no_warning_when_matched(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_fov_consistency(1e-7, 1e-7)
        assert len(w) == 0

    def test_warning_when_mismatched(self):
        with pytest.warns(UserWarning, match="FOV mismatch"):
            validate_fov_consistency(1e-7, 2e-7)  # 100% difference

    def test_none_skips_validation(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            validate_fov_consistency(None, 1e-7)
        assert len(w) == 0
