"""Unit tests for configuration validation with intelligent error messages.

Tests the ConfigValidator class and ValidationError functionality including:
- Spelling suggestions for typos
- Detailed error messages with descriptions
- Help flag suggestions
- Range validations
"""

from __future__ import annotations

import pytest

from prism.config.validation import ConfigValidator, ValidationError


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error_is_exception(self):
        """ValidationError should be an Exception."""
        assert issubclass(ValidationError, Exception)

    def test_validation_error_message(self):
        """ValidationError should store error message."""
        error = ValidationError("Test error message")
        assert str(error) == "Test error message"


class TestSuggestions:
    """Test spelling suggestion functionality."""

    def test_suggest_correction_exact_match(self):
        """Should not suggest for exact matches."""
        result = ConfigValidator.suggest_correction("fraunhofer", ConfigValidator.VALID_PROPAGATORS)
        # get_close_matches returns exact matches too
        assert result == "fraunhofer"

    def test_suggest_correction_typo(self):
        """Should suggest correction for common typos."""
        result = ConfigValidator.suggest_correction(
            "fraunhaufer", ConfigValidator.VALID_PROPAGATORS
        )
        assert result == "fraunhofer"

    def test_suggest_correction_no_match(self):
        """Should return None when no good match."""
        result = ConfigValidator.suggest_correction("xyz123", ConfigValidator.VALID_PROPAGATORS)
        assert result is None

    def test_suggest_correction_multiple_options(self):
        """Should return best match among multiple similar options."""
        result = ConfigValidator.suggest_correction("ferma", ConfigValidator.VALID_PATTERNS)
        assert result == "fermat"


class TestPropagatorValidation:
    """Test propagator method validation."""

    def test_valid_propagators(self):
        """All valid propagators should pass validation."""
        for propagator in ConfigValidator.VALID_PROPAGATORS:
            ConfigValidator.validate_propagator(propagator)  # Should not raise

    def test_none_propagator(self):
        """None should be allowed (uses default)."""
        ConfigValidator.validate_propagator(None)  # Should not raise

    def test_invalid_propagator(self):
        """Invalid propagator should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_propagator("invalid")
        assert "Invalid propagator_method: 'invalid'" in str(exc_info.value)
        assert "Valid options:" in str(exc_info.value)

    def test_propagator_typo_suggestion(self):
        """Typo should suggest correct propagator."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_propagator("fraunhaufer")
        assert "Did you mean 'fraunhofer'?" in str(exc_info.value)

    def test_propagator_error_includes_descriptions(self):
        """Error should include descriptions of valid options."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_propagator("invalid")
        error_msg = str(exc_info.value)
        assert "far-field" in error_msg.lower()  # fraunhofer description
        assert "auto" in error_msg
        assert "angular_spectrum" in error_msg

    def test_propagator_error_includes_help_flag(self):
        """Error should include help flag suggestion."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_propagator("invalid")
        assert "--help-propagator" in str(exc_info.value)


class TestPatternValidation:
    """Test pattern name validation."""

    def test_valid_patterns(self):
        """All valid patterns should pass validation."""
        for pattern in ConfigValidator.VALID_PATTERNS:
            ConfigValidator.validate_pattern(pattern)  # Should not raise

    def test_none_pattern(self):
        """None should be allowed."""
        ConfigValidator.validate_pattern(None)  # Should not raise

    def test_invalid_pattern(self):
        """Invalid pattern should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_pattern("invalid")
        assert "Invalid pattern: 'invalid'" in str(exc_info.value)

    def test_pattern_typo_suggestion(self):
        """Typo should suggest correct pattern."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_pattern("ferma")
        assert "Did you mean 'fermat'?" in str(exc_info.value)


class TestLossTypeValidation:
    """Test loss type validation."""

    def test_valid_loss_types(self):
        """All valid loss types should pass validation."""
        for loss_type in ConfigValidator.VALID_LOSS_TYPES:
            ConfigValidator.validate_loss_type(loss_type)  # Should not raise

    def test_invalid_loss_type(self):
        """Invalid loss type should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_loss_type("invalid")
        assert "Invalid loss_type: 'invalid'" in str(exc_info.value)

    def test_loss_type_error_includes_descriptions(self):
        """Error should include descriptions of valid options."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_loss_type("invalid")
        error_msg = str(exc_info.value)
        assert "L1 loss" in error_msg or "Mean Absolute Error" in error_msg


class TestActivationValidation:
    """Test activation function validation."""

    def test_valid_activations(self):
        """All valid activations should pass validation."""
        for activation in ConfigValidator.VALID_ACTIVATIONS:
            ConfigValidator.validate_activation(activation)  # Should not raise

    def test_invalid_activation(self):
        """Invalid activation should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_activation("invalid")
        assert "Invalid activation: 'invalid'" in str(exc_info.value)

    def test_activation_custom_param_name(self):
        """Should use custom parameter name in error."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_activation("invalid", param_name="output_activation")
        assert "Invalid output_activation: 'invalid'" in str(exc_info.value)


class TestPositiveValidation:
    """Test positive value validation."""

    def test_positive_value(self):
        """Positive values should pass."""
        ConfigValidator.validate_positive(1.0, "test_param")
        ConfigValidator.validate_positive(0.001, "test_param")
        ConfigValidator.validate_positive(1000, "test_param")

    def test_zero_invalid(self):
        """Zero should be invalid for positive check."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_positive(0, "test_param")
        assert "test_param" in str(exc_info.value)
        assert "positive (> 0)" in str(exc_info.value)

    def test_negative_invalid(self):
        """Negative values should be invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_positive(-1.0, "test_param")
        assert "test_param" in str(exc_info.value)
        assert "positive (> 0)" in str(exc_info.value)

    def test_typical_values_in_error(self):
        """Error should include typical values if provided."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_positive(-1.0, "learning_rate", typical_values="1e-4 to 1e-2")
        assert "1e-4 to 1e-2" in str(exc_info.value)


class TestNonNegativeValidation:
    """Test non-negative value validation."""

    def test_non_negative_values(self):
        """Non-negative values should pass."""
        ConfigValidator.validate_non_negative(0, "test_param")
        ConfigValidator.validate_non_negative(1.0, "test_param")
        ConfigValidator.validate_non_negative(1000, "test_param")

    def test_negative_invalid(self):
        """Negative values should be invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_non_negative(-1.0, "test_param")
        assert "test_param" in str(exc_info.value)
        assert "non-negative" in str(exc_info.value)


class TestRangeValidation:
    """Test range validation."""

    def test_value_in_range(self):
        """Values within range should pass."""
        ConfigValidator.validate_in_range(5, "test_param", min_val=0, max_val=10)
        ConfigValidator.validate_in_range(0, "test_param", min_val=0, max_val=10)
        ConfigValidator.validate_in_range(10, "test_param", min_val=0, max_val=10)

    def test_value_below_min(self):
        """Values below minimum should be invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_in_range(-1, "test_param", min_val=0, max_val=10)
        assert "test_param" in str(exc_info.value)

    def test_value_above_max(self):
        """Values above maximum should be invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_in_range(11, "test_param", min_val=0, max_val=10)
        assert "test_param" in str(exc_info.value)

    def test_min_only(self):
        """Should work with only minimum specified."""
        ConfigValidator.validate_in_range(10, "test_param", min_val=0)
        with pytest.raises(ValidationError):
            ConfigValidator.validate_in_range(-1, "test_param", min_val=0)

    def test_max_only(self):
        """Should work with only maximum specified."""
        ConfigValidator.validate_in_range(5, "test_param", max_val=10)
        with pytest.raises(ValidationError):
            ConfigValidator.validate_in_range(11, "test_param", max_val=10)


class TestErrorFormatting:
    """Test error message formatting."""

    def test_format_enum_error_basic(self):
        """Should format basic enum error."""
        error = ConfigValidator.format_enum_error(
            param_name="test_param",
            invalid_value="bad_value",
            valid_options=["option1", "option2", "option3"],
        )
        assert "Invalid test_param: 'bad_value'" in error
        assert "option1" in error
        assert "option2" in error
        assert "option3" in error

    def test_format_enum_error_with_descriptions(self):
        """Should include descriptions when provided."""
        error = ConfigValidator.format_enum_error(
            param_name="test_param",
            invalid_value="bad_value",
            valid_options=["option1", "option2"],
            descriptions={"option1": "First option", "option2": "Second option"},
        )
        assert "First option" in error
        assert "Second option" in error

    def test_format_enum_error_with_help_flag(self):
        """Should include help flag when provided."""
        error = ConfigValidator.format_enum_error(
            param_name="test_param",
            invalid_value="bad_value",
            valid_options=["option1"],
            help_flag="--help-test",
        )
        assert "--help-test" in error

    def test_format_range_error_basic(self):
        """Should format basic range error."""
        error = ConfigValidator.format_range_error(
            param_name="test_param", invalid_value=100, valid_range="[0, 10]"
        )
        assert "Invalid test_param: 100" in error
        assert "[0, 10]" in error

    def test_format_range_error_with_typical_values(self):
        """Should include typical values when provided."""
        error = ConfigValidator.format_range_error(
            param_name="test_param",
            invalid_value=100,
            valid_range="positive",
            typical_values="1-10",
        )
        assert "1-10" in error


class TestIntegrationWithConfig:
    """Test integration with PRISMConfig validation."""

    def test_invalid_propagator_in_config(self):
        """Config validation should use enhanced error messages."""
        from prism.config.base import TelescopeConfig

        # The __post_init__ should raise ValueError with enhanced message
        with pytest.raises(ValueError) as exc_info:
            TelescopeConfig(propagator_method="invalid")
        # Verify enhanced error message is used
        assert "Valid options:" in str(exc_info.value)

    def test_config_validation_preserves_behavior(self):
        """Enhanced validation should not break existing behavior."""
        from prism.config.base import TelescopeConfig

        # Valid config should work
        TelescopeConfig(propagator_method="fraunhofer")
        # Should complete without error


class TestHelpTopics:
    """Test help topic printing (basic tests)."""

    def test_print_help_propagator(self, capsys):
        """Should print propagator help without error."""
        ConfigValidator.print_help_topic("propagator")
        captured = capsys.readouterr()
        # Just verify it runs and outputs something
        assert len(captured.out) > 0

    def test_print_help_patterns(self, capsys):
        """Should print patterns help without error."""
        ConfigValidator.print_help_topic("patterns")
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_help_loss(self, capsys):
        """Should print loss help without error."""
        ConfigValidator.print_help_topic("loss")
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_help_model(self, capsys):
        """Should print model help without error."""
        ConfigValidator.print_help_topic("model")
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_help_objects(self, capsys):
        """Should print objects help without error."""
        ConfigValidator.print_help_topic("objects")
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_help_invalid_topic(self, capsys):
        """Should handle invalid help topic gracefully."""
        ConfigValidator.print_help_topic("invalid_topic")
        captured = capsys.readouterr()
        assert "Unknown help topic" in captured.out
