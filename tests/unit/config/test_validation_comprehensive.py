"""
Comprehensive Validation Tests for SPIDS Configuration

Tests all 40+ validation rules in spids/config/base.py:validate()
"""

from __future__ import annotations

import pytest

from prism.config import (
    ImageConfig,
    ModelConfig,
    MoPIEConfig,
    PhysicsConfig,
    PointSourceConfig,
    PRISMConfig,
    TelescopeConfig,
    TrainingConfig,
)


class TestImageValidation:
    """Test image configuration validation rules."""

    def test_image_size_must_be_positive(self):
        """Test image_size > 0 validation."""
        with pytest.raises(ValueError, match="image_size must be positive"):
            config = PRISMConfig(
                image=ImageConfig(image_size=-1),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_image_size_zero_invalid(self):
        """Test image_size = 0 is invalid."""
        with pytest.raises(ValueError, match="image_size must be positive"):
            config = PRISMConfig(
                image=ImageConfig(image_size=0),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_obj_size_positive_if_set(self):
        """Test obj_size must be positive if specified."""
        with pytest.raises(ValueError, match="obj_size must be positive"):
            config = PRISMConfig(
                image=ImageConfig(image_size=1024, obj_size=-100),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_obj_size_cannot_exceed_image_size(self):
        """Test obj_size <= image_size."""
        with pytest.raises(ValueError, match="cannot exceed image_size"):
            config = PRISMConfig(
                image=ImageConfig(image_size=1024, obj_size=2048),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_input_file_not_found_error(self, tmp_path):
        """Test helpful error when input file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.jpg"
        with pytest.raises(ValueError, match="Input image file not found"):
            config = PRISMConfig(
                image=ImageConfig(image_size=1024, input=str(nonexistent)),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_input_file_exists_passes(self, tmp_path):
        """Test validation passes when input file exists."""
        existing_file = tmp_path / "test.jpg"
        existing_file.write_text("fake image")

        config = PRISMConfig(
            image=ImageConfig(image_size=1024, input=str(existing_file)),
            telescope=TelescopeConfig(),
            model=ModelConfig(),
            training=TrainingConfig(),
            physics=PhysicsConfig(obj_name="europa"),
            point_source=PointSourceConfig(),
        )
        # Should not raise
        config.validate()


class TestTelescopeValidation:
    """Test telescope configuration validation rules."""

    def test_n_samples_must_be_positive(self):
        """Test n_samples > 0."""
        with pytest.raises(ValueError, match="n_samples must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(n_samples=0),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_sample_diameter_positive_if_set(self):
        """Test sample_diameter > 0 if specified."""
        with pytest.raises(ValueError, match="sample_diameter must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(sample_diameter=-10.0),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_sample_length_non_negative(self):
        """Test sample_length >= 0."""
        with pytest.raises(ValueError, match="sample_length cannot be negative"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(sample_length=-5.0),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_snr_positive_if_set(self):
        """Test SNR > 0 if specified."""
        with pytest.raises(ValueError, match="Invalid SNR"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(snr=-10.0),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_sampling_pattern_mutual_exclusion(self):
        """Test cannot use both star_sample and fermat_sample."""
        with pytest.raises(ValueError, match="Cannot use both star_sample and fermat_sample"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(star_sample=True, fermat_sample=True),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_sample_sort_valid_choices(self):
        """Test sample_sort must be cntr/rand/energy."""
        with pytest.raises(ValueError, match="Invalid sample_sort"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(sample_sort="invalid"),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_line_sampling_params_positive(self):
        """Test samples_per_line_meas > 0 if specified."""
        with pytest.raises(ValueError, match="samples_per_line_meas must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(sample_length=64, samples_per_line_meas=0),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_propagator_method_validation_valid_auto(self):
        """Test propagator_method='auto' is valid."""
        config = TelescopeConfig(propagator_method="auto")
        assert config.propagator_method == "auto"

    def test_propagator_method_validation_valid_fraunhofer(self):
        """Test propagator_method='fraunhofer' is valid."""
        config = TelescopeConfig(propagator_method="fraunhofer")
        assert config.propagator_method == "fraunhofer"

    def test_propagator_method_validation_valid_fresnel(self):
        """Test propagator_method='fresnel' is valid."""
        config = TelescopeConfig(propagator_method="fresnel")
        assert config.propagator_method == "fresnel"

    def test_propagator_method_validation_valid_angular_spectrum(self):
        """Test propagator_method='angular_spectrum' is valid."""
        config = TelescopeConfig(propagator_method="angular_spectrum")
        assert config.propagator_method == "angular_spectrum"

    def test_propagator_method_validation_none_is_valid(self):
        """Test propagator_method=None is valid (optional field)."""
        config = TelescopeConfig(propagator_method=None)
        assert config.propagator_method is None

    def test_propagator_method_validation_invalid(self):
        """Test propagator_method rejects invalid values."""
        with pytest.raises(ValueError, match="Invalid propagator_method"):
            TelescopeConfig(propagator_method="invalid")

    def test_propagator_method_validation_invalid_typo(self):
        """Test propagator_method rejects common typos."""
        with pytest.raises(ValueError, match="Invalid propagator_method"):
            TelescopeConfig(propagator_method="fraun")


class TestTrainingValidation:
    """Test training configuration validation rules."""

    def test_n_epochs_positive(self):
        """Test n_epochs > 0."""
        with pytest.raises(ValueError, match="n_epochs must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(n_epochs=0),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_max_epochs_positive(self):
        """Test max_epochs > 0."""
        with pytest.raises(ValueError, match="max_epochs must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(max_epochs=-1),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_lr_positive(self):
        """Test learning rate > 0."""
        with pytest.raises(ValueError, match="Invalid learning_rate"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(lr=-0.001),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_loss_threshold_positive(self):
        """Test loss_threshold > 0."""
        with pytest.raises(ValueError, match="loss_threshold must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(loss_threshold=0),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_new_weight_non_negative(self):
        """Test new_weight >= 0."""
        with pytest.raises(ValueError, match="new_weight cannot be negative"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(new_weight=-1.0),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_f_weight_non_negative(self):
        """Test f_weight >= 0."""
        with pytest.raises(ValueError, match="f_weight cannot be negative"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(f_weight=-0.0001),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_loss_type_valid(self):
        """Test loss_type must be l1 or l2."""
        with pytest.raises(ValueError, match="Invalid loss_type"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(loss_type="invalid"),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_initialization_target_valid(self):
        """Test initialization_target must be measurement or circle."""
        with pytest.raises(ValueError, match="Invalid initialization_target"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(initialization_target="invalid"),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()


class TestModelValidation:
    """Test model configuration validation rules."""

    def test_output_activation_valid(self):
        """Test output_activation must be in valid list."""
        with pytest.raises(ValueError, match="Invalid output_activation"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(output_activation="invalid"),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_middle_activation_valid(self):
        """Test middle_activation must be in valid list."""
        with pytest.raises(ValueError, match="Invalid middle_activation"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(middle_activation="invalid"),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()


class TestPhysicsValidation:
    """Test physics configuration validation rules."""

    def test_dxf_positive(self):
        """Test dxf > 0."""
        with pytest.raises(ValueError, match="dxf must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(dxf=-1.0, obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_missing_physics_params_helpful_error(self):
        """Test helpful error when physics params missing."""
        with pytest.raises(ValueError, match="Missing required physics parameters"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name=None, wavelength=None),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_wavelength_positive_if_manual(self):
        """Test wavelength > 0 if manually specified."""
        with pytest.raises(ValueError, match="wavelength must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(wavelength=-500e-9, obj_diameter=3000e3, obj_distance=600e6),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_obj_diameter_positive_if_manual(self):
        """Test obj_diameter > 0 if manually specified."""
        with pytest.raises(ValueError, match="obj_diameter must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(wavelength=500e-9, obj_diameter=-1000, obj_distance=600e6),
                point_source=PointSourceConfig(),
            )
            config.validate()


class TestPointSourceValidation:
    """Test point source configuration validation rules."""

    def test_point_source_number_positive(self):
        """Test point_source.number > 0 when enabled."""
        with pytest.raises(ValueError, match="point_source.number must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(is_point_source=True, number=0),
            )
            config.validate()

    def test_point_source_diameter_positive(self):
        """Test point_source.diameter > 0 when enabled."""
        with pytest.raises(ValueError, match="point_source.diameter must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(is_point_source=True, diameter=-3.0),
            )
            config.validate()

    def test_point_source_spacing_non_negative(self):
        """Test point_source.spacing >= 0."""
        with pytest.raises(ValueError, match="point_source.spacing cannot be negative"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(is_point_source=True, spacing=-5.0),
            )
            config.validate()


class TestMoPIEValidation:
    """Test Mo-PIE configuration validation rules."""

    def test_mopie_lr_obj_positive(self):
        """Test mopie.lr_obj > 0."""
        with pytest.raises(ValueError, match="mopie.lr_obj must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
                mopie=MoPIEConfig(lr_obj=-1.0),
            )
            config.validate()

    def test_mopie_lr_probe_positive(self):
        """Test mopie.lr_probe > 0."""
        with pytest.raises(ValueError, match="mopie.lr_probe must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
                mopie=MoPIEConfig(lr_probe=0),
            )
            config.validate()

    def test_mopie_plot_every_positive(self):
        """Test mopie.plot_every > 0."""
        with pytest.raises(ValueError, match="mopie.plot_every must be positive"):
            config = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
                mopie=MoPIEConfig(plot_every=-5),
            )
            config.validate()


class TestGeneralValidation:
    """Test general configuration validation rules."""

    def test_checkpoint_missing_error(self, tmp_path):
        """Test helpful error when checkpoint doesn't exist."""
        with pytest.raises(ValueError, match="Checkpoint file not found"):
            config = PRISMConfig(
                checkpoint="nonexistent_checkpoint",
                log_dir=str(tmp_path),
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(),
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config.validate()

    def test_checkpoint_exists_passes(self, tmp_path):
        """Test validation passes when checkpoint exists."""
        # Create checkpoint directory and file
        checkpoint_dir = tmp_path / "test_checkpoint"
        checkpoint_dir.mkdir()
        checkpoint_file = checkpoint_dir / "checkpoint.pt"
        checkpoint_file.write_text("fake checkpoint")

        config = PRISMConfig(
            checkpoint="test_checkpoint",
            log_dir=str(tmp_path),
            image=ImageConfig(),
            telescope=TelescopeConfig(),
            model=ModelConfig(),
            training=TrainingConfig(),
            physics=PhysicsConfig(obj_name="europa"),
            point_source=PointSourceConfig(),
        )
        # Should not raise
        config.validate()


class TestValidationEdgeCases:
    """Test validation edge cases and boundary values."""

    def test_boundary_value_very_small_positive(self):
        """Test very small positive values accepted."""
        config = PRISMConfig(
            image=ImageConfig(),
            telescope=TelescopeConfig(),
            model=ModelConfig(),
            training=TrainingConfig(lr=1e-10, loss_threshold=1e-10),
            physics=PhysicsConfig(obj_name="europa"),
            point_source=PointSourceConfig(),
        )
        # Should not raise
        config.validate()

    def test_none_vs_zero_distinction(self):
        """Test None (optional) vs 0 (invalid) handled correctly."""
        # None should be fine (optional)
        config1 = PRISMConfig(
            image=ImageConfig(obj_size=None),
            telescope=TelescopeConfig(snr=None),
            model=ModelConfig(),
            training=TrainingConfig(),
            physics=PhysicsConfig(obj_name="europa"),
            point_source=PointSourceConfig(),
        )
        config1.validate()  # Should pass

        # 0 should be invalid for some fields
        with pytest.raises(ValueError):
            config2 = PRISMConfig(
                image=ImageConfig(),
                telescope=TelescopeConfig(),
                model=ModelConfig(),
                training=TrainingConfig(lr=0),  # 0 is invalid
                physics=PhysicsConfig(obj_name="europa"),
                point_source=PointSourceConfig(),
            )
            config2.validate()
