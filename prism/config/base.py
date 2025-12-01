"""Configuration dataclasses for SPIDS experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

from prism.config.validation import ConfigValidator, ValidationError


@dataclass
class ImageConfig:
    """Image parameters configuration."""

    image_size: int = 1024
    """Size of the recorded image"""

    obj_size: Optional[int] = None
    """Size of the object. If None, calculated from physics parameters"""

    crop: bool = False
    """Crop the object to the object size"""

    invert: bool = False
    """Invert the image"""

    input: Optional[str] = None
    """Path to the input image"""


@dataclass
class SamplingConfig:
    """Aperture sampling pattern configuration for progressive imaging.

    This config controls how samples are taken during progressive reconstruction:
    - Pattern type (Fermat spiral, star, random)
    - Sample positions and ordering
    - Region of interest in k-space

    Note: For optical telescope properties (aperture diameter, focal length),
    use spids.core.instruments.TelescopeConfig instead.
    """

    # === Pattern Function System ===
    pattern_fn: Optional[str] = None
    """
    Pattern function specification. Either:
    - 'builtin:name' for builtin patterns (fermat, star, random)
    - '/path/to/pattern.py' for custom pattern functions

    If None, falls back to legacy star_sample/fermat_sample flags.
    """

    sample_diameter: Optional[float] = None
    """Diameter of the sampling aperture [pix]"""

    sample_shape: Literal["circle", "line"] = "circle"
    """The shape of the sample: circle or line"""

    sample_length: float = 0
    """Length of a line sample [pix]. If 0, use circles"""

    samples_per_line_meas: Optional[int] = None
    """Number of samples per line in a measurement. If None, auto-calculated"""

    samples_per_line_rec: Optional[int] = None
    """Number of samples per line in reconstruction. If None, same as measurement"""

    line_angle: Optional[float] = None
    """Angle of line samples [rad]. If None, use random angles"""

    roi_diameter: Optional[float] = None
    """Diameter of ROI in k-space [pix]. If None, set to image_size"""

    samples_r_cutoff: Optional[float] = None
    """Highest radius allowed for sample center in Fermat spiral sampling"""

    roi_shape: Literal["circle", "square"] = "circle"
    """The shape of the ROI: circle or square"""

    sample_sort: Literal["center", "rand", "energy"] = "center"
    """Sample sorting: 'center' (center proximity), 'rand' (random), 'energy' (energy)"""

    n_samples: int = 200
    """Number of samples"""

    n_angs: int = 4
    """Number of angles for star sampling"""

    # === LEGACY: Deprecated Pattern Flags ===
    star_sample: bool = False
    """DEPRECATED: Use pattern_fn='builtin:star' instead"""

    fermat_sample: bool = False
    """DEPRECATED: Use pattern_fn='builtin:fermat' instead"""

    snr: Optional[float] = None
    """Image SNR [dB]"""

    blur: bool = False
    """Apply blurring to measurement output"""

    propagator_method: Optional[str] = None
    """
    Propagator method to use for light propagation.
    Options:
    - 'auto': Automatic selection based on physics parameters (Fresnel number)
    - 'fraunhofer': Fraunhofer approximation (far-field, Fresnel << 1)
    - 'fresnel': Fresnel approximation (near-field, Fresnel ~ 1)
    - 'angular_spectrum': Angular spectrum method (general purpose, all cases)

    If None, uses default behavior.
    """

    def __post_init__(self) -> None:
        """Validate configuration and handle deprecations."""
        # Handle legacy pattern flags
        if self.star_sample or self.fermat_sample:
            import warnings

            warnings.warn(
                "star_sample and fermat_sample are deprecated. "
                "Use pattern_fn='builtin:fermat' or pattern_fn='builtin:star' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

            # Auto-convert to new system
            if self.pattern_fn is None:
                if self.fermat_sample:
                    self.pattern_fn = "builtin:fermat"
                elif self.star_sample:
                    self.pattern_fn = "builtin:star"

        # Default to random if nothing specified
        if self.pattern_fn is None and not (self.star_sample or self.fermat_sample):
            self.pattern_fn = "builtin:random"

        # Validate propagator method
        try:
            ConfigValidator.validate_propagator(self.propagator_method)
        except ValidationError as e:
            raise ValueError(str(e)) from e


# Backward compatibility alias (deprecated)
TelescopeConfig = SamplingConfig


@dataclass
class ModelConfig:
    """Neural network model parameters configuration."""

    use_bn: bool = True
    """Use batch normalization"""

    output_activation: str = "none"
    """Network output activation function (ProgressiveDecoder)"""

    use_leaky: bool = True
    """Use LeakyReLU activation"""

    middle_activation: str = "sigmoid"
    """Network middle activation function (ProgressiveDecoder)"""

    complex_data: bool = False
    """Allow complex-valued outputs"""


@dataclass
class TrainingConfig:
    """Training parameters configuration."""

    n_epochs: int = 1000
    """Number of epochs per sample"""

    max_epochs: int = 1
    """Max number of repetitions of epochs"""

    n_epochs_init: int = 100
    """Number of epochs at the initialization stage"""

    max_epochs_init: int = 100
    """Max number of repetitions of epochs at initialization"""

    initialization_target: Literal["measurement", "circle", "synthetic_aperture"] = "circle"
    """
    Target for initialization stage:
    - 'measurement': Use first aperture measurement
    - 'circle': Use circular mask (default)
    - 'synthetic_aperture': Use averaged k-space synthesis from all positions
    """

    loss_type: Literal["l1", "l2", "ssim", "ms-ssim"] = "l1"
    """
    Loss function type:
    - l1: L1 loss (mean absolute error) - default
    - l2: L2 loss (mean squared error)
    - ssim: Structural Similarity Index loss (single scale)
    - ms-ssim: Multi-Scale SSIM loss (perceptual quality, 5 scales)

    Note: SSIM-based losses operate in measurement space (diffraction patterns)
    and use DSSIM formulation: (1 - SSIM) / 2 for range [0, 0.5]
    """

    new_weight: float = 1.0
    """Weight of the new sample loss"""

    f_weight: float = 1e-4
    """Weight of the Fourier constraint loss"""

    lr: float = 1e-3
    """Learning rate"""

    loss_threshold: float = 1e-3
    """Loss threshold for stopping the training"""

    use_amsgrad: bool = False
    """Use AMSGrad in the optimizer"""

    device_num: int = 0
    """CUDA device number"""

    use_cuda: bool = True
    """Use GPU instead of CPU"""

    # Adaptive convergence settings
    enable_adaptive_convergence: bool = True
    """Enable adaptive per-sample convergence (early exit, escalation, retries)"""

    early_stop_patience: int = 10
    """Epochs of no improvement before considering stop"""

    plateau_window: int = 50
    """Window size for plateau detection"""

    plateau_threshold: float = 0.01
    """Relative improvement threshold (<1% = plateau)"""

    escalation_epochs: int = 200
    """Epochs before considering escalation to aggressive tier"""

    aggressive_lr_multiplier: float = 2.0
    """Learning rate multiplier in aggressive mode"""

    max_retries: int = 2
    """Maximum retry attempts for failed samples"""

    retry_lr_multiplier: float = 0.1
    """Learning rate multiplier on retry"""

    retry_switch_loss: bool = True
    """Switch loss function on retry attempts (cycles through L1, SSIM, L2, MS-SSIM)"""


@dataclass
class PhysicsConfig:
    """Physical parameters configuration."""

    wavelength: Optional[float] = None
    """Wavelength of the light [m]"""

    dxf: float = 1e-2
    """Pixel size on the detector plane [m]"""

    obj_diameter: Optional[float] = None
    """The real diameter of the object [m]"""

    obj_distance: Optional[float] = None
    """The distance of the object from the telescope [m]"""

    obj_name: str = "europa"
    """Predefined object name"""


@dataclass
class PointSourceConfig:
    """Point source parameters configuration."""

    is_point_source: bool = False
    """Use point source"""

    diameter: float = 3.0
    """Diameter of the point source [pix]"""

    spacing: float = 5.0
    """Spacing between point sources [pix]"""

    number: int = 4
    """Number of point sources"""


@dataclass
class MoPIEConfig:
    """Mo-PIE-specific parameters configuration."""

    lr_obj: float = 1.0
    """Mo-PIE object learning rate"""

    lr_probe: float = 1.0
    """Mo-PIE probe learning rate"""

    grad_mopie: bool = False
    """Update Mo-PIE with gradients"""

    fix_probe: bool = True
    """Keep probe fixed (don't update probe in Mo-PIE)"""

    parallel_update: bool = True
    """Update obj and probe in parallel for line samples"""

    plot_every: int = 1
    """Plot outputs every n epochs"""

    single_sample: bool = False
    """Use a single sample (middle one) even when recording a line"""

    rand_perm: bool = False
    """Randomly permute the samples on each epoch"""

    load_config_only: bool = False
    """When loading checkpoint, load only config and params, not reconstruction state"""


@dataclass
class PRISMConfig:
    """Master configuration for SPIDS experiments."""

    # Experiment metadata
    name: Optional[str] = None
    """Experiment name for saving data"""

    comment: str = ""
    """Experiment comment/description"""

    log_dir: str = "runs"
    """Directory for tensorboard logs and data"""

    save_data: bool = True
    """Save experiment data"""

    checkpoint: Optional[str] = None
    """Load a checkpoint file to continue training from"""

    # Configuration sections
    image: ImageConfig = field(default_factory=ImageConfig)
    """Image parameters"""

    telescope: SamplingConfig = field(default_factory=SamplingConfig)
    """Sampling pattern parameters (named 'telescope' for backward compatibility)"""

    model: ModelConfig = field(default_factory=ModelConfig)
    """Model parameters"""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Training parameters"""

    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    """Physics parameters"""

    point_source: PointSourceConfig = field(default_factory=PointSourceConfig)
    """Point source parameters"""

    mopie: Optional[MoPIEConfig] = None
    """Mo-PIE-specific parameters (only for main_mopie.py)"""

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)

    def validate(self) -> None:
        """Validate configuration parameters with comprehensive checks.

        Raises:
            ValueError: If configuration is invalid with helpful error messages
        """
        import os

        # =====================================================================
        # Image Configuration Validation
        # =====================================================================
        if self.image.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image.image_size}")

        if self.image.obj_size is not None and self.image.obj_size <= 0:
            raise ValueError(f"obj_size must be positive, got {self.image.obj_size}")

        if self.image.obj_size is not None and self.image.obj_size > self.image.image_size:
            raise ValueError(
                f"obj_size ({self.image.obj_size}) cannot exceed image_size ({self.image.image_size})"
            )

        # Check input file existence (if specified and not using obj_name)
        if self.image.input is not None and not os.path.exists(self.image.input):
            raise ValueError(
                f"Input image file not found: {self.image.input}\n"
                f"  → Check file path or use --obj_name to select a predefined object"
            )

        # =====================================================================
        # Telescope Configuration Validation
        # =====================================================================
        if self.telescope.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.telescope.n_samples}")

        if self.telescope.sample_diameter is not None and self.telescope.sample_diameter <= 0:
            raise ValueError(
                f"sample_diameter must be positive, got {self.telescope.sample_diameter}"
            )

        if self.telescope.sample_length < 0:
            raise ValueError(
                f"sample_length cannot be negative, got {self.telescope.sample_length}"
            )

        if self.telescope.roi_diameter is not None and self.telescope.roi_diameter <= 0:
            raise ValueError(f"roi_diameter must be positive, got {self.telescope.roi_diameter}")

        if self.telescope.samples_r_cutoff is not None and self.telescope.samples_r_cutoff < 0:
            raise ValueError(
                f"samples_r_cutoff cannot be negative, got {self.telescope.samples_r_cutoff}"
            )

        # Validate sampling pattern (mutually exclusive)
        if self.telescope.star_sample and self.telescope.fermat_sample:
            raise ValueError("Cannot use both star_sample and fermat_sample - choose one")

        # Validate pattern function specification
        if self.telescope.pattern_fn is not None:
            import os

            # Check if it's a builtin pattern
            if self.telescope.pattern_fn.startswith("builtin:"):
                builtin_name = self.telescope.pattern_fn.split(":", 1)[1]
                valid_builtins = ["fermat", "star", "random"]
                if builtin_name not in valid_builtins:
                    raise ValueError(
                        f"Unknown builtin pattern '{builtin_name}'. "
                        f"Valid options: {', '.join(valid_builtins)}"
                    )
            # Otherwise it should be a file path
            elif not os.path.exists(self.telescope.pattern_fn):
                raise ValueError(
                    f"Pattern file not found: {self.telescope.pattern_fn}\n"
                    f"  → Either use 'builtin:name' (fermat, star, random) "
                    f"or provide a valid path to a .py file"
                )
            elif not self.telescope.pattern_fn.endswith(".py"):
                raise ValueError(
                    f"Pattern file must be a .py file, got: {self.telescope.pattern_fn}"
                )

        # Validate sample sorting
        valid_sorts = ["center", "rand", "energy"]
        if self.telescope.sample_sort not in valid_sorts:
            error_msg = ConfigValidator.format_enum_error(
                param_name="sample_sort",
                invalid_value=self.telescope.sample_sort,
                valid_options=valid_sorts,
                descriptions={
                    "center": "Sort by proximity to center (default)",
                    "rand": "Random sorting",
                    "energy": "Sort by energy content",
                },
            )
            raise ValueError(error_msg)

        # Validate SNR
        if self.telescope.snr is not None:
            try:
                ConfigValidator.validate_positive(
                    self.telescope.snr, "SNR", typical_values="20-60 dB"
                )
            except ValidationError as e:
                raise ValueError(str(e)) from e

        # Validate line sampling parameters
        if self.telescope.sample_length > 0:
            if self.telescope.samples_per_line_meas is not None:
                if self.telescope.samples_per_line_meas <= 0:
                    raise ValueError(
                        f"samples_per_line_meas must be positive, got {self.telescope.samples_per_line_meas}"
                    )
            if self.telescope.samples_per_line_rec is not None:
                if self.telescope.samples_per_line_rec <= 0:
                    raise ValueError(
                        f"samples_per_line_rec must be positive, got {self.telescope.samples_per_line_rec}"
                    )

        # =====================================================================
        # Training Configuration Validation
        # =====================================================================
        if self.training.n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {self.training.n_epochs}")

        if self.training.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.training.max_epochs}")

        if self.training.n_epochs_init <= 0:
            raise ValueError(f"n_epochs_init must be positive, got {self.training.n_epochs_init}")

        if self.training.max_epochs_init <= 0:
            raise ValueError(
                f"max_epochs_init must be positive, got {self.training.max_epochs_init}"
            )

        try:
            ConfigValidator.validate_positive(
                self.training.lr, "learning_rate", typical_values="1e-4 to 1e-2"
            )
        except ValidationError as e:
            raise ValueError(str(e)) from e

        if self.training.loss_threshold <= 0:
            raise ValueError(f"loss_threshold must be positive, got {self.training.loss_threshold}")

        if self.training.new_weight < 0:
            raise ValueError(f"new_weight cannot be negative, got {self.training.new_weight}")

        if self.training.f_weight < 0:
            raise ValueError(f"f_weight cannot be negative, got {self.training.f_weight}")

        # Validate loss type
        try:
            ConfigValidator.validate_loss_type(self.training.loss_type)
        except ValidationError as e:
            raise ValueError(str(e)) from e

        # Validate initialization target
        valid_init_targets = ["measurement", "circle", "synthetic_aperture"]
        if self.training.initialization_target not in valid_init_targets:
            error_msg = ConfigValidator.format_enum_error(
                param_name="initialization_target",
                invalid_value=self.training.initialization_target,
                valid_options=valid_init_targets,
                descriptions={
                    "measurement": "Initialize from first measurement",
                    "circle": "Initialize with circular mask (default)",
                    "synthetic_aperture": "Initialize with synthetic aperture preview",
                },
            )
            raise ValueError(error_msg)

        # =====================================================================
        # Model Configuration Validation
        # =====================================================================
        try:
            ConfigValidator.validate_activation(
                self.model.output_activation, param_name="output_activation"
            )
        except ValidationError as e:
            raise ValueError(str(e)) from e

        try:
            ConfigValidator.validate_activation(
                self.model.middle_activation, param_name="middle_activation"
            )
        except ValidationError as e:
            raise ValueError(str(e)) from e

        # =====================================================================
        # Physics Configuration Validation
        # =====================================================================
        if self.physics.dxf <= 0:
            raise ValueError(f"dxf must be positive, got {self.physics.dxf}")

        # Check that either obj_name is set OR all manual physics params are provided
        has_obj_name = self.physics.obj_name is not None
        has_wavelength = self.physics.wavelength is not None
        has_obj_diameter = self.physics.obj_diameter is not None
        has_obj_distance = self.physics.obj_distance is not None

        manual_physics_complete = has_wavelength and has_obj_diameter and has_obj_distance

        if not has_obj_name and not manual_physics_complete:
            missing = []
            if not has_wavelength:
                missing.append("wavelength")
            if not has_obj_diameter:
                missing.append("obj_diameter")
            if not has_obj_distance:
                missing.append("obj_distance")

            raise ValueError(
                "Missing required physics parameters.\n"
                f"  → Either set 'obj_name' OR manually specify: {', '.join(missing)}\n"
                "  → Available obj_name values: europa, titan, betelgeuse, neptune\n"
                "  → Suggestion: Try --obj europa or --preset quick_test --obj europa"
            )

        # Validate physics values if manually specified
        if has_wavelength and self.physics.wavelength is not None and self.physics.wavelength <= 0:
            raise ValueError(f"wavelength must be positive, got {self.physics.wavelength}")

        if (
            has_obj_diameter
            and self.physics.obj_diameter is not None
            and self.physics.obj_diameter <= 0
        ):
            raise ValueError(f"obj_diameter must be positive, got {self.physics.obj_diameter}")

        if (
            has_obj_distance
            and self.physics.obj_distance is not None
            and self.physics.obj_distance <= 0
        ):
            raise ValueError(f"obj_distance must be positive, got {self.physics.obj_distance}")

        # =====================================================================
        # Point Source Configuration Validation
        # =====================================================================
        if self.point_source.is_point_source:
            if self.point_source.number <= 0:
                raise ValueError(
                    f"point_source.number must be positive, got {self.point_source.number}"
                )

            if self.point_source.diameter <= 0:
                raise ValueError(
                    f"point_source.diameter must be positive, got {self.point_source.diameter}"
                )

            if self.point_source.spacing < 0:
                raise ValueError(
                    f"point_source.spacing cannot be negative, got {self.point_source.spacing}"
                )

        # =====================================================================
        # Mo-PIE Configuration Validation (if present)
        # =====================================================================
        if hasattr(self, "mopie") and self.mopie is not None:
            if self.mopie.lr_obj <= 0:
                raise ValueError(f"mopie.lr_obj must be positive, got {self.mopie.lr_obj}")

            if self.mopie.lr_probe <= 0:
                raise ValueError(f"mopie.lr_probe must be positive, got {self.mopie.lr_probe}")

            if self.mopie.plot_every <= 0:
                raise ValueError(f"mopie.plot_every must be positive, got {self.mopie.plot_every}")

        # =====================================================================
        # General Configuration Validation
        # =====================================================================
        if self.checkpoint is not None:
            checkpoint_path = os.path.join(self.log_dir, self.checkpoint, "checkpoint.pt")
            if not os.path.exists(checkpoint_path):
                raise ValueError(
                    f"Checkpoint file not found: {checkpoint_path}\n"
                    f"  → Check that checkpoint directory '{self.checkpoint}' exists in '{self.log_dir}/'\n"
                    f"  → Available checkpoints can be listed with: ls {self.log_dir}/"
                )
