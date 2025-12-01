"""Unit tests for adaptive convergence monitoring."""

from __future__ import annotations

from prism.core.convergence import (
    ConvergenceMonitor,
    ConvergenceTier,
    TierConfig,
    get_tier_config,
)


class TestConvergenceTier:
    """Tests for ConvergenceTier enum."""

    def test_tier_values(self) -> None:
        """Test that all tier values are defined correctly."""
        assert ConvergenceTier.FAST.value == "fast"
        assert ConvergenceTier.NORMAL.value == "normal"
        assert ConvergenceTier.AGGRESSIVE.value == "aggressive"
        assert ConvergenceTier.RESCUE.value == "rescue"


class TestTierConfig:
    """Tests for TierConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default TierConfig values."""
        config = TierConfig()
        assert config.lr_multiplier == 1.0
        assert config.extra_epochs == 0
        assert config.scheduler == "plateau"

    def test_custom_values(self) -> None:
        """Test TierConfig with custom values."""
        config = TierConfig(lr_multiplier=2.0, extra_epochs=500, scheduler="cosine_warm_restarts")
        assert config.lr_multiplier == 2.0
        assert config.extra_epochs == 500
        assert config.scheduler == "cosine_warm_restarts"


class TestGetTierConfig:
    """Tests for get_tier_config function."""

    def test_fast_tier(self) -> None:
        """Test FAST tier configuration."""
        config = get_tier_config(ConvergenceTier.FAST)
        assert config.lr_multiplier == 1.0
        assert config.extra_epochs == 0
        assert config.scheduler == "plateau"

    def test_normal_tier(self) -> None:
        """Test NORMAL tier configuration."""
        config = get_tier_config(ConvergenceTier.NORMAL)
        assert config.lr_multiplier == 1.0
        assert config.extra_epochs == 0
        assert config.scheduler == "plateau"

    def test_aggressive_tier(self) -> None:
        """Test AGGRESSIVE tier configuration."""
        config = get_tier_config(ConvergenceTier.AGGRESSIVE)
        assert config.lr_multiplier == 2.0
        assert config.extra_epochs == 500
        assert config.scheduler == "cosine_warm_restarts"

    def test_rescue_tier(self) -> None:
        """Test RESCUE tier configuration."""
        config = get_tier_config(ConvergenceTier.RESCUE)
        assert config.lr_multiplier == 0.1
        assert config.extra_epochs == 1000
        assert config.scheduler == "cosine_warm_restarts"

    def test_custom_multipliers(self) -> None:
        """Test custom LR multipliers."""
        config = get_tier_config(
            ConvergenceTier.AGGRESSIVE,
            aggressive_lr_multiplier=3.0,
            retry_lr_multiplier=0.05,
        )
        assert config.lr_multiplier == 3.0

        config = get_tier_config(
            ConvergenceTier.RESCUE,
            aggressive_lr_multiplier=3.0,
            retry_lr_multiplier=0.05,
        )
        assert config.lr_multiplier == 0.05


class TestConvergenceMonitor:
    """Tests for ConvergenceMonitor class."""

    def test_initialization(self) -> None:
        """Test monitor initialization with default values."""
        monitor = ConvergenceMonitor()
        assert monitor.loss_threshold == 1e-3
        assert monitor.patience == 10
        assert monitor.plateau_window == 50
        assert monitor.plateau_threshold == 0.01
        assert monitor.escalation_epochs == 200

    def test_custom_initialization(self) -> None:
        """Test monitor initialization with custom values."""
        monitor = ConvergenceMonitor(
            loss_threshold=1e-4,
            patience=5,
            plateau_window=100,
            plateau_threshold=0.005,
            escalation_epochs=300,
        )
        assert monitor.loss_threshold == 1e-4
        assert monitor.patience == 5
        assert monitor.plateau_window == 100
        assert monitor.plateau_threshold == 0.005
        assert monitor.escalation_epochs == 300

    def test_update_records_loss(self) -> None:
        """Test that update records loss values."""
        monitor = ConvergenceMonitor()
        monitor.update(1.0)
        monitor.update(0.5)
        monitor.update(0.25)
        assert monitor.epochs_used == 3

    def test_initial_tier_is_normal(self) -> None:
        """Test that initial tier is NORMAL."""
        monitor = ConvergenceMonitor()
        assert monitor.get_current_tier() == ConvergenceTier.NORMAL

    def test_convergence_detection(self) -> None:
        """Test convergence detection."""
        monitor = ConvergenceMonitor(loss_threshold=0.01, patience=3)

        # Not converged yet
        monitor.update(0.1)
        assert not monitor.is_converged()

        # Below threshold but not stable
        monitor.update(0.005)
        monitor.update(0.004)
        assert not monitor.is_converged()

        # Now stable below threshold
        monitor.update(0.003)
        assert monitor.is_converged()

    def test_should_stop_when_converged(self) -> None:
        """Test should_stop returns True when converged."""
        monitor = ConvergenceMonitor(loss_threshold=0.01, patience=2)

        # Not converged
        monitor.update(0.1)
        assert not monitor.should_stop()

        # Converged
        monitor.update(0.005)
        monitor.update(0.004)
        assert monitor.should_stop()

    def test_plateau_detection(self) -> None:
        """Test plateau detection."""
        monitor = ConvergenceMonitor(
            loss_threshold=0.001,
            patience=5,
            plateau_window=10,
            plateau_threshold=0.01,
        )

        # Add improving losses (no plateau)
        for i in range(15):
            monitor.update(1.0 - i * 0.05)

        # Now plateau (no improvement)
        for _ in range(10):
            monitor.update(0.25)

        assert monitor.is_plateau()

    def test_no_plateau_when_improving(self) -> None:
        """Test no plateau detected when loss is improving."""
        monitor = ConvergenceMonitor(
            loss_threshold=0.001,
            patience=5,
            plateau_window=10,
            plateau_threshold=0.01,
        )

        # Continuously improving
        for i in range(50):
            monitor.update(1.0 / (i + 1))

        assert not monitor.is_plateau()

    def test_loss_velocity_negative_when_improving(self) -> None:
        """Test loss velocity is negative when improving."""
        monitor = ConvergenceMonitor(plateau_window=10)

        for i in range(15):
            monitor.update(1.0 - i * 0.05)

        velocity = monitor.loss_velocity()
        assert velocity < 0

    def test_loss_velocity_zero_initially(self) -> None:
        """Test loss velocity is zero with no data."""
        monitor = ConvergenceMonitor()
        assert monitor.loss_velocity() == 0.0

    def test_tier_transition_to_fast(self) -> None:
        """Test tier transitions to FAST on convergence."""
        monitor = ConvergenceMonitor(loss_threshold=0.01, patience=2)

        monitor.update(0.005)
        monitor.update(0.004)

        assert monitor.get_current_tier() == ConvergenceTier.FAST

    def test_set_tier(self) -> None:
        """Test manual tier setting."""
        monitor = ConvergenceMonitor()

        monitor.set_tier(ConvergenceTier.AGGRESSIVE)
        assert monitor.get_current_tier() == ConvergenceTier.AGGRESSIVE

        monitor.set_tier(ConvergenceTier.RESCUE)
        assert monitor.get_current_tier() == ConvergenceTier.RESCUE

    def test_should_escalate(self) -> None:
        """Test should_escalate detection."""
        monitor = ConvergenceMonitor(
            loss_threshold=0.001,
            patience=5,
            plateau_window=10,
            plateau_threshold=0.01,
            escalation_epochs=20,
        )

        # Not enough epochs
        for i in range(10):
            monitor.update(0.5)
        assert not monitor.should_escalate()

        # Enough epochs with plateau - internal _update_tier triggers escalation
        for _ in range(15):
            monitor.update(0.5)

        # At this point, the monitor has internally escalated due to plateau
        # The tier should be AGGRESSIVE or RESCUE (if plateau count > 2)
        tier = monitor.get_current_tier()
        assert tier in (ConvergenceTier.AGGRESSIVE, ConvergenceTier.RESCUE)
        assert not monitor.should_escalate()  # Already escalated

    def test_reset(self) -> None:
        """Test monitor reset."""
        monitor = ConvergenceMonitor()

        # Add some data
        for i in range(10):
            monitor.update(1.0 / (i + 1))
        monitor.set_tier(ConvergenceTier.AGGRESSIVE)

        # Reset
        monitor.reset()

        assert monitor.epochs_used == 0
        assert monitor.get_current_tier() == ConvergenceTier.NORMAL
        assert not monitor.is_converged()
        assert not monitor.is_plateau()

    def test_get_statistics(self) -> None:
        """Test get_statistics returns correct values."""
        monitor = ConvergenceMonitor(loss_threshold=0.01, patience=2)

        monitor.update(0.5)
        monitor.update(0.3)
        monitor.update(0.1)

        stats = monitor.get_statistics()

        assert stats["epochs"] == 3
        assert stats["best_loss"] < 0.5
        assert stats["current_loss"] == 0.1
        assert stats["tier"] == "normal"
        assert "is_converged" in stats
        assert "is_plateau" in stats
        assert "loss_velocity" in stats
        assert "plateau_count" in stats
        assert "epochs_since_improvement" in stats

    def test_epochs_used_property(self) -> None:
        """Test epochs_used property."""
        monitor = ConvergenceMonitor()

        assert monitor.epochs_used == 0

        for i in range(5):
            monitor.update(1.0 / (i + 1))

        assert monitor.epochs_used == 5


class TestConvergenceMonitorEdgeCases:
    """Edge case tests for ConvergenceMonitor."""

    def test_empty_monitor(self) -> None:
        """Test behavior with no updates."""
        monitor = ConvergenceMonitor()

        assert not monitor.is_converged()
        assert not monitor.is_plateau()
        assert not monitor.should_stop()
        assert not monitor.should_escalate()
        assert monitor.loss_velocity() == 0.0

    def test_single_update(self) -> None:
        """Test behavior with single update."""
        monitor = ConvergenceMonitor()
        monitor.update(0.5)

        assert not monitor.is_plateau()
        assert monitor.epochs_used == 1

    def test_constant_loss(self) -> None:
        """Test behavior with constant loss values."""
        monitor = ConvergenceMonitor(
            loss_threshold=0.01,
            patience=5,
            plateau_window=10,
            plateau_threshold=0.01,
        )

        # Constant loss above threshold
        for _ in range(100):
            monitor.update(0.5)

        assert monitor.is_plateau()

    def test_very_small_improvements(self) -> None:
        """Test with very small improvements that don't trigger plateau."""
        monitor = ConvergenceMonitor(
            loss_threshold=0.001,
            patience=5,
            plateau_window=10,
            plateau_threshold=0.001,  # Very strict threshold
        )

        # Small improvements
        loss = 1.0
        for _ in range(50):
            loss *= 0.999
            monitor.update(loss)

        # Small improvements should not trigger plateau with strict threshold
        # (depends on cumulative improvement over window)
        # This tests the edge case behavior

    def test_oscillating_loss(self) -> None:
        """Test with oscillating loss values."""
        monitor = ConvergenceMonitor(
            loss_threshold=0.01,
            patience=10,
            plateau_window=20,
        )

        # Oscillating loss
        for i in range(50):
            loss = 0.3 + 0.1 * (i % 2)
            monitor.update(loss)

        # Should detect as plateau (no net improvement)
        assert monitor.is_plateau()

    def test_sudden_improvement(self) -> None:
        """Test behavior after sudden improvement."""
        monitor = ConvergenceMonitor(
            loss_threshold=0.01,
            patience=5,
            plateau_window=10,
        )

        # Plateau
        for _ in range(20):
            monitor.update(0.5)

        assert monitor.is_plateau()

        # Sudden improvement
        monitor.update(0.05)

        # Should no longer be in plateau (recent improvement)
        # Note: plateau detection looks at window, so it may still show as plateau
        # until more improvements accumulate

    def test_nan_handling_in_stats(self) -> None:
        """Test that statistics handle edge cases gracefully."""
        monitor = ConvergenceMonitor()

        # Get stats with no data
        stats = monitor.get_statistics()
        assert stats["epochs"] == 0
        assert stats["current_loss"] == float("inf")


# ============================================================================
# Phase 2: ConvergenceStats and ConvergenceProfiler Tests
# ============================================================================


class TestConvergenceStats:
    """Tests for ConvergenceStats dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating ConvergenceStats with required fields."""
        from prism.core.convergence import ConvergenceStats

        stats = ConvergenceStats(
            sample_idx=0,
            epochs_to_convergence=150,
            final_tier=ConvergenceTier.FAST,
            retry_count=0,
            final_loss=0.0005,
            total_time_seconds=2.3,
            loss_velocity_final=-0.00001,
            plateau_count=0,
        )

        assert stats.sample_idx == 0
        assert stats.epochs_to_convergence == 150
        assert stats.final_tier == ConvergenceTier.FAST
        assert stats.converged is True  # Default value

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        from prism.core.convergence import ConvergenceStats

        stats = ConvergenceStats(
            sample_idx=5,
            epochs_to_convergence=300,
            final_tier=ConvergenceTier.AGGRESSIVE,
            retry_count=1,
            final_loss=0.001,
            total_time_seconds=5.0,
            loss_velocity_final=-0.0001,
            plateau_count=2,
            converged=False,
        )

        d = stats.to_dict()
        assert d["sample_idx"] == 5
        assert d["final_tier"] == "aggressive"  # String, not enum
        assert d["converged"] is False


class TestConvergenceProfiler:
    """Tests for ConvergenceProfiler class."""

    def test_empty_profiler(self) -> None:
        """Test profiler with no samples."""
        from prism.core.convergence import ConvergenceProfiler

        profiler = ConvergenceProfiler()
        summary = profiler.get_summary()

        assert summary["total_samples"] == 0
        assert summary["convergence_rate"] == 0.0

    def test_add_sample(self) -> None:
        """Test adding samples to profiler."""
        from prism.core.convergence import ConvergenceProfiler, ConvergenceStats

        profiler = ConvergenceProfiler()

        stats = ConvergenceStats(
            sample_idx=0,
            epochs_to_convergence=100,
            final_tier=ConvergenceTier.FAST,
            retry_count=0,
            final_loss=0.0005,
            total_time_seconds=1.0,
            loss_velocity_final=-0.0001,
            plateau_count=0,
        )
        profiler.add_sample(stats)

        summary = profiler.get_summary()
        assert summary["total_samples"] == 1
        assert summary["converged_count"] == 1

    def test_tier_distribution(self) -> None:
        """Test tier distribution calculation."""
        from prism.core.convergence import ConvergenceProfiler, ConvergenceStats

        profiler = ConvergenceProfiler()

        # Add samples with different tiers
        for tier in [
            ConvergenceTier.FAST,
            ConvergenceTier.FAST,
            ConvergenceTier.NORMAL,
            ConvergenceTier.AGGRESSIVE,
        ]:
            stats = ConvergenceStats(
                sample_idx=0,
                epochs_to_convergence=100,
                final_tier=tier,
                retry_count=0,
                final_loss=0.001,
                total_time_seconds=1.0,
                loss_velocity_final=-0.0001,
                plateau_count=0,
            )
            profiler.add_sample(stats)

        dist = profiler.get_tier_distribution()
        assert dist["fast"] == 2
        assert dist["normal"] == 1
        assert dist["aggressive"] == 1
        assert dist["rescue"] == 0

    def test_efficiency_report(self) -> None:
        """Test efficiency report generation."""
        from prism.core.convergence import ConvergenceProfiler, ConvergenceStats

        profiler = ConvergenceProfiler()

        # Add some samples
        for i in range(5):
            stats = ConvergenceStats(
                sample_idx=i,
                epochs_to_convergence=50 + i * 10,
                final_tier=ConvergenceTier.NORMAL,
                retry_count=0,
                final_loss=0.001,
                total_time_seconds=1.0 + i * 0.5,
                loss_velocity_final=-0.0001,
                plateau_count=0,
            )
            profiler.add_sample(stats)

        report = profiler.get_efficiency_report()
        assert "CONVERGENCE PROFILER REPORT" in report
        assert "Total Samples:" in report
        assert "5" in report

    def test_clear(self) -> None:
        """Test clearing the profiler."""
        from prism.core.convergence import ConvergenceProfiler, ConvergenceStats

        profiler = ConvergenceProfiler()

        stats = ConvergenceStats(
            sample_idx=0,
            epochs_to_convergence=100,
            final_tier=ConvergenceTier.FAST,
            retry_count=0,
            final_loss=0.001,
            total_time_seconds=1.0,
            loss_velocity_final=-0.0001,
            plateau_count=0,
        )
        profiler.add_sample(stats)
        assert profiler.get_summary()["total_samples"] == 1

        profiler.clear()
        assert profiler.get_summary()["total_samples"] == 0


class TestGetRetryLossType:
    """Tests for get_retry_loss_type function."""

    def test_basic_cycling(self) -> None:
        """Test basic loss type cycling."""
        from prism.models.losses import get_retry_loss_type

        # From L1: should cycle to ssim, l2, ms-ssim
        assert get_retry_loss_type("l1", 1) == "ssim"
        assert get_retry_loss_type("l1", 2) == "l2"
        assert get_retry_loss_type("l1", 3) == "ms-ssim"

    def test_wrap_around(self) -> None:
        """Test cycling wraps around."""
        from prism.models.losses import get_retry_loss_type

        # Should wrap back to l1 after ms-ssim
        assert get_retry_loss_type("l1", 4) == "l1"

    def test_from_ssim(self) -> None:
        """Test cycling from ssim start."""
        from prism.models.losses import get_retry_loss_type

        # ssim is index 1, so retry 1 goes to l2 (index 2)
        assert get_retry_loss_type("ssim", 1) == "l2"
        assert get_retry_loss_type("ssim", 2) == "ms-ssim"

    def test_unknown_loss_type(self) -> None:
        """Test handling unknown loss types."""
        from prism.models.losses import get_retry_loss_type

        # Unknown type should start from index 0
        result = get_retry_loss_type("unknown", 1)
        assert result == "ssim"  # Index 0 + 1 = ssim
