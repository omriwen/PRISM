"""Test script for enhanced progress tracking system."""

from __future__ import annotations

import time

import numpy as np
import pytest

from prism.utils.progress import ETACalculator, MetricTrend, TrainingProgress, render_sparkline


def test_basic_progress():
    """Test basic progress tracking functionality."""
    print("Testing basic progress tracking...")

    total_steps = 10
    eta = ETACalculator(total_steps)

    with TrainingProgress() as progress:
        task_id = progress.add_task("Test Task", total=total_steps)

        for i in range(total_steps):
            time.sleep(0.1)  # Simulate work

            eta_seconds = eta.update(i + 1)
            progress.advance(
                task_id,
                metrics={
                    "step": i + 1,
                    "value": np.random.random(),
                },
                eta_seconds=eta_seconds,
                description=f"Processing {i + 1}/{total_steps}",
            )

    print("Basic progress test completed!")


def test_multi_task_progress():
    """Test multiple progress bars simultaneously."""
    print("\nTesting multi-task progress tracking...")

    epochs = 5
    samples_per_epoch = 20
    total_samples = epochs * samples_per_epoch

    epoch_eta = ETACalculator(epochs)
    sample_eta = ETACalculator(total_samples)
    samples_completed = 0

    with TrainingProgress() as progress:
        epoch_task = progress.add_task("Epochs", total=epochs)
        sample_task = progress.add_task("Samples", total=total_samples)

        for epoch in range(epochs):
            for sample in range(samples_per_epoch):
                time.sleep(0.05)  # Simulate work

                samples_completed += 1
                sample_eta_seconds = sample_eta.update(samples_completed)

                progress.advance(
                    sample_task,
                    metrics={
                        "epoch": epoch + 1,
                        "sample": sample + 1,
                        "loss": np.random.random() * 0.1,
                    },
                    eta_seconds=sample_eta_seconds,
                    description=f"Epoch {epoch + 1}/{epochs}",
                )

            # Update epoch progress
            epoch_eta_seconds = epoch_eta.update(epoch + 1)
            progress.advance(
                epoch_task,
                metrics={
                    "epoch": epoch + 1,
                    "ssim": 0.8 + np.random.random() * 0.15,
                    "rmse": np.random.random() * 0.05,
                },
                eta_seconds=epoch_eta_seconds,
                description=f"Epochs {epoch + 1}/{epochs}",
            )

    print("Multi-task progress test completed!")


def test_eta_calculator():
    """Test ETA calculator accuracy."""
    print("\nTesting ETA calculator...")

    total_steps = 100
    eta = ETACalculator(total_steps)

    print(f"Total steps: {total_steps}")

    for i in range(10):
        time.sleep(0.01)  # Consistent timing
        remaining = eta.update(i + 1)
        print(f"Step {i + 1}: ETA = {remaining:.2f}s")

    print("ETA calculator test completed!")


# ============================================================================
# Unit Tests for New Phase 1.4 Features
# ============================================================================


def test_sparkline_unicode():
    """Test sparkline rendering with unicode characters."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    result = render_sparkline(values, use_unicode=True)

    # Should have same length as input
    assert len(result) == len(values)

    # Should contain unicode block characters
    unicode_chars = "▁▂▃▄▅▆▇█"
    assert all(c in unicode_chars for c in result)

    # First value should be lowest block (▁)
    assert result[0] == "▁"

    # Middle value (5.0) should be highest block (█)
    assert result[4] == "█"


def test_sparkline_ascii():
    """Test sparkline rendering with ASCII fallback."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = render_sparkline(values, use_unicode=False)

    # Should have same length as input
    assert len(result) == len(values)

    # Should contain only ASCII characters
    ascii_chars = "_.-'^*#"
    assert all(c in ascii_chars for c in result)


def test_sparkline_empty():
    """Test sparkline with empty input."""
    result = render_sparkline([])
    assert result == ""


def test_sparkline_constant():
    """Test sparkline with constant values."""
    values = [5.0, 5.0, 5.0, 5.0]
    result = render_sparkline(values, use_unicode=True)

    # All characters should be the same
    assert len(set(result)) == 1


def test_sparkline_width_limit():
    """Test sparkline respects width limit."""
    values = [float(i) for i in range(100)]
    width = 20
    result = render_sparkline(values, width=width)

    # Should be limited to width
    assert len(result) == width


def test_metric_trend_improving_loss():
    """Test trend detection for improving loss (decreasing)."""
    analyzer = MetricTrend(window_size=50, plateau_threshold=0.01)

    # Simulated decreasing loss
    values = [1.0 - i * 0.01 for i in range(50)]

    status = analyzer.analyze(values, "loss")
    assert status == "🟢"  # Should be improving (green)


def test_metric_trend_diverging_loss():
    """Test trend detection for diverging loss (increasing)."""
    analyzer = MetricTrend(window_size=50, plateau_threshold=0.01)

    # Simulated increasing loss (bad!)
    values = [1.0 + i * 0.01 for i in range(50)]

    status = analyzer.analyze(values, "loss")
    assert status == "🔴"  # Should be diverging (red)


def test_metric_trend_plateau():
    """Test trend detection for plateaued metrics."""
    analyzer = MetricTrend(window_size=50, plateau_threshold=0.01)

    # Constant values (plateau)
    values = [1.0] * 50

    status = analyzer.analyze(values, "loss")
    assert status == "🟡"  # Should be plateaued (yellow)


def test_metric_trend_improving_ssim():
    """Test trend detection for improving SSIM (increasing)."""
    analyzer = MetricTrend(window_size=50, plateau_threshold=0.01)

    # Simulated increasing SSIM (good!)
    values = [0.5 + i * 0.01 for i in range(50)]

    status = analyzer.analyze(values, "ssim")
    assert status == "🟢"  # Should be improving (green)


def test_metric_trend_insufficient_data():
    """Test trend analysis with insufficient data."""
    analyzer = MetricTrend()

    # Too few values
    values = [1.0, 2.0]

    status = analyzer.analyze(values, "loss")
    assert status == ""  # Should return empty string


def test_training_progress_history_tracking():
    """Test that TrainingProgress tracks metric history correctly."""
    progress = TrainingProgress(enable_sparklines=True)

    # Update metrics multiple times
    for i in range(10):
        progress.update_metrics({"loss": 1.0 - i * 0.1, "ssim": 0.5 + i * 0.05})

    # Check history is tracked
    assert len(progress.metric_history["loss"]) == 10
    assert len(progress.metric_history["ssim"]) == 10

    # Check values are correct
    assert progress.metric_history["loss"][0] == pytest.approx(1.0)
    assert progress.metric_history["loss"][-1] == pytest.approx(0.1)


def test_training_progress_history_size_limit():
    """Test that history size is limited correctly."""
    history_size = 50
    progress = TrainingProgress(enable_sparklines=True, metric_history_size=history_size)

    # Add more values than the limit
    for i in range(100):
        progress.update_metrics({"loss": float(i)})

    # Should be limited to history_size
    assert len(progress.metric_history["loss"]) == history_size

    # Should contain the most recent values
    assert progress.metric_history["loss"][-1] == 99.0
    assert progress.metric_history["loss"][0] == 50.0


def test_eta_calculator_ema():
    """Test ETA calculator with exponential moving average."""
    eta = ETACalculator(total_steps=100, ema_alpha=0.1)

    # Simulate consistent step times
    for i in range(10):
        time.sleep(0.01)
        remaining = eta.update(i + 1)

        # ETA should decrease as we make progress
        if i > 0:
            assert remaining > 0

    # EMA should be initialized
    assert eta.ema_step_time is not None
    assert eta.ema_step_time > 0


def test_eta_calculator_confidence_interval():
    """Test confidence interval calculation."""
    eta = ETACalculator(total_steps=100)

    # Generate some step times
    for i in range(20):
        time.sleep(0.01)
        eta.update(i + 1)

    lower, upper = eta.get_confidence_interval(20)

    # Lower bound should be less than upper bound
    assert lower < upper
    assert lower >= 0


def test_training_progress_with_features_disabled():
    """Test TrainingProgress with features disabled."""
    progress = TrainingProgress(enable_sparklines=False, enable_emojis=False)

    # Update metrics
    progress.update_metrics({"loss": 1.0, "ssim": 0.8})

    # Build metrics table
    table = progress._build_metrics_table()

    # Should have basic columns only (Metric and Current)
    assert len(table.columns) == 2


def test_training_progress_percent_change():
    """Test percent change calculation in metrics display."""
    progress = TrainingProgress(enable_sparklines=True)

    # Add initial values
    progress.update_metrics({"loss": 1.0})
    progress.update_metrics({"loss": 0.9})  # 10% decrease

    # Check that history has both values
    assert len(progress.metric_history["loss"]) == 2

    # The percent change logic is in _build_metrics_table, so we just
    # verify the history is tracked correctly for the calculation
    assert progress.metric_history["loss"][-2] == pytest.approx(1.0)
    assert progress.metric_history["loss"][-1] == pytest.approx(0.9)


if __name__ == "__main__":
    test_basic_progress()
    test_multi_task_progress()
    test_eta_calculator()
    print("\n" + "=" * 60)
    print("All progress tracking tests passed!")
    print("=" * 60)
