"""Unit tests for dashboard launcher module."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, Mock, patch

import pytest

from prism.web.launcher import DashboardLauncher, launch_dashboard_if_requested


class TestDashboardLauncher:
    """Tests for DashboardLauncher class."""

    def test_initialization(self, tmp_path):
        """Test launcher initialization."""
        launcher = DashboardLauncher(runs_dir=tmp_path, port=8050)

        assert launcher.runs_dir == tmp_path
        assert launcher.port == 8050
        assert launcher.process is None
        assert launcher._started is False

    def test_is_port_available_free_port(self):
        """Test port availability check for free port."""
        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            free_port = s.getsockname()[1]

        launcher = DashboardLauncher(port=free_port)
        assert launcher.is_port_available() is True

    def test_is_port_available_occupied_port(self):
        """Test port availability check for occupied port."""
        # Create a server to occupy a port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            occupied_port = s.getsockname()[1]
            s.listen(1)

            launcher = DashboardLauncher(port=occupied_port)
            assert launcher.is_port_available() is False

    @patch("prism.web.launcher.multiprocessing.Process")
    @patch.object(DashboardLauncher, "is_port_available")
    def test_start_success(self, mock_port_check, mock_process_class, tmp_path):
        """Test successful dashboard start."""
        # Mock the process
        mock_process = Mock()
        mock_process.is_alive.return_value = True
        mock_process_class.return_value = mock_process

        # Mock port availability
        mock_port_check.return_value = True

        launcher = DashboardLauncher(runs_dir=tmp_path, port=8050)
        result = launcher.start()

        assert result is True
        assert launcher._started is True
        assert launcher.process is not None
        mock_process.start.assert_called_once()

    @patch.object(DashboardLauncher, "is_port_available")
    def test_start_port_not_available(self, mock_port_check, tmp_path):
        """Test dashboard start when port is not available."""
        # Mock port as not available
        mock_port_check.return_value = False

        launcher = DashboardLauncher(runs_dir=tmp_path, port=8050)
        result = launcher.start()

        assert result is False
        assert launcher._started is False

    def test_start_already_started(self, tmp_path):
        """Test starting dashboard when already started."""
        launcher = DashboardLauncher(runs_dir=tmp_path, port=8050)
        launcher._started = True

        result = launcher.start()

        assert result is False

    def test_stop(self, tmp_path):
        """Test stopping dashboard."""
        launcher = DashboardLauncher(runs_dir=tmp_path, port=8050)

        # Mock process
        mock_process = Mock()
        mock_process.is_alive.return_value = False
        launcher.process = mock_process
        launcher._started = True

        launcher.stop()

        mock_process.terminate.assert_called_once()
        mock_process.join.assert_called()
        assert launcher._started is False
        assert launcher.process is None

    def test_stop_not_started(self, tmp_path):
        """Test stopping dashboard when not started."""
        launcher = DashboardLauncher(runs_dir=tmp_path, port=8050)

        # Should not raise any errors
        launcher.stop()

        assert launcher._started is False

    def test_is_running_true(self, tmp_path):
        """Test is_running when dashboard is running."""
        launcher = DashboardLauncher(runs_dir=tmp_path, port=8050)

        mock_process = Mock()
        mock_process.is_alive.return_value = True
        launcher.process = mock_process
        launcher._started = True

        assert launcher.is_running() is True

    def test_is_running_false(self, tmp_path):
        """Test is_running when dashboard is not running."""
        launcher = DashboardLauncher(runs_dir=tmp_path, port=8050)
        assert launcher.is_running() is False

    @patch("prism.web.launcher.multiprocessing.Process")
    def test_context_manager(self, mock_process_class, tmp_path):
        """Test using launcher as context manager."""
        # Mock the process
        mock_process = Mock()
        mock_process.is_alive.return_value = True
        mock_process_class.return_value = mock_process

        # Mock port availability
        with patch.object(DashboardLauncher, "is_port_available", return_value=True):
            with DashboardLauncher(runs_dir=tmp_path, port=8050) as launcher:
                assert launcher._started is True

            # After exiting context, should be stopped
            mock_process.terminate.assert_called()


class TestLaunchDashboardIfRequested:
    """Tests for launch_dashboard_if_requested helper function."""

    def test_launch_not_requested(self, tmp_path):
        """Test when dashboard is not requested."""
        args = MagicMock()
        args.dashboard = False
        args.log_dir = str(tmp_path)

        result = launch_dashboard_if_requested(args)

        assert result is None

    def test_launch_requested_no_dashboard_attr(self, tmp_path):
        """Test when args doesn't have dashboard attribute."""
        args = MagicMock()
        delattr(args, "dashboard")
        args.log_dir = str(tmp_path)

        result = launch_dashboard_if_requested(args)

        assert result is None

    @patch("prism.web.launcher.DashboardLauncher")
    def test_launch_requested_success(self, mock_launcher_class, tmp_path):
        """Test successful dashboard launch via helper."""
        # Mock args
        args = MagicMock()
        args.dashboard = True
        args.log_dir = str(tmp_path)
        args.dashboard_port = 8050

        # Mock launcher
        mock_launcher = Mock()
        mock_launcher.start.return_value = True
        mock_launcher_class.return_value = mock_launcher

        result = launch_dashboard_if_requested(args, runs_dir=tmp_path)

        assert result is not None
        assert result == mock_launcher
        mock_launcher.start.assert_called_once()

    @patch("prism.web.launcher.DashboardLauncher")
    def test_launch_requested_failure(self, mock_launcher_class, tmp_path):
        """Test failed dashboard launch via helper."""
        # Mock args
        args = MagicMock()
        args.dashboard = True
        args.log_dir = str(tmp_path)
        args.dashboard_port = 8050

        # Mock launcher that fails to start
        mock_launcher = Mock()
        mock_launcher.start.return_value = False
        mock_launcher_class.return_value = mock_launcher

        result = launch_dashboard_if_requested(args, runs_dir=tmp_path)

        assert result is None

    @patch("prism.web.launcher.DashboardLauncher")
    def test_launch_uses_default_port(self, mock_launcher_class, tmp_path):
        """Test that default port is used when not specified."""
        # Mock args without dashboard_port
        args = MagicMock()
        args.dashboard = True
        args.log_dir = str(tmp_path)
        delattr(args, "dashboard_port")

        # Mock launcher
        mock_launcher = Mock()
        mock_launcher.start.return_value = True
        mock_launcher_class.return_value = mock_launcher

        launch_dashboard_if_requested(args, runs_dir=tmp_path)

        # Verify DashboardLauncher was called with default port 8050
        mock_launcher_class.assert_called_once_with(runs_dir=tmp_path, port=8050)


@pytest.mark.integration
class TestDashboardLauncherIntegration:
    """Integration tests for dashboard launcher (requires full dependencies)."""

    @pytest.mark.skip(
        reason="Requires full dashboard dependencies and may conflict with other tests"
    )
    def test_full_start_stop_cycle(self, tmp_path):
        """Test full start/stop cycle with real process."""
        # Create temporary runs directory
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        # Find a free port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            free_port = s.getsockname()[1]

        launcher = DashboardLauncher(runs_dir=runs_dir, port=free_port)

        try:
            # Start dashboard
            result = launcher.start()
            assert result is True
            assert launcher.is_running() is True

            # Give it a moment to fully initialize
            import time

            time.sleep(3)

            # Verify process is still running
            assert launcher.is_running() is True
        finally:
            # Stop dashboard
            launcher.stop()
            assert launcher.is_running() is False
