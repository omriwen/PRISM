"""Dashboard launcher for managing dashboard subprocess during training."""

from __future__ import annotations

import multiprocessing
import socket
import time
from pathlib import Path
from typing import Optional

from loguru import logger


class DashboardLauncher:
    """Launch and manage dashboard process alongside training.

    This class provides a clean way to start the dashboard in a separate process
    and ensures proper cleanup when training completes or fails.

    Parameters
    ----------
    runs_dir : Path
        Directory containing experiment runs
    port : int
        Port number for dashboard server (default: 8050)

    Examples
    --------
    >>> launcher = DashboardLauncher(port=8050)
    >>> launcher.start()
    >>> # ... run training ...
    >>> launcher.stop()

    Or use as context manager:
    >>> with DashboardLauncher(port=8050) as launcher:
    ...     # ... run training ...
    ...     pass  # Dashboard stops automatically
    """

    def __init__(self, runs_dir: Path = Path("runs"), port: int = 8050):
        """Initialize dashboard launcher.

        Parameters
        ----------
        runs_dir : Path
            Directory containing experiment runs
        port : int
            Port number for dashboard server
        """
        self.runs_dir = Path(runs_dir)
        self.port = port
        self.process: Optional[multiprocessing.Process] = None
        self._started = False

    def is_port_available(self) -> bool:
        """Check if the specified port is available.

        Returns
        -------
        bool
            True if port is available, False otherwise
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("localhost", self.port))
                return True
            except socket.error:
                return False

    def start(self) -> bool:
        """Launch dashboard in background process.

        Returns
        -------
        bool
            True if dashboard started successfully, False otherwise
        """
        if self._started:
            logger.warning("Dashboard already started")
            return False

        # Check if port is available
        if not self.is_port_available():
            logger.warning(
                f"Port {self.port} is already in use. Dashboard not started. "
                f"You may already have a dashboard running or another service using this port."
            )
            return False

        try:
            # Create process to run dashboard
            logger.info(f"Launching dashboard on port {self.port}...")

            # Use multiprocessing instead of subprocess for better cross-platform support
            # and cleaner shutdown
            from prism.web.dashboard import run_dashboard

            self.process = multiprocessing.Process(
                target=run_dashboard,
                kwargs={"runs_dir": self.runs_dir, "port": self.port, "debug": False},
                daemon=True,
            )
            self.process.start()

            # Wait a moment and check if process started successfully
            time.sleep(2)

            if self.process.is_alive():
                self._started = True
                logger.success(
                    f"Dashboard started successfully!\n"
                    f"  URL: http://localhost:{self.port}\n"
                    f"  Monitoring: {self.runs_dir.absolute()}\n"
                )

                # Print additional information for remote training
                logger.info(
                    "For remote training, use SSH port forwarding:\n"
                    f"  ssh -L {self.port}:localhost:{self.port} user@remote-host\n"
                    f"  Then access: http://localhost:{self.port}"
                )
                return True
            else:
                logger.error("Dashboard process failed to start")
                return False

        except Exception as e:  # noqa: BLE001 - Process launcher must handle all startup errors
            logger.error(f"Error starting dashboard: {e}")
            return False

    def stop(self) -> None:
        """Stop dashboard process.

        Gracefully terminates the dashboard subprocess if it's running.
        """
        if not self._started:
            return

        if self.process is not None:
            try:
                logger.info("Stopping dashboard...")
                self.process.terminate()

                # Wait up to 5 seconds for graceful shutdown
                self.process.join(timeout=5)

                # Force kill if still alive
                if self.process.is_alive():
                    logger.warning("Dashboard did not stop gracefully, forcing shutdown...")
                    self.process.kill()
                    self.process.join(timeout=2)

                logger.info("Dashboard stopped")
            except Exception as e:  # noqa: BLE001 - Cleanup must not raise
                logger.error(f"Error stopping dashboard: {e}")
            finally:
                self.process = None
                self._started = False

    def is_running(self) -> bool:
        """Check if dashboard is currently running.

        Returns
        -------
        bool
            True if dashboard process is running, False otherwise
        """
        return self._started and self.process is not None and self.process.is_alive()

    def __enter__(self):
        """Context manager entry: start dashboard."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: stop dashboard."""
        self.stop()
        return False  # Don't suppress exceptions

    def __del__(self):
        """Destructor: ensure dashboard is stopped."""
        if self._started:
            self.stop()


def launch_dashboard_if_requested(
    args, runs_dir: Optional[Path] = None
) -> Optional[DashboardLauncher]:
    """Helper function to launch dashboard based on command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments
    runs_dir : Optional[Path]
        Directory containing experiment runs (defaults to args.log_dir)

    Returns
    -------
    Optional[DashboardLauncher]
        Launcher instance if dashboard was started, None otherwise

    Examples
    --------
    >>> launcher = launch_dashboard_if_requested(args)
    >>> if launcher:
    ...     try:
    ...         run_training()
    ...     finally:
    ...         launcher.stop()
    """
    # Check if dashboard flag is set
    if not hasattr(args, "dashboard") or not args.dashboard:
        return None

    # Determine runs directory
    if runs_dir is None:
        runs_dir = Path(args.log_dir if hasattr(args, "log_dir") else "runs")

    # Get port
    port = args.dashboard_port if hasattr(args, "dashboard_port") else 8050

    # Create and start launcher
    launcher = DashboardLauncher(runs_dir=runs_dir, port=port)

    if launcher.start():
        return launcher
    else:
        logger.warning("Dashboard launch failed, continuing without dashboard")
        return None
