"""
Configure logging for scripts and experiments.

Provides reusable logging setup that can write to both file and console
with customizable formatting and log levels.
"""

import logging
from pathlib import Path
from typing import Optional


class LoggingConfigurator:
    """
    Configure logging with file and console handlers.

    Supports:
    - Dual output (file + console)
    - Custom log levels
    - Custom formatting
    - Automatic log file creation in output directories
    """

    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    SIMPLE_FORMAT = '%(levelname)s: %(message)s'
    DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - ' \
        '[%(filename)s:%(lineno)d] - %(message)s'

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        log_filename: str = 'log.txt',
        file_level: int = logging.INFO,
        console_level: int = logging.INFO,
        format_string: Optional[str] = None,
        mode: str = 'w'
    ):
        """
        Initialize logging configurator.

        Args:
            output_dir: Directory where log file will be saved.
                If None, only console logging.
            log_filename: Name of the log file (default: 'log.txt')
            file_level: Logging level for file output (default: INFO)
            console_level: Logging level for console output (default: INFO)
            format_string: Custom format string (default: DEFAULT_FORMAT)
            mode: File mode ('w' for overwrite, 'a' for append)
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.log_filename = log_filename
        self.file_level = file_level
        self.console_level = console_level
        self.format_string = format_string or self.DEFAULT_FORMAT
        self.mode = mode
        self.log_file_path = None

    def setup(self) -> Optional[Path]:
        """
        Configure logging with file and/or console handlers.

        Returns:
            Path to log file if created, None if console-only
        """
        # Create formatter
        formatter = logging.Formatter(self.format_string)

        # Get root logger
        root_logger = logging.getLogger()
        # Capture all, filter at handler level
        root_logger.setLevel(logging.DEBUG)

        # Clear any existing handlers
        root_logger.handlers.clear()

        # Console handler (always added)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File handler (optional)
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_file_path = self.output_dir / self.log_filename

            file_handler = logging.FileHandler(self.log_file_path,
                                               mode=self.mode)
            file_handler.setLevel(self.file_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        return self.log_file_path

    def get_log_file_path(self) -> Optional[Path]:
        """Get path to the log file if it was created."""
        return self.log_file_path

    @classmethod
    def setup_basic(
        cls,
        output_dir: Path,
        log_filename: str = 'log.txt'
            ) -> Path:
        """
        Quick setup with default settings.

        Args:
            output_dir: Directory for log file
            log_filename: Name of log file
        Returns:
            Path to created log file
        """
        configurator = cls(output_dir=output_dir, log_filename=log_filename)
        return configurator.setup()

    @classmethod
    def setup_console_only(
            cls, level: int = logging.INFO,
            format_string: Optional[str] = None
            ) -> None:
        """
        Setup console-only logging (no file output).

        Args:
            level: Logging level
            format_string: Optional custom format
        """
        configurator = cls(
            output_dir=None,
            console_level=level,
            format_string=format_string
        )
        configurator.setup()

    @classmethod
    def setup_detailed(
            cls,
            output_dir: Path,
            log_filename: str = 'detailed_log.txt'
            ) -> Path:
        """
        Setup with detailed formatting including filename and line numbers.

        Args:
            output_dir: Directory for log file
            log_filename: Name of log file

        Returns:
            Path to created log file
        """
        configurator = cls(
            output_dir=output_dir,
            log_filename=log_filename,
            format_string=cls.DETAILED_FORMAT
        )
        return configurator.setup()


# Usage examples:
#
# # Basic setup with file and console:
# from loggers import LoggingConfigurator
# log_file = LoggingConfigurator.setup_basic(
#       Path('./results'), 'my_analysis.log')
#
# # Console only:
# LoggingConfigurator.setup_console_only()
#
# # Custom configuration:
# config = LoggingConfigurator(
#     output_dir=Path('./results'),
#     log_filename='experiment.log',
#     file_level=logging.DEBUG,
#     console_level=logging.WARNING,
#     format_string='%(asctime)s - %(message)s'
# )
# log_file = config.setup()
#
# # Detailed logging:
# log_file = LoggingConfigurator.setup_detailed(Path('./results'))
