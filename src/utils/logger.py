"""
Logging configuration and utilities for the Real-time RAG system.
"""

import logging
import sys
from typing import Optional
from pathlib import Path

import structlog
from rich.console import Console
from rich.logging import RichHandler

from ..config import LoggingConfig


def setup_logger(config: LoggingConfig) -> None:
    """
    Setup structured logging with Rich formatting.

    Args:
        config: Logging configuration
    """
    # Clear any existing handlers
    logging.getLogger().handlers.clear()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.level.upper())
        ),
        logger_factory=structlog.WriteLoggerFactory(),
        context_class=dict,
        cache_logger_on_first_use=True,
    )

    # Setup console handler with Rich formatting
    console = Console(stderr=True)
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(getattr(logging, config.level.upper()))

    # Setup file handler if path is provided
    handlers = [console_handler]

    if config.file_path:
        file_path = Path(config.file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(getattr(logging, config.level.upper()))
        file_handler.setFormatter(logging.Formatter(config.format))
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.level.upper()),
        handlers=handlers,
        format=config.format,
    )

    # Set third-party library log levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("multipart").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


class LoggerMixin:
    """
    Mixin class to add logging capabilities to any class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)

    def log_info(self, message: str, **kwargs):
        """Log info message with context."""
        self.logger.info(message, **kwargs)

    def log_warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self.logger.warning(message, **kwargs)

    def log_error(self, message: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with context."""
        if error:
            kwargs["error"] = str(error)
            kwargs["error_type"] = type(error).__name__
        self.logger.error(message, **kwargs)

    def log_debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self.logger.debug(message, **kwargs)


def log_function_call(func):
    """
    Decorator to log function calls with arguments and results.

    Usage:
        @log_function_call
        def my_function(arg1, arg2):
            return arg1 + arg2
    """

    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)

        logger.debug(
            f"Calling {func.__name__}",
            args=args[:3] if len(args) > 3 else args,  # Limit args to prevent log spam
            kwargs=kwargs,
        )

        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(
                f"{func.__name__} failed", error=str(e), error_type=type(e).__name__
            )
            raise

    return wrapper


async def log_async_function_call(func):
    """
    Async decorator to log function calls with arguments and results.

    Usage:
        @log_async_function_call
        async def my_async_function(arg1, arg2):
            return arg1 + arg2
    """

    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)

        logger.debug(
            f"Calling async {func.__name__}",
            args=args[:3] if len(args) > 3 else args,
            kwargs=kwargs,
        )

        try:
            result = await func(*args, **kwargs)
            logger.debug(f"Async {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(
                f"Async {func.__name__} failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    return wrapper


class ContextLogger:
    """
    Context manager for logging with additional context.

    Usage:
        with ContextLogger("processing_feed", feed_url=url) as logger:
            logger.info("Starting processing")
            # ... do work ...
            logger.info("Processing completed")
    """

    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
        self.logger = get_logger("context")

    def __enter__(self):
        self.logger = self.logger.bind(operation=self.operation, **self.context)
        self.logger.info(f"Starting {self.operation}")
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(
                f"{self.operation} failed",
                error=str(exc_val),
                error_type=exc_type.__name__,
            )
        else:
            self.logger.info(f"{self.operation} completed successfully")


# Convenience functions
def setup_basic_logging(level: str = "INFO") -> None:
    """Setup basic logging for simple use cases."""
    config = LoggingConfig(level=level)
    setup_logger(config)


def get_file_logger(name: str, file_path: str, level: str = "INFO") -> logging.Logger:
    """
    Get a logger that writes to a specific file.

    Args:
        name: Logger name
        file_path: Path to log file
        level: Log level

    Returns:
        Configured file logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Check if handler already exists
    if not logger.handlers:
        handler = logging.FileHandler(file_path)
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)

    return logger
