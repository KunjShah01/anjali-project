"""Centralized error types and utilities for the project.

Provides custom exceptions and a simple retry decorator with exponential backoff.
"""
from __future__ import annotations

import time
import functools
import asyncio
import inspect
from typing import Callable, Type, Tuple, Any


class IngestionError(Exception):
    """Base exception for ingestion-related failures."""


class VectorStoreError(Exception):
    """Base exception for vector store related failures."""


class TransientError(Exception):
    """Indicates an error that may succeed if retried (network, timeouts)."""


def retry(
    exceptions: Tuple[Type[BaseException], ...] = (TransientError,),
    retries: int = 3,
    backoff_factor: float = 0.5,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """A simple retry decorator with exponential backoff.

    Args:
        exceptions: Tuple of exception types that should trigger a retry.
        retries: Number of retry attempts (not counting initial call).
        backoff_factor: Base backoff in seconds, multiplied exponentially.

    Usage:
        @retry((TransientError,), retries=3, backoff_factor=0.2)
        def flaky():
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Support both sync and async functions
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt = 0
                while True:
                    try:
                        return await func(*args, **kwargs)
                    except exceptions:
                        if attempt >= retries:
                            raise
                        sleep_time = backoff_factor * (2 ** attempt)
                        await asyncio.sleep(sleep_time)
                        attempt += 1

            return async_wrapper  # type: ignore

        else:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                attempt = 0
                while True:
                    try:
                        return func(*args, **kwargs)
                    except exceptions:
                        if attempt >= retries:
                            raise
                        sleep_time = backoff_factor * (2 ** attempt)
                        time.sleep(sleep_time)
                        attempt += 1

            return wrapper

    return decorator
