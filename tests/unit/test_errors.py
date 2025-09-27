"""Unit tests for centralized error types and retry decorator."""

import time
import pytest

from src import errors


def test_exceptions_hierarchy():
    assert issubclass(errors.IngestionError, Exception)
    assert issubclass(errors.VectorStoreError, Exception)
    assert issubclass(errors.TransientError, Exception)


def test_retry_decorator_succeeds_after_retries(monkeypatch):
    calls = {"count": 0}

    class MyTransient(errors.TransientError):
        pass

    @errors.retry((MyTransient,), retries=2, backoff_factor=0.01)
    def flaky():
        calls["count"] += 1
        if calls["count"] < 2:
            raise MyTransient("temporary")
        return "ok"

    start = time.time()
    assert flaky() == "ok"
    duration = time.time() - start
    # Should have retried at least once (small backoff applied)
    assert calls["count"] == 2
    assert duration >= 0


def test_retry_decorator_raises_after_max_attempts():
    class MyTransient(errors.TransientError):
        pass

    @errors.retry((MyTransient,), retries=1, backoff_factor=0.01)
    def always_fail():
        raise MyTransient("still failing")

    with pytest.raises(MyTransient):
        always_fail()
