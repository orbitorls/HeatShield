"""Circuit breaker pattern for resilient external API calls.

Prevents cascade failures by opening the circuit after consecutive failures,
allowing the system to fall back to alternative data sources.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for external API resilience.

    Args:
        failure_threshold: Number of consecutive failures before opening.
        recovery_timeout_s: Seconds before attempting half-open.
        half_open_max_calls: Max test calls in half-open state.
    """
    name: str
    failure_threshold: int = 5
    recovery_timeout_s: float = 60.0
    half_open_max_calls: int = 3

    _state: CircuitState = field(default=CircuitState.CLOSED, repr=False)
    _failures: int = field(default=0, repr=False)
    _last_failure_time: float = field(default=0.0, repr=False)
    _half_open_calls: int = field(default=0, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute func with circuit breaker protection.

        Supports both sync and async functions.
        """
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if time.monotonic() - self._last_failure_time >= self.recovery_timeout_s:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit %s entering half-open state", self.name)
                else:
                    raise CircuitOpenError(f"Circuit {self.name} is OPEN")

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError(
                        f"Circuit {self.name} half-open limit reached"
                    )
                self._half_open_calls += 1

        # Execute outside lock
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            await self._on_success()
            return result  # type: ignore[return-value]
        except Exception as exc:
            logger.debug("Circuit %s call failed: %s", self.name, exc)
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failures = 0
                self._half_open_calls = 0
                logger.info("Circuit %s closed (recovered)", self.name)
            else:
                self._failures = 0

    async def _on_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit %s re-opened after half-open failure", self.name
                )
            elif self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit %s opened after %d consecutive failures",
                    self.name, self._failures
                )

    def reset(self) -> None:
        """Manually reset circuit to CLOSED."""
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._half_open_calls = 0
        self._last_failure_time = 0.0


class CircuitOpenError(Exception):
    """Raised when circuit breaker is OPEN or half-open limit reached."""


# Global circuit breakers for shared state across requests
_tmd_circuit: CircuitBreaker | None = None


def get_tmd_circuit() -> CircuitBreaker:
    """Return shared TMD circuit breaker instance."""
    global _tmd_circuit
    if _tmd_circuit is None:
        _tmd_circuit = CircuitBreaker(
            name="tmd_api",
            failure_threshold=5,
            recovery_timeout_s=60.0,
            half_open_max_calls=2,
        )
    return _tmd_circuit


def reset_tmd_circuit() -> None:
    """Reset TMD circuit breaker (useful for testing)."""
    global _tmd_circuit
    if _tmd_circuit is not None:
        _tmd_circuit.reset()
