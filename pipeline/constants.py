"""Pipeline Autonomo Constants.

CQ-008 FIX: Centralized magic numbers and constants for the pipeline.

This module provides a single source of truth for all constants used
across the pipeline, making them easier to maintain and configure.

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-22)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


# =============================================================================
# TIMEOUTS
# =============================================================================

@dataclass(frozen=True)
class Timeouts:
    """Timeout constants in seconds.

    GAP-FIX: Hardcoded timeouts centralized here.
    Use these constants instead of magic numbers in code.
    """

    # Network/HTTP timeouts
    HTTP_DEFAULT: Final[int] = 30
    HTTP_LONG: Final[int] = 60
    HTTP_VERY_LONG: Final[int] = 120
    HTTP_CONNECT: Final[int] = 10

    # LLM/Claude timeouts
    LLM_DEFAULT: Final[int] = 120
    LLM_LONG: Final[int] = 300
    LLM_VERY_LONG: Final[int] = 600
    LLM_SAFETY_CHECK: Final[int] = 30
    LLM_REFLEXION: Final[int] = 30

    # Database timeouts
    DB_DEFAULT: Final[int] = 30
    DB_TRANSACTION: Final[int] = 60
    REDIS_DEFAULT: Final[int] = 10

    # Gate timeouts
    GATE_DEFAULT: Final[int] = 60
    GATE_COMPLEX: Final[int] = 120
    GATE_DAG_FULL: Final[int] = 600  # 10 minutes for full DAG execution

    # Process/subprocess timeouts
    PROCESS_WAIT: Final[int] = 10
    PROCESS_KILL: Final[int] = 5
    SUBPROCESS_SHORT: Final[int] = 2
    SUBPROCESS_MEDIUM: Final[int] = 5

    # MCP timeouts
    MCP_CONNECT: Final[int] = 60

    # Model download (can be very long)
    MODEL_DOWNLOAD: Final[int] = 600

    # E2E/Behavioral test timeouts
    E2E_TEST: Final[int] = 30

    # WebSocket timeouts
    WEBSOCKET_PING: Final[int] = 60

    # Thread/async pool timeouts
    THREAD_JOIN: Final[float] = 5.0
    ASYNC_POOL: Final[int] = 30

    # Document generation
    DOC_GENERATION: Final[int] = 60

    # Progress check interval
    PROGRESS_CHECK_INTERVAL: Final[int] = 300


TIMEOUTS = Timeouts()


# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class RetryLimits:
    """Retry configuration constants."""

    # Default retry limits
    DEFAULT_RETRIES: Final[int] = 3
    NETWORK_RETRIES: Final[int] = 5
    LLM_RETRIES: Final[int] = 3
    CRITICAL_RETRIES: Final[int] = 10

    # Delays (seconds)
    BASE_DELAY: Final[float] = 1.0
    MAX_DELAY: Final[float] = 60.0
    MULTIPLIER: Final[float] = 2.0
    JITTER_FACTOR: Final[float] = 0.25


RETRY_LIMITS = RetryLimits()


# =============================================================================
# CODE QUALITY LIMITS
# =============================================================================

@dataclass(frozen=True)
class CodeQualityLimits:
    """Code quality threshold constants."""

    # Function limits
    MAX_FUNCTION_LINES: Final[int] = 100
    MAX_CYCLOMATIC_COMPLEXITY: Final[int] = 10
    MAX_FUNCTION_PARAMETERS: Final[int] = 7

    # Class limits
    MAX_CLASS_METHODS: Final[int] = 20
    MAX_CLASS_LINES: Final[int] = 500

    # Module limits
    MAX_MODULE_LINES: Final[int] = 1000

    # Documentation
    MIN_DOCSTRING_LENGTH: Final[int] = 10


CODE_QUALITY_LIMITS = CodeQualityLimits()


# =============================================================================
# COVERAGE THRESHOLDS
# =============================================================================

@dataclass(frozen=True)
class CoverageThresholds:
    """Test coverage threshold constants."""

    # Line coverage
    LINE_COVERAGE_MIN: Final[float] = 0.80
    LINE_COVERAGE_TARGET: Final[float] = 0.90
    LINE_COVERAGE_EXCELLENT: Final[float] = 0.95

    # Branch coverage
    BRANCH_COVERAGE_MIN: Final[float] = 0.70
    BRANCH_COVERAGE_TARGET: Final[float] = 0.80

    # Mutation testing
    MUTATION_KILL_RATE_MIN: Final[float] = 0.60
    MUTATION_KILL_RATE_TARGET: Final[float] = 0.70
    MUTATION_KILL_RATE_CRITICAL: Final[float] = 0.80


COVERAGE_THRESHOLDS = CoverageThresholds()


# =============================================================================
# GATE THRESHOLDS
# =============================================================================

@dataclass(frozen=True)
class GateThresholds:
    """Gate validation threshold constants."""

    # Pass thresholds
    DEFAULT_PASS_THRESHOLD: Final[float] = 0.70
    STRICT_PASS_THRESHOLD: Final[float] = 0.85
    CRITICAL_PASS_THRESHOLD: Final[float] = 0.95

    # Score weights
    DEFAULT_WEIGHT: Final[float] = 1.0
    HIGH_WEIGHT: Final[float] = 1.5
    CRITICAL_WEIGHT: Final[float] = 2.0


GATE_THRESHOLDS = GateThresholds()


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class CacheLimits:
    """Cache configuration constants."""

    # TTL (seconds)
    TTL_SHORT: Final[int] = 60
    TTL_MEDIUM: Final[int] = 300
    TTL_LONG: Final[int] = 3600
    TTL_VERY_LONG: Final[int] = 86400

    # Size limits
    MAX_CACHE_SIZE: Final[int] = 10000
    MAX_MEMORY_MB: Final[int] = 512


CACHE_LIMITS = CacheLimits()


# =============================================================================
# RESOURCE LIMITS
# =============================================================================

@dataclass(frozen=True)
class ResourceLimits:
    """Resource usage limits."""

    # Token limits
    MAX_PROMPT_TOKENS: Final[int] = 4096
    MAX_CONTEXT_TOKENS: Final[int] = 8192
    MAX_OUTPUT_TOKENS: Final[int] = 4096

    # Batch sizes
    DEFAULT_BATCH_SIZE: Final[int] = 10
    LARGE_BATCH_SIZE: Final[int] = 100
    MAX_BATCH_SIZE: Final[int] = 1000

    # Concurrency
    DEFAULT_WORKERS: Final[int] = 4
    MAX_WORKERS: Final[int] = 16


RESOURCE_LIMITS = ResourceLimits()


# =============================================================================
# SECURITY CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class SecurityLimits:
    """Security-related constants."""

    # Secret detection
    MIN_SECRET_LENGTH: Final[int] = 8
    MAX_SECRET_SCAN_SIZE: Final[int] = 1_000_000

    # Rate limiting
    DEFAULT_RATE_LIMIT: Final[int] = 100
    STRICT_RATE_LIMIT: Final[int] = 10

    # Session
    SESSION_TIMEOUT_SECONDS: Final[int] = 3600
    MAX_SESSIONS_PER_USER: Final[int] = 5


SECURITY_LIMITS = SecurityLimits()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TIMEOUTS",
    "RETRY_LIMITS",
    "CODE_QUALITY_LIMITS",
    "COVERAGE_THRESHOLDS",
    "GATE_THRESHOLDS",
    "CACHE_LIMITS",
    "RESOURCE_LIMITS",
    "SECURITY_LIMITS",
    # Individual classes for type hints
    "Timeouts",
    "RetryLimits",
    "CodeQualityLimits",
    "CoverageThresholds",
    "GateThresholds",
    "CacheLimits",
    "ResourceLimits",
    "SecurityLimits",
]
