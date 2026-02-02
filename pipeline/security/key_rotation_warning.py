"""Key Rotation Warning Module - GAP-SEC-002 FIX.

This module provides warnings for keys that need rotation.

GAP-SEC-002 FIX: Document key rotation requirements and add warnings.

Usage:
    from pipeline.security.key_rotation_warning import (
        check_key_rotation_warnings,
        log_key_rotation_warnings,
        record_key_rotation,
    )

    # At startup
    log_key_rotation_warnings()

    # After rotating a key
    record_key_rotation("SECRET_KEY")

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-22)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Key metadata file (tracks when keys were last rotated)
KEY_METADATA_FILE = Path.home() / ".humangr" / "key_rotation_metadata.json"

# Key rotation periods in days
KEY_ROTATION_PERIODS = {
    # Critical keys - 30 days
    "SECRET_KEY": 30,
    "DATABASE_ENCRYPTION_KEY": 30,
    "MODEL_SIGNING_KEY": 30,

    # Standard keys - 90 days
    "ANTHROPIC_API_KEY": 90,
    "OPENAI_API_KEY": 90,
    "LANGFUSE_SECRET_KEY": 90,
    "LANGFUSE_PUBLIC_KEY": 90,
    "GOOGLE_API_KEY": 90,
    "LANGCHAIN_API_KEY": 90,
    "TAVILY_API_KEY": 90,
    "NEWSDATA_API_KEY": 90,

    # Infrastructure keys - 180 days
    "AWS_ACCESS_KEY_ID": 180,
    "AWS_SECRET_ACCESS_KEY": 180,
    "GITHUB_TOKEN": 180,
    "CLERK_SECRET_KEY": 180,
    "STRIPE_SECRET_KEY": 180,
    "GRAFANA_API_KEY": 180,
    "ALCHEMY_API_KEY": 180,
}

# Warning thresholds based on rotation period
WARNING_THRESHOLDS = {
    30: 7,    # 7 days warning for 30-day keys
    90: 14,   # 14 days warning for 90-day keys
    180: 30,  # 30 days warning for 180-day keys
}


def _get_warning_threshold(rotation_days: int) -> int:
    """Get warning threshold for a given rotation period."""
    return WARNING_THRESHOLDS.get(rotation_days, 7)


def _load_metadata() -> dict:
    """Load key rotation metadata from file."""
    if not KEY_METADATA_FILE.exists():
        return {}

    try:
        return json.loads(KEY_METADATA_FILE.read_text())
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"GAP-SEC-002: Failed to load key metadata: {e}")
        return {}


def _save_metadata(metadata: dict) -> None:
    """Save key rotation metadata to file."""
    KEY_METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_METADATA_FILE.write_text(json.dumps(metadata, indent=2))


def check_key_rotation_warnings() -> list[str]:
    """Check for keys that need rotation soon.

    GAP-SEC-002: Returns list of warning messages for keys nearing rotation.

    Returns:
        List of warning messages. Empty if all keys are up to date.
    """
    warnings = []
    metadata = _load_metadata()

    if not metadata:
        logger.warning(
            "GAP-SEC-002: Key rotation metadata not found. "
            f"Create {KEY_METADATA_FILE} to track key ages. "
            "See docs/security/KEY_ROTATION_POLICY.md"
        )
        return ["Key rotation tracking not configured. See KEY_ROTATION_POLICY.md"]

    today = datetime.now()

    for key_name, rotation_days in KEY_ROTATION_PERIODS.items():
        # Skip keys that aren't in the environment
        if not os.environ.get(key_name):
            continue

        if key_name not in metadata:
            warnings.append(
                f"GAP-SEC-002: {key_name} rotation date not tracked. "
                "Please record when this key was last rotated."
            )
            continue

        try:
            last_rotated = datetime.fromisoformat(metadata[key_name])
        except (ValueError, TypeError):
            warnings.append(f"GAP-SEC-002: Invalid date for {key_name} in metadata")
            continue

        days_since_rotation = (today - last_rotated).days
        days_until_due = rotation_days - days_since_rotation
        warning_threshold = _get_warning_threshold(rotation_days)

        if days_until_due < 0:
            warnings.append(
                f"GAP-SEC-002 CRITICAL: {key_name} is {abs(days_until_due)} days "
                f"OVERDUE for rotation! (Rotation period: {rotation_days} days)"
            )
        elif days_until_due <= warning_threshold:
            warnings.append(
                f"GAP-SEC-002 WARNING: {key_name} needs rotation in {days_until_due} days "
                f"(Rotation period: {rotation_days} days, last rotated: {last_rotated.date()})"
            )

    return warnings


def log_key_rotation_warnings() -> int:
    """Log key rotation warnings at startup.

    Call this function early in application startup.

    Returns:
        Number of warnings logged.
    """
    warnings = check_key_rotation_warnings()

    for warning in warnings:
        if "CRITICAL" in warning:
            logger.critical(warning)
        elif "WARNING" in warning:
            logger.warning(warning)
        else:
            logger.info(warning)

    if warnings:
        logger.info(
            f"GAP-SEC-002: {len(warnings)} key rotation warnings. "
            "See docs/security/KEY_ROTATION_POLICY.md for rotation procedures."
        )

    return len(warnings)


def record_key_rotation(key_name: str, rotated_at: Optional[datetime] = None) -> None:
    """Record that a key was rotated.

    Call this after successfully rotating a key.

    Args:
        key_name: Name of the key that was rotated.
        rotated_at: When the key was rotated. Defaults to now.
    """
    metadata = _load_metadata()

    if rotated_at is None:
        rotated_at = datetime.now()

    metadata[key_name] = rotated_at.isoformat()
    _save_metadata(metadata)

    rotation_period = KEY_ROTATION_PERIODS.get(key_name, "unknown")
    logger.info(
        f"GAP-SEC-002: Recorded rotation of {key_name}. "
        f"Next rotation due in {rotation_period} days."
    )


def get_key_status(key_name: str) -> dict:
    """Get the rotation status of a specific key.

    Args:
        key_name: Name of the key to check.

    Returns:
        Dict with key status information.
    """
    metadata = _load_metadata()
    rotation_period = KEY_ROTATION_PERIODS.get(key_name)

    if rotation_period is None:
        return {
            "key_name": key_name,
            "tracked": False,
            "error": "Key not in rotation policy",
        }

    if key_name not in metadata:
        return {
            "key_name": key_name,
            "tracked": True,
            "last_rotated": None,
            "rotation_period_days": rotation_period,
            "days_until_due": None,
            "status": "not_recorded",
        }

    try:
        last_rotated = datetime.fromisoformat(metadata[key_name])
        days_since = (datetime.now() - last_rotated).days
        days_until_due = rotation_period - days_since

        if days_until_due < 0:
            status = "overdue"
        elif days_until_due <= _get_warning_threshold(rotation_period):
            status = "warning"
        else:
            status = "ok"

        return {
            "key_name": key_name,
            "tracked": True,
            "last_rotated": last_rotated.isoformat(),
            "rotation_period_days": rotation_period,
            "days_since_rotation": days_since,
            "days_until_due": days_until_due,
            "status": status,
        }
    except Exception as e:
        return {
            "key_name": key_name,
            "tracked": True,
            "error": str(e),
        }


def get_all_key_statuses() -> list[dict]:
    """Get rotation status for all tracked keys.

    Returns:
        List of status dicts for all keys in rotation policy.
    """
    return [get_key_status(key_name) for key_name in KEY_ROTATION_PERIODS.keys()]


def initialize_key_tracking() -> None:
    """Initialize key tracking metadata with current date for all keys.

    Use this when first setting up key rotation tracking.
    All keys will be marked as "just rotated" today.
    """
    metadata = _load_metadata()
    today = datetime.now().isoformat()

    for key_name in KEY_ROTATION_PERIODS.keys():
        if key_name not in metadata and os.environ.get(key_name):
            metadata[key_name] = today
            logger.info(f"GAP-SEC-002: Initialized tracking for {key_name}")

    _save_metadata(metadata)
    logger.info(
        f"GAP-SEC-002: Key tracking initialized. "
        f"Tracking {len(metadata)} keys in {KEY_METADATA_FILE}"
    )


# Module-level availability flag
KEY_ROTATION_WARNING_AVAILABLE = True

__all__ = [
    "check_key_rotation_warnings",
    "log_key_rotation_warnings",
    "record_key_rotation",
    "get_key_status",
    "get_all_key_statuses",
    "initialize_key_tracking",
    "KEY_ROTATION_PERIODS",
    "KEY_METADATA_FILE",
    "KEY_ROTATION_WARNING_AVAILABLE",
]
