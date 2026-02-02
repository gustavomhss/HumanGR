"""Pydantic Custom Validators for Pipeline V2.

This module provides custom Pydantic validators for pipeline state,
serialization/deserialization utilities, and type-safe state management.

Key Features:
- Custom validators for PipelineState fields
- State serialization/deserialization utilities
- Type coercion for flexible input handling
- Cross-field validation for invariant enforcement
- Graceful degradation with validation error handling

Architecture:
    PipelineState (TypedDict)
        |
        v
    StateValidator (Pydantic model wrapper)
        |
        ├─> Field validators (individual fields)
        ├─> Model validators (cross-field)
        └─> Serializers (JSON, YAML, etc.)

Usage:
    from pipeline.langgraph.pydantic_validators import (
        validate_pipeline_state,
        StateSerializer,
        SprintIdValidator,
    )

    # Validate state
    result = validate_pipeline_state(state_dict)
    if result.is_valid:
        validated_state = result.validated_data
    else:
        print(result.errors)

    # Serialize/deserialize
    serializer = StateSerializer()
    json_str = serializer.to_json(state)
    restored_state = serializer.from_json(json_str)

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import os
import logging
import json
import re
from typing import Any, Dict, List, Optional, TypedDict
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

STRICT_VALIDATION = os.getenv("PYDANTIC_STRICT_VALIDATION", "false").lower() == "true"
COERCE_TYPES = os.getenv("PYDANTIC_COERCE_TYPES", "true").lower() == "true"
MAX_ERROR_DETAILS = int(os.getenv("PYDANTIC_MAX_ERRORS", "10"))

# Check Pydantic availability
PYDANTIC_V2_AVAILABLE = False
try:
    from pydantic import (
        BaseModel,
        Field,
        field_validator,
        model_validator,
        ConfigDict,
        ValidationError,
    )
    PYDANTIC_V2_AVAILABLE = True
except ImportError:
    logger.debug("Pydantic V2 not available")

PYDANTIC_VALIDATORS_AVAILABLE = PYDANTIC_V2_AVAILABLE


# =============================================================================
# VALIDATION RESULT TYPES
# =============================================================================


class ValidationErrorDetail(TypedDict):
    """Details of a validation error."""
    field: str
    message: str
    input_value: Any
    error_type: str


class ValidationResult(TypedDict):
    """Result of state validation."""
    is_valid: bool
    validated_data: Optional[Dict[str, Any]]
    errors: List[ValidationErrorDetail]
    warnings: List[str]
    validation_time_ms: float


class SerializationResult(TypedDict):
    """Result of serialization."""
    success: bool
    data: Optional[str]
    format: str
    size_bytes: int
    error: Optional[str]


class DeserializationResult(TypedDict):
    """Result of deserialization."""
    success: bool
    data: Optional[Dict[str, Any]]
    format: str
    validation_applied: bool
    error: Optional[str]


# =============================================================================
# CUSTOM VALIDATORS (functions that work without Pydantic)
# =============================================================================


def validate_sprint_id(value: Any) -> str:
    """Validate sprint ID format.

    Args:
        value: Value to validate

    Returns:
        Validated sprint ID

    Raises:
        ValueError: If invalid format
    """
    if not isinstance(value, str):
        value = str(value)

    # Sprint ID format: S00-S99 or S00H (half sprint)
    pattern = r"^S\d{2}H?$"
    if not re.match(pattern, value):
        raise ValueError(
            f"Invalid sprint ID format: {value}. Expected S00-S99 or S00H."
        )

    return value


def validate_gate_id(value: Any) -> str:
    """Validate gate ID format.

    Args:
        value: Value to validate

    Returns:
        Validated gate ID

    Raises:
        ValueError: If invalid format
    """
    if not isinstance(value, str):
        value = str(value)

    # Gate ID format: G0-G8 or G0.x
    pattern = r"^G\d(\.\d+)?$"
    if not re.match(pattern, value):
        raise ValueError(
            f"Invalid gate ID format: {value}. Expected G0-G8 or G0.x."
        )

    return value


def validate_phase(value: Any) -> str:
    """Validate phase value.

    Args:
        value: Value to validate

    Returns:
        Validated phase

    Raises:
        ValueError: If invalid phase
    """
    valid_phases = [
        "INIT", "LOADING", "EXEC", "GATE",
        "SIGNOFF", "ARTIFACT", "COMPLETE", "FAILED"
    ]

    if isinstance(value, str):
        value_upper = value.upper()
        if value_upper in valid_phases:
            return value_upper

    raise ValueError(
        f"Invalid phase: {value}. Expected one of: {valid_phases}"
    )


def validate_confidence_score(value: Any) -> float:
    """Validate confidence score (0.0-1.0).

    Args:
        value: Value to validate

    Returns:
        Validated confidence score

    Raises:
        ValueError: If out of range
    """
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"Cannot convert to float: {value}")

    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric value, got: {type(value)}")

    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"Confidence score must be between 0.0 and 1.0, got: {value}"
        )

    return value


def validate_timestamp(value: Any) -> str:
    """Validate and normalize timestamp.

    Args:
        value: Value to validate

    Returns:
        ISO format timestamp string

    Raises:
        ValueError: If invalid timestamp
    """
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, str):
        try:
            # Try parsing ISO format
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return value
        except ValueError as e:
            logger.debug(f"GRAPH: Graph operation failed: {e}")

        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M:%S",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(value, fmt)
                return dt.isoformat()
            except ValueError:
                continue

    raise ValueError(f"Invalid timestamp format: {value}")


def validate_url(value: Any) -> str:
    """Validate URL format.

    Args:
        value: Value to validate

    Returns:
        Validated URL

    Raises:
        ValueError: If invalid URL
    """
    if not isinstance(value, str):
        value = str(value)

    pattern = r"^https?://[^\s]+$"
    if not re.match(pattern, value):
        raise ValueError(f"Invalid URL format: {value}")

    return value


def validate_claim_verdict(value: Any) -> str:
    """Validate claim verdict.

    Args:
        value: Value to validate

    Returns:
        Validated verdict

    Raises:
        ValueError: If invalid verdict
    """
    valid_verdicts = [
        "TRUE", "FALSE", "PARTIALLY_TRUE",
        "UNVERIFIABLE", "PENDING"
    ]

    if isinstance(value, str):
        value_upper = value.upper().replace(" ", "_")
        if value_upper in valid_verdicts:
            return value_upper

    raise ValueError(
        f"Invalid verdict: {value}. Expected one of: {valid_verdicts}"
    )


def validate_error_severity(value: Any) -> str:
    """Validate error severity level.

    Args:
        value: Value to validate

    Returns:
        Validated severity

    Raises:
        ValueError: If invalid severity
    """
    valid_severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

    if isinstance(value, str):
        value_upper = value.upper()
        if value_upper in valid_severities:
            return value_upper

    raise ValueError(
        f"Invalid severity: {value}. Expected one of: {valid_severities}"
    )


# =============================================================================
# PYDANTIC MODELS (if available)
# =============================================================================

if PYDANTIC_V2_AVAILABLE:

    class SprintIdType(str):
        """Custom type for Sprint ID."""

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type, handler):
            from pydantic_core import core_schema
            return core_schema.str_schema(
                pattern=r"^S\d{2}H?$",
            )

    class GateIdType(str):
        """Custom type for Gate ID."""

        @classmethod
        def __get_pydantic_core_schema__(cls, source_type, handler):
            from pydantic_core import core_schema
            return core_schema.str_schema(
                pattern=r"^G\d(\.\d+)?$",
            )

    class IdentityStateModel(BaseModel):
        """Pydantic model for identity state."""

        model_config = ConfigDict(
            str_strip_whitespace=True,
            validate_assignment=True,
        )

        sprint_id: str
        run_id: str
        pack_id: Optional[str] = None
        cerebro_id: Optional[str] = None

        @field_validator("sprint_id")
        @classmethod
        def validate_sprint(cls, v: str) -> str:
            return validate_sprint_id(v)

    class GateStateModel(BaseModel):
        """Pydantic model for gate state."""

        model_config = ConfigDict(
            str_strip_whitespace=True,
        )

        gate_id: str
        status: str
        passed: bool = False
        score: float = 0.0
        errors: List[str] = Field(default_factory=list)

        @field_validator("gate_id")
        @classmethod
        def validate_gate(cls, v: str) -> str:
            return validate_gate_id(v)

        @field_validator("score")
        @classmethod
        def validate_score(cls, v: float) -> float:
            return validate_confidence_score(v)

    class PipelineStateModel(BaseModel):
        """Pydantic model for pipeline state validation."""

        model_config = ConfigDict(
            str_strip_whitespace=True,
            validate_assignment=True,
            extra="allow",  # Allow extra fields for extensibility
        )

        # Required fields
        sprint_id: str
        run_id: str
        current_phase: str
        status: str

        # Optional fields with defaults
        gates_passed: List[str] = Field(default_factory=list)
        gates_failed: List[str] = Field(default_factory=list)
        current_gate: Optional[str] = None
        confidence_score: float = 0.0
        error_count: int = 0

        # Timestamps
        created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
        updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

        @field_validator("sprint_id")
        @classmethod
        def validate_sprint_field(cls, v: str) -> str:
            return validate_sprint_id(v)

        @field_validator("current_phase")
        @classmethod
        def validate_phase_field(cls, v: str) -> str:
            return validate_phase(v)

        @field_validator("confidence_score")
        @classmethod
        def validate_confidence_field(cls, v: float) -> float:
            return validate_confidence_score(v)

        @field_validator("current_gate")
        @classmethod
        def validate_gate_field(cls, v: Optional[str]) -> Optional[str]:
            if v is None:
                return None
            return validate_gate_id(v)

        @field_validator("created_at", "updated_at")
        @classmethod
        def validate_timestamps(cls, v: str) -> str:
            return validate_timestamp(v)

        @model_validator(mode="after")
        def validate_state_consistency(self):
            """Cross-field validation for state consistency."""
            # If phase is COMPLETE, must have passed gates
            if self.current_phase == "COMPLETE" and not self.gates_passed:
                raise ValueError(
                    "COMPLETE phase requires at least one passed gate"
                )

            # If phase is FAILED, must have error_count > 0 or failed gates
            if self.current_phase == "FAILED":
                if self.error_count == 0 and not self.gates_failed:
                    raise ValueError(
                        "FAILED phase requires error_count > 0 or failed gates"
                    )

            return self


# =============================================================================
# STATE VALIDATOR
# =============================================================================


class StateValidator:
    """Validates pipeline state with custom rules.

    Uses Pydantic models if available, falls back to
    manual validation otherwise.
    """

    def __init__(
        self,
        strict: bool = STRICT_VALIDATION,
        coerce_types: bool = COERCE_TYPES,
    ):
        """Initialize state validator.

        Args:
            strict: Whether to use strict validation
            coerce_types: Whether to coerce types
        """
        self.strict = strict
        self.coerce_types = coerce_types
        self._validators = {
            "sprint_id": validate_sprint_id,
            "gate_id": validate_gate_id,
            "current_phase": validate_phase,
            "confidence_score": validate_confidence_score,
            "timestamp": validate_timestamp,
            "verdict": validate_claim_verdict,
            "severity": validate_error_severity,
        }

    def validate(
        self,
        state: Dict[str, Any],
        required_fields: Optional[List[str]] = None,
    ) -> ValidationResult:
        """Validate pipeline state.

        Args:
            state: State dictionary to validate
            required_fields: Fields that must be present

        Returns:
            ValidationResult with validation details
        """
        import time
        start_time = time.time()

        errors: List[ValidationErrorDetail] = []
        warnings: List[str] = []
        validated_data = state.copy()

        # Check required fields
        required = required_fields or ["sprint_id", "run_id"]
        for field in required:
            if field not in state:
                errors.append(ValidationErrorDetail(
                    field=field,
                    message=f"Required field '{field}' is missing",
                    input_value=None,
                    error_type="missing_field",
                ))

        if errors and self.strict:
            elapsed_ms = (time.time() - start_time) * 1000
            return ValidationResult(
                is_valid=False,
                validated_data=None,
                errors=errors[:MAX_ERROR_DETAILS],
                warnings=warnings,
                validation_time_ms=elapsed_ms,
            )

        # Use Pydantic if available
        if PYDANTIC_V2_AVAILABLE and not errors:
            try:
                model = PipelineStateModel(**state)
                validated_data = model.model_dump()
            except ValidationError as e:
                for error in e.errors():
                    errors.append(ValidationErrorDetail(
                        field=".".join(str(loc) for loc in error["loc"]),
                        message=error["msg"],
                        input_value=error.get("input"),
                        error_type=error["type"],
                    ))
        else:
            # Manual validation
            validated_data = self._validate_manually(state, errors, warnings)

        elapsed_ms = (time.time() - start_time) * 1000

        return ValidationResult(
            is_valid=len(errors) == 0,
            validated_data=validated_data if len(errors) == 0 else None,
            errors=errors[:MAX_ERROR_DETAILS],
            warnings=warnings,
            validation_time_ms=elapsed_ms,
        )

    def _validate_manually(
        self,
        state: Dict[str, Any],
        errors: List[ValidationErrorDetail],
        warnings: List[str],
    ) -> Dict[str, Any]:
        """Manually validate state fields.

        Args:
            state: State to validate
            errors: List to append errors to
            warnings: List to append warnings to

        Returns:
            Validated state dictionary
        """
        validated = state.copy()

        # Validate specific fields
        field_validators = {
            "sprint_id": validate_sprint_id,
            "current_phase": validate_phase,
            "current_gate": lambda v: validate_gate_id(v) if v else None,
            "confidence_score": validate_confidence_score,
        }

        for field, validator in field_validators.items():
            if field in validated:
                try:
                    validated[field] = validator(validated[field])
                except ValueError as e:
                    if self.coerce_types:
                        warnings.append(f"Field {field}: {str(e)}")
                    else:
                        errors.append(ValidationErrorDetail(
                            field=field,
                            message=str(e),
                            input_value=validated[field],
                            error_type="validation_error",
                        ))

        return validated

    def register_validator(
        self,
        field_name: str,
        validator_func: callable,
    ) -> None:
        """Register a custom validator for a field.

        Args:
            field_name: Name of the field
            validator_func: Validation function
        """
        self._validators[field_name] = validator_func


# =============================================================================
# STATE SERIALIZER
# =============================================================================


class StateSerializer:
    """Serializes and deserializes pipeline state.

    Supports JSON and YAML formats with optional validation.
    """

    def __init__(
        self,
        validate_on_deserialize: bool = True,
        pretty_print: bool = False,
    ):
        """Initialize state serializer.

        Args:
            validate_on_deserialize: Validate when deserializing
            pretty_print: Format output for readability
        """
        self.validate_on_deserialize = validate_on_deserialize
        self.pretty_print = pretty_print
        self._validator = StateValidator()

    def to_json(
        self,
        state: Dict[str, Any],
        validate: bool = False,
    ) -> SerializationResult:
        """Serialize state to JSON.

        Args:
            state: State to serialize
            validate: Whether to validate before serializing

        Returns:
            SerializationResult with JSON string
        """
        try:
            if validate:
                result = self._validator.validate(state)
                if not result["is_valid"]:
                    return SerializationResult(
                        success=False,
                        data=None,
                        format="json",
                        size_bytes=0,
                        error=f"Validation failed: {result['errors']}",
                    )
                state = result["validated_data"]

            if self.pretty_print:
                json_str = json.dumps(state, indent=2, default=str)
            else:
                json_str = json.dumps(state, default=str)

            return SerializationResult(
                success=True,
                data=json_str,
                format="json",
                size_bytes=len(json_str.encode()),
                error=None,
            )

        except Exception as e:
            return SerializationResult(
                success=False,
                data=None,
                format="json",
                size_bytes=0,
                error=str(e),
            )

    def from_json(
        self,
        json_str: str,
    ) -> DeserializationResult:
        """Deserialize state from JSON.

        Args:
            json_str: JSON string to deserialize

        Returns:
            DeserializationResult with state dictionary
        """
        try:
            data = json.loads(json_str)

            if self.validate_on_deserialize:
                result = self._validator.validate(data)
                if not result["is_valid"]:
                    return DeserializationResult(
                        success=False,
                        data=None,
                        format="json",
                        validation_applied=True,
                        error=f"Validation failed: {result['errors']}",
                    )
                data = result["validated_data"]

            return DeserializationResult(
                success=True,
                data=data,
                format="json",
                validation_applied=self.validate_on_deserialize,
                error=None,
            )

        except json.JSONDecodeError as e:
            return DeserializationResult(
                success=False,
                data=None,
                format="json",
                validation_applied=False,
                error=f"JSON decode error: {str(e)}",
            )

        except Exception as e:
            return DeserializationResult(
                success=False,
                data=None,
                format="json",
                validation_applied=False,
                error=str(e),
            )

    def to_dict(
        self,
        state: Any,
        validate: bool = True,
    ) -> Dict[str, Any]:
        """Convert state to validated dictionary.

        Args:
            state: State object (dict, model, etc.)
            validate: Whether to validate

        Returns:
            Validated dictionary
        """
        if PYDANTIC_V2_AVAILABLE and hasattr(state, "model_dump"):
            data = state.model_dump()
        elif hasattr(state, "__dict__"):
            data = dict(state.__dict__)
        elif isinstance(state, dict):
            data = dict(state)
        else:
            data = {"value": state}

        if validate:
            result = self._validator.validate(data)
            if result["is_valid"]:
                return result["validated_data"]

        return data


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def validate_pipeline_state(
    state: Dict[str, Any],
    strict: bool = False,
) -> ValidationResult:
    """Validate pipeline state.

    Convenience function for quick validation.

    Args:
        state: State to validate
        strict: Use strict validation

    Returns:
        ValidationResult
    """
    validator = StateValidator(strict=strict)
    return validator.validate(state)


def serialize_state(
    state: Dict[str, Any],
    format: str = "json",
) -> SerializationResult:
    """Serialize state to string.

    Convenience function for serialization.

    Args:
        state: State to serialize
        format: Output format

    Returns:
        SerializationResult
    """
    serializer = StateSerializer()
    if format == "json":
        return serializer.to_json(state)
    else:
        return SerializationResult(
            success=False,
            data=None,
            format=format,
            size_bytes=0,
            error=f"Unsupported format: {format}",
        )


def deserialize_state(
    data: str,
    format: str = "json",
) -> DeserializationResult:
    """Deserialize state from string.

    Convenience function for deserialization.

    Args:
        data: String to deserialize
        format: Input format

    Returns:
        DeserializationResult
    """
    serializer = StateSerializer()
    if format == "json":
        return serializer.from_json(data)
    else:
        return DeserializationResult(
            success=False,
            data=None,
            format=format,
            validation_applied=False,
            error=f"Unsupported format: {format}",
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "StateValidator",
    "StateSerializer",
    # Result types
    "ValidationResult",
    "ValidationErrorDetail",
    "SerializationResult",
    "DeserializationResult",
    # Validator functions
    "validate_sprint_id",
    "validate_gate_id",
    "validate_phase",
    "validate_confidence_score",
    "validate_timestamp",
    "validate_url",
    "validate_claim_verdict",
    "validate_error_severity",
    # Convenience functions
    "validate_pipeline_state",
    "serialize_state",
    "deserialize_state",
    # Constants
    "PYDANTIC_VALIDATORS_AVAILABLE",
    "PYDANTIC_V2_AVAILABLE",
]

# Pydantic models (if available)
if PYDANTIC_V2_AVAILABLE:
    __all__.extend([
        "PipelineStateModel",
        "IdentityStateModel",
        "GateStateModel",
    ])
