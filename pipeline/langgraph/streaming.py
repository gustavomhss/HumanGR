"""LangGraph Streaming and Human-in-the-Loop Module for Pipeline V2.

This module provides enhanced LangGraph integration with streaming support,
advanced checkpointing, and human-in-the-loop patterns.

Key Features:
- Streaming support for real-time output
- Enhanced checkpointing with compression and versioning
- Human-in-the-loop patterns for approval workflows
- Breakpoint management for debugging
- Graceful degradation when LangGraph unavailable

Architecture:
    StreamingWorkflow
        |
        ├─> StreamHandler (real-time output)
        │       ↓
        │   StreamEvent (node_output, token, etc.)
        │
        ├─> EnhancedCheckpointer (versioned snapshots)
        │       ↓
        │   CheckpointVersion (compressed state)
        │
        └─> HumanLoop (approval workflows)
                ↓
            ApprovalRequest -> HumanDecision -> Resume

Usage:
    from pipeline.langgraph.streaming import (
        get_streaming_workflow,
        stream_workflow_execution,
        create_approval_request,
        resume_after_approval,
    )

    # Stream workflow execution
    async for event in stream_workflow_execution(state):
        if event.event_type == "node_output":
            print(f"Node {event.node_id}: {event.data}")
        elif event.event_type == "approval_needed":
            await handle_approval(event)

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import os
import logging
import asyncio
import gzip
import json
from typing import (
    Any, Dict, List, Optional, TypedDict, AsyncIterator,
    Callable, Awaitable
)
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

CHECKPOINT_COMPRESSION_ENABLED = os.getenv("LANGGRAPH_CHECKPOINT_COMPRESS", "true").lower() == "true"
CHECKPOINT_MAX_VERSIONS = int(os.getenv("LANGGRAPH_CHECKPOINT_MAX_VERSIONS", "10"))
STREAM_BUFFER_SIZE = int(os.getenv("LANGGRAPH_STREAM_BUFFER", "100"))
APPROVAL_TIMEOUT_SECONDS = float(os.getenv("LANGGRAPH_APPROVAL_TIMEOUT", "3600"))
BREAKPOINT_ENABLED = os.getenv("LANGGRAPH_BREAKPOINTS_ENABLED", "true").lower() == "true"

# Check LangGraph availability
LANGGRAPH_AVAILABLE = False
try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.base import BaseCheckpointSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    logger.debug("LangGraph not available")

STREAMING_AVAILABLE = True  # Core functionality always available


# =============================================================================
# ENUMS
# =============================================================================


class StreamEventType(str, Enum):
    """Types of stream events."""
    NODE_START = "node_start"
    NODE_OUTPUT = "node_output"
    NODE_END = "node_end"
    TOKEN = "token"
    CHECKPOINT = "checkpoint"
    APPROVAL_NEEDED = "approval_needed"
    ERROR = "error"
    COMPLETE = "complete"


class ApprovalStatus(str, Enum):
    """Status of an approval request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class BreakpointType(str, Enum):
    """Types of breakpoints."""
    BEFORE_NODE = "before_node"
    AFTER_NODE = "after_node"
    ON_CONDITION = "on_condition"
    ON_ERROR = "on_error"


class CheckpointStrategy(str, Enum):
    """Checkpoint strategy."""
    EVERY_NODE = "every_node"
    ON_BRANCH = "on_branch"
    ON_APPROVAL = "on_approval"
    MANUAL = "manual"


# =============================================================================
# EVENT TYPES
# =============================================================================


class StreamEvent(TypedDict):
    """A streaming event."""
    event_id: str
    event_type: str
    node_id: Optional[str]
    data: Any
    timestamp: str
    metadata: Dict[str, Any]


class CheckpointVersion(TypedDict):
    """A versioned checkpoint."""
    version_id: str
    checkpoint_id: str
    state: Dict[str, Any]
    compressed: bool
    size_bytes: int
    created_at: str
    node_id: Optional[str]
    parent_version: Optional[str]


class ApprovalRequest(TypedDict):
    """A request for human approval."""
    request_id: str
    workflow_id: str
    node_id: str
    checkpoint_id: str
    reason: str
    context: Dict[str, Any]
    options: List[str]
    timeout_at: str
    status: str
    created_at: str


class HumanDecision(TypedDict):
    """A human decision on an approval request."""
    request_id: str
    decision: str
    reason: Optional[str]
    made_by: str
    made_at: str
    metadata: Dict[str, Any]


class BreakpointConfig(TypedDict):
    """Configuration for a breakpoint."""
    breakpoint_id: str
    breakpoint_type: str
    node_id: Optional[str]
    condition: Optional[str]
    enabled: bool
    hit_count: int


class StreamingResult(TypedDict):
    """Result of streaming execution."""
    workflow_id: str
    final_state: Dict[str, Any]
    events: List[StreamEvent]
    checkpoints_created: int
    approvals_requested: int
    total_time_ms: float
    success: bool
    error: Optional[str]


# =============================================================================
# STREAM HANDLER
# =============================================================================


class StreamHandler:
    """Handles streaming output from workflow execution.

    Buffers events and provides async iteration for real-time consumption.
    """

    def __init__(
        self,
        buffer_size: int = STREAM_BUFFER_SIZE,
        include_tokens: bool = True,
    ):
        """Initialize stream handler.

        Args:
            buffer_size: Maximum events to buffer
            include_tokens: Whether to stream individual tokens
        """
        self.buffer_size = buffer_size
        self.include_tokens = include_tokens
        self._buffer: asyncio.Queue = asyncio.Queue(maxsize=buffer_size)
        self._completed = False
        self._error: Optional[str] = None

    async def emit(
        self,
        event_type: StreamEventType,
        data: Any,
        node_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a stream event.

        Args:
            event_type: Type of event
            data: Event data
            node_id: Optional node identifier
            metadata: Optional metadata
        """
        import uuid

        event = StreamEvent(
            event_id=str(uuid.uuid4())[:8],
            event_type=event_type.value,
            node_id=node_id,
            data=data,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

        try:
            await asyncio.wait_for(
                self._buffer.put(event),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            # Buffer full, drop oldest
            try:
                self._buffer.get_nowait()
                await self._buffer.put(event)
            except asyncio.QueueEmpty:
                logger.debug(f"GRAPH: Graph operation failed: {e}")

    async def complete(
        self,
        final_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark stream as complete.

        Args:
            final_state: Optional final state to include
        """
        await self.emit(
            StreamEventType.COMPLETE,
            data={"final_state": final_state},
        )
        self._completed = True

    async def error(self, error_message: str) -> None:
        """Emit error and mark stream as failed.

        Args:
            error_message: Error message
        """
        await self.emit(
            StreamEventType.ERROR,
            data={"error": error_message},
        )
        self._error = error_message
        self._completed = True

    async def __aiter__(self) -> AsyncIterator[StreamEvent]:
        """Async iterator for stream events."""
        while not self._completed or not self._buffer.empty():
            try:
                event = await asyncio.wait_for(
                    self._buffer.get(),
                    timeout=0.1,
                )
                yield event
            except asyncio.TimeoutError:
                if self._completed:
                    break
                continue


# =============================================================================
# ENHANCED CHECKPOINTER
# =============================================================================


class EnhancedCheckpointer:
    """Enhanced checkpointer with compression and versioning.

    Provides versioned checkpoints with optional compression
    and automatic cleanup of old versions.
    """

    def __init__(
        self,
        compression_enabled: bool = CHECKPOINT_COMPRESSION_ENABLED,
        max_versions: int = CHECKPOINT_MAX_VERSIONS,
        strategy: CheckpointStrategy = CheckpointStrategy.EVERY_NODE,
    ):
        """Initialize enhanced checkpointer.

        Args:
            compression_enabled: Whether to compress checkpoints
            max_versions: Maximum versions to keep
            strategy: Checkpointing strategy
        """
        self.compression_enabled = compression_enabled
        self.max_versions = max_versions
        self.strategy = strategy

        self._checkpoints: Dict[str, List[CheckpointVersion]] = {}
        self._current_versions: Dict[str, str] = {}

    def _generate_version_id(
        self,
        checkpoint_id: str,
        state: Dict[str, Any],
    ) -> str:
        """Generate unique version ID.

        GAP HUNTER V2 FIX: Use SHA256 instead of MD5.
        """
        content = json.dumps(state, sort_keys=True)
        hash_value = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"{checkpoint_id}-{hash_value}"

    def _compress(self, data: bytes) -> bytes:
        """Compress data using gzip."""
        return gzip.compress(data)

    def _decompress(self, data: bytes) -> bytes:
        """Decompress gzip data."""
        return gzip.decompress(data)

    async def save_checkpoint(
        self,
        checkpoint_id: str,
        state: Dict[str, Any],
        node_id: Optional[str] = None,
    ) -> CheckpointVersion:
        """Save a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier
            state: State to checkpoint
            node_id: Optional node that triggered checkpoint

        Returns:
            CheckpointVersion with checkpoint details
        """
        # Serialize state
        state_json = json.dumps(state).encode()
        original_size = len(state_json)

        # Compress if enabled
        if self.compression_enabled:
            state_data = self._compress(state_json)
            compressed = True
        else:
            state_data = state_json
            compressed = False

        # Get parent version
        parent_version = self._current_versions.get(checkpoint_id)

        # Create version
        version_id = self._generate_version_id(checkpoint_id, state)

        version = CheckpointVersion(
            version_id=version_id,
            checkpoint_id=checkpoint_id,
            state=state,  # Store original for quick access
            compressed=compressed,
            size_bytes=len(state_data),
            created_at=datetime.now(timezone.utc).isoformat(),
            node_id=node_id,
            parent_version=parent_version,
        )

        # Store version
        if checkpoint_id not in self._checkpoints:
            self._checkpoints[checkpoint_id] = []

        self._checkpoints[checkpoint_id].append(version)
        self._current_versions[checkpoint_id] = version_id

        # Cleanup old versions
        self._cleanup_old_versions(checkpoint_id)

        return version

    async def load_checkpoint(
        self,
        checkpoint_id: str,
        version_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier
            version_id: Optional specific version (latest if None)

        Returns:
            State dictionary or None if not found
        """
        if checkpoint_id not in self._checkpoints:
            return None

        versions = self._checkpoints[checkpoint_id]

        if version_id:
            # Find specific version
            for version in versions:
                if version["version_id"] == version_id:
                    return version["state"]
            return None
        else:
            # Return latest
            if versions:
                return versions[-1]["state"]
            return None

    async def list_versions(
        self,
        checkpoint_id: str,
    ) -> List[CheckpointVersion]:
        """List all versions of a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier

        Returns:
            List of checkpoint versions
        """
        return self._checkpoints.get(checkpoint_id, [])

    async def rollback(
        self,
        checkpoint_id: str,
        version_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Rollback to a specific version.

        Args:
            checkpoint_id: Checkpoint identifier
            version_id: Version to rollback to

        Returns:
            State at that version or None if not found
        """
        state = await self.load_checkpoint(checkpoint_id, version_id)

        if state:
            self._current_versions[checkpoint_id] = version_id

        return state

    def _cleanup_old_versions(self, checkpoint_id: str) -> None:
        """Remove old versions beyond max_versions."""
        if checkpoint_id in self._checkpoints:
            versions = self._checkpoints[checkpoint_id]
            if len(versions) > self.max_versions:
                self._checkpoints[checkpoint_id] = versions[-self.max_versions:]


# =============================================================================
# HUMAN-IN-THE-LOOP
# =============================================================================


class HumanLoop:
    """Manages human-in-the-loop approval workflows.

    Handles approval requests, waiting for decisions,
    and resuming workflow execution.
    """

    def __init__(
        self,
        timeout_seconds: float = APPROVAL_TIMEOUT_SECONDS,
        decision_callback: Optional[Callable[[ApprovalRequest], Awaitable[HumanDecision]]] = None,
    ):
        """Initialize human loop manager.

        Args:
            timeout_seconds: Default timeout for approvals
            decision_callback: Optional callback for automated decisions
        """
        self.timeout_seconds = timeout_seconds
        self.decision_callback = decision_callback

        self._pending_requests: Dict[str, ApprovalRequest] = {}
        self._decisions: Dict[str, HumanDecision] = {}
        # BUG FIX 2026-01-24: Use Dict of bools instead of Events to avoid
        # "bound to different event loop" errors. Events are created lazily
        # in wait_for_decision to ensure they're bound to the correct loop.
        self._signaled: Dict[str, bool] = {}
        self._waiting: Dict[str, asyncio.Event] = {}

    async def request_approval(
        self,
        workflow_id: str,
        node_id: str,
        checkpoint_id: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
        options: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> ApprovalRequest:
        """Create an approval request.

        Args:
            workflow_id: Workflow identifier
            node_id: Node requiring approval
            checkpoint_id: Checkpoint to resume from
            reason: Reason for approval request
            context: Additional context
            options: Available options (default: ["approve", "reject"])
            timeout: Custom timeout

        Returns:
            ApprovalRequest with request details
        """
        import uuid

        request_id = str(uuid.uuid4())[:8]
        timeout_seconds = timeout or self.timeout_seconds

        request = ApprovalRequest(
            request_id=request_id,
            workflow_id=workflow_id,
            node_id=node_id,
            checkpoint_id=checkpoint_id,
            reason=reason,
            context=context or {},
            options=options or ["approve", "reject"],
            timeout_at=(datetime.now(timezone.utc) + timedelta(seconds=timeout_seconds)).isoformat(),
            status=ApprovalStatus.PENDING.value,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self._pending_requests[request_id] = request
        # BUG FIX 2026-01-24: Don't create Event here - it binds to current loop
        # Instead, mark as not signaled and create Event lazily in wait_for_decision
        self._signaled[request_id] = False

        return request

    async def submit_decision(
        self,
        request_id: str,
        decision: str,
        reason: Optional[str] = None,
        made_by: str = "human",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Submit a decision for an approval request.

        Args:
            request_id: Request identifier
            decision: Decision ("approve", "reject", etc.)
            reason: Optional reason for decision
            made_by: Who made the decision
            metadata: Additional metadata

        Returns:
            True if decision was accepted
        """
        if request_id not in self._pending_requests:
            return False

        request = self._pending_requests[request_id]

        # Validate decision is in options
        if decision not in request["options"]:
            return False

        # Create decision record
        human_decision = HumanDecision(
            request_id=request_id,
            decision=decision,
            reason=reason,
            made_by=made_by,
            made_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {},
        )

        self._decisions[request_id] = human_decision

        # Update request status
        if decision == "approve":
            request["status"] = ApprovalStatus.APPROVED.value
        elif decision == "reject":
            request["status"] = ApprovalStatus.REJECTED.value
        else:
            request["status"] = decision

        # Signal waiting coroutine
        # BUG FIX 2026-01-24: Set signaled flag first (always works)
        self._signaled[request_id] = True
        # Then try to set Event if it exists (may fail if different loop)
        if request_id in self._waiting:
            try:
                self._waiting[request_id].set()
            except RuntimeError as e:
                # Event bound to different loop - signaled flag will be checked instead
                logger.debug(f"Event.set() failed (different loop), using signal flag: {e}")

        return True

    async def wait_for_decision(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[HumanDecision]:
        """Wait for a decision on an approval request.

        Args:
            request_id: Request identifier
            timeout: Custom timeout (uses request timeout if None)

        Returns:
            HumanDecision or None if timeout/cancelled
        """
        if request_id not in self._pending_requests:
            return None

        request = self._pending_requests[request_id]

        # Calculate remaining timeout
        timeout_at = datetime.fromisoformat(request["timeout_at"])
        remaining = (timeout_at - datetime.now(timezone.utc)).total_seconds()
        wait_timeout = min(remaining, timeout) if timeout else remaining

        if wait_timeout <= 0:
            request["status"] = ApprovalStatus.TIMEOUT.value
            return None

        # Use callback if available
        if self.decision_callback:
            try:
                decision = await asyncio.wait_for(
                    self.decision_callback(request),
                    timeout=wait_timeout,
                )
                self._decisions[request_id] = decision
                return decision
            except asyncio.TimeoutError:
                request["status"] = ApprovalStatus.TIMEOUT.value
                return None

        # Wait for manual decision
        # BUG FIX 2026-01-24: Check signaled flag first (set by submit_decision)
        if self._signaled.get(request_id, False):
            return self._decisions.get(request_id)

        # Create Event lazily in current event loop to avoid binding issues
        if request_id not in self._waiting:
            self._waiting[request_id] = asyncio.Event()

        try:
            # Use short polling intervals to check signaled flag
            poll_interval = 0.5
            elapsed = 0.0
            while elapsed < wait_timeout:
                # Check if already signaled
                if self._signaled.get(request_id, False):
                    return self._decisions.get(request_id)
                # Wait on Event with short timeout
                try:
                    await asyncio.wait_for(
                        self._waiting[request_id].wait(),
                        timeout=min(poll_interval, wait_timeout - elapsed),
                    )
                    return self._decisions.get(request_id)
                except asyncio.TimeoutError:
                    elapsed += poll_interval
                    continue
            # Overall timeout
            request["status"] = ApprovalStatus.TIMEOUT.value
            return None
        except asyncio.TimeoutError:
            request["status"] = ApprovalStatus.TIMEOUT.value
            return None

        return None

    def get_pending_requests(
        self,
        workflow_id: Optional[str] = None,
    ) -> List[ApprovalRequest]:
        """Get all pending approval requests.

        Args:
            workflow_id: Optional filter by workflow

        Returns:
            List of pending requests
        """
        requests = [
            r for r in self._pending_requests.values()
            if r["status"] == ApprovalStatus.PENDING.value
        ]

        if workflow_id:
            requests = [r for r in requests if r["workflow_id"] == workflow_id]

        return requests

    def cancel_request(self, request_id: str) -> bool:
        """Cancel an approval request.

        Args:
            request_id: Request identifier

        Returns:
            True if cancelled
        """
        if request_id not in self._pending_requests:
            return False

        self._pending_requests[request_id]["status"] = ApprovalStatus.CANCELLED.value

        if request_id in self._waiting:
            self._waiting[request_id].set()

        return True


# =============================================================================
# BREAKPOINT MANAGER
# =============================================================================


class BreakpointManager:
    """Manages breakpoints for workflow debugging.

    Allows setting breakpoints before/after nodes or on conditions.
    """

    def __init__(
        self,
        enabled: bool = BREAKPOINT_ENABLED,
    ):
        """Initialize breakpoint manager.

        Args:
            enabled: Whether breakpoints are enabled
        """
        self.enabled = enabled
        self._breakpoints: Dict[str, BreakpointConfig] = {}

    def add_breakpoint(
        self,
        breakpoint_type: BreakpointType,
        node_id: Optional[str] = None,
        condition: Optional[str] = None,
    ) -> BreakpointConfig:
        """Add a breakpoint.

        Args:
            breakpoint_type: Type of breakpoint
            node_id: Node to break at (for node breakpoints)
            condition: Condition expression (for conditional breakpoints)

        Returns:
            BreakpointConfig for the new breakpoint
        """
        import uuid

        breakpoint_id = str(uuid.uuid4())[:8]

        config = BreakpointConfig(
            breakpoint_id=breakpoint_id,
            breakpoint_type=breakpoint_type.value,
            node_id=node_id,
            condition=condition,
            enabled=True,
            hit_count=0,
        )

        self._breakpoints[breakpoint_id] = config
        return config

    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """Remove a breakpoint.

        Args:
            breakpoint_id: Breakpoint identifier

        Returns:
            True if removed
        """
        if breakpoint_id in self._breakpoints:
            del self._breakpoints[breakpoint_id]
            return True
        return False

    def toggle_breakpoint(
        self,
        breakpoint_id: str,
        enabled: Optional[bool] = None,
    ) -> bool:
        """Toggle a breakpoint on/off.

        Args:
            breakpoint_id: Breakpoint identifier
            enabled: New state (toggles if None)

        Returns:
            New enabled state
        """
        if breakpoint_id not in self._breakpoints:
            return False

        bp = self._breakpoints[breakpoint_id]
        bp["enabled"] = not bp["enabled"] if enabled is None else enabled
        return bp["enabled"]

    def should_break(
        self,
        node_id: str,
        state: Dict[str, Any],
        is_before: bool = True,
    ) -> Optional[BreakpointConfig]:
        """Check if execution should break.

        Args:
            node_id: Current node
            state: Current state
            is_before: Whether this is before or after node

        Returns:
            BreakpointConfig if should break, None otherwise
        """
        if not self.enabled:
            return None

        for bp in self._breakpoints.values():
            if not bp["enabled"]:
                continue

            # Check node breakpoints
            if bp["node_id"] == node_id:
                bp_type = BreakpointType(bp["breakpoint_type"])
                if (
                    (is_before and bp_type == BreakpointType.BEFORE_NODE) or
                    (not is_before and bp_type == BreakpointType.AFTER_NODE)
                ):
                    bp["hit_count"] += 1
                    return bp

            # Check conditional breakpoints
            if bp["breakpoint_type"] == BreakpointType.ON_CONDITION.value:
                if self._evaluate_condition(bp["condition"], state):
                    bp["hit_count"] += 1
                    return bp

        return None

    def _evaluate_condition(
        self,
        condition: Optional[str],
        state: Dict[str, Any],
    ) -> bool:
        """Evaluate a condition expression.

        Args:
            condition: Condition expression
            state: Current state

        Returns:
            True if condition is met
        """
        if not condition:
            return False

        # Simple key-value matching for safety
        try:
            if "==" in condition:
                key, value = condition.split("==", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                return str(state.get(key)) == value
            elif "in" in condition:
                value, key = condition.split(" in ", 1)
                value = value.strip().strip("'\"")
                key = key.strip()
                return value in str(state.get(key, ""))
        except Exception as parse_err:
            # Bloco 5 FIX: Log condition parsing failures
            logger.debug(f"Failed to parse breakpoint condition '{condition}': {parse_err}")

        return False

    def list_breakpoints(self) -> List[BreakpointConfig]:
        """List all breakpoints.

        Returns:
            List of breakpoint configurations
        """
        return list(self._breakpoints.values())


# =============================================================================
# STREAMING WORKFLOW
# =============================================================================


class StreamingWorkflow:
    """Workflow execution with streaming and human-in-the-loop support.

    Combines stream handling, checkpointing, and human approvals
    for enhanced workflow execution.
    """

    def __init__(
        self,
        checkpointer: Optional[EnhancedCheckpointer] = None,
        human_loop: Optional[HumanLoop] = None,
        breakpoint_manager: Optional[BreakpointManager] = None,
    ):
        """Initialize streaming workflow.

        Args:
            checkpointer: Custom checkpointer (creates default if None)
            human_loop: Custom human loop manager
            breakpoint_manager: Custom breakpoint manager
        """
        self.checkpointer = checkpointer or EnhancedCheckpointer()
        self.human_loop = human_loop or HumanLoop()
        self.breakpoint_manager = breakpoint_manager or BreakpointManager()

        self._stream_handler: Optional[StreamHandler] = None
        self._workflow_id: Optional[str] = None

    async def execute_with_streaming(
        self,
        workflow_id: str,
        initial_state: Dict[str, Any],
        nodes: List[Dict[str, Any]],
        approval_nodes: Optional[List[str]] = None,
    ) -> AsyncIterator[StreamEvent]:
        """Execute workflow with streaming output.

        Args:
            workflow_id: Workflow identifier
            initial_state: Initial state
            nodes: List of node definitions
            approval_nodes: Nodes requiring approval

        Yields:
            StreamEvent for each step
        """
        import uuid

        self._workflow_id = workflow_id
        self._stream_handler = StreamHandler()
        approval_nodes = approval_nodes or []

        checkpoint_id = f"wf-{workflow_id}"
        current_state = initial_state.copy()

        try:
            for node in nodes:
                node_id = node.get("id", str(uuid.uuid4())[:8])
                node_name = node.get("name", node_id)

                # Check breakpoints before node
                bp = self.breakpoint_manager.should_break(node_id, current_state, is_before=True)
                if bp:
                    await self._stream_handler.emit(
                        StreamEventType.CHECKPOINT,
                        data={"breakpoint": bp, "state": current_state},
                        node_id=node_id,
                    )
                    # In production, would wait for debugger to continue

                # Emit node start
                await self._stream_handler.emit(
                    StreamEventType.NODE_START,
                    data={"node_name": node_name},
                    node_id=node_id,
                )

                # Check if approval needed
                if node_id in approval_nodes:
                    # Save checkpoint before approval
                    version = await self.checkpointer.save_checkpoint(
                        checkpoint_id,
                        current_state,
                        node_id,
                    )

                    # Request approval
                    request = await self.human_loop.request_approval(
                        workflow_id=workflow_id,
                        node_id=node_id,
                        checkpoint_id=version["version_id"],
                        reason=f"Approval required before executing {node_name}",
                        context={"state": current_state},
                    )

                    await self._stream_handler.emit(
                        StreamEventType.APPROVAL_NEEDED,
                        data={"request": request},
                        node_id=node_id,
                    )

                    # Wait for decision
                    decision = await self.human_loop.wait_for_decision(request["request_id"])

                    if not decision or decision["decision"] == "reject":
                        await self._stream_handler.error(
                            f"Approval rejected for node {node_id}"
                        )
                        return

                # Execute node (simulated)
                node_handler = node.get("handler")
                if node_handler:
                    try:
                        result = await node_handler(current_state)
                        current_state.update(result or {})
                    except Exception as e:
                        await self._stream_handler.emit(
                            StreamEventType.ERROR,
                            data={"error": str(e)},
                            node_id=node_id,
                        )
                        continue

                # Emit node output
                await self._stream_handler.emit(
                    StreamEventType.NODE_OUTPUT,
                    data={"state": current_state},
                    node_id=node_id,
                )

                # Save checkpoint
                await self.checkpointer.save_checkpoint(
                    checkpoint_id,
                    current_state,
                    node_id,
                )

                await self._stream_handler.emit(
                    StreamEventType.CHECKPOINT,
                    data={"version": checkpoint_id},
                    node_id=node_id,
                )

                # Emit node end
                await self._stream_handler.emit(
                    StreamEventType.NODE_END,
                    data={"node_name": node_name},
                    node_id=node_id,
                )

                # Yield events from buffer
                while not self._stream_handler._buffer.empty():
                    event = await self._stream_handler._buffer.get()
                    yield event

            await self._stream_handler.complete(current_state)

            # Yield remaining events
            while not self._stream_handler._buffer.empty():
                event = await self._stream_handler._buffer.get()
                yield event

        except Exception as e:
            await self._stream_handler.error(str(e))
            while not self._stream_handler._buffer.empty():
                event = await self._stream_handler._buffer.get()
                yield event

    async def resume_from_checkpoint(
        self,
        checkpoint_id: str,
        version_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Resume from a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier
            version_id: Optional specific version

        Returns:
            State at checkpoint or None
        """
        return await self.checkpointer.load_checkpoint(checkpoint_id, version_id)

    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status.

        Returns:
            Status dictionary
        """
        return {
            "workflow_id": self._workflow_id,
            "pending_approvals": len(self.human_loop.get_pending_requests(self._workflow_id)),
            "active_breakpoints": len([
                bp for bp in self.breakpoint_manager.list_breakpoints()
                if bp["enabled"]
            ]),
            "checkpoint_count": sum(
                len(versions)
                for versions in self.checkpointer._checkpoints.values()
            ),
        }


# =============================================================================
# SINGLETON
# =============================================================================

_streaming_workflow: Optional[StreamingWorkflow] = None


def get_streaming_workflow() -> StreamingWorkflow:
    """Get singleton streaming workflow instance.

    Returns:
        StreamingWorkflow instance
    """
    global _streaming_workflow
    if _streaming_workflow is None:
        _streaming_workflow = StreamingWorkflow()
    return _streaming_workflow


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def stream_workflow_execution(
    workflow_id: str,
    initial_state: Dict[str, Any],
    nodes: List[Dict[str, Any]],
) -> AsyncIterator[StreamEvent]:
    """Stream workflow execution.

    Convenience function for quick streaming.
    """
    workflow = get_streaming_workflow()
    async for event in workflow.execute_with_streaming(
        workflow_id,
        initial_state,
        nodes,
    ):
        yield event


async def create_approval_request(
    workflow_id: str,
    node_id: str,
    reason: str,
) -> ApprovalRequest:
    """Create an approval request.

    Convenience function for approvals.
    """
    workflow = get_streaming_workflow()
    return await workflow.human_loop.request_approval(
        workflow_id=workflow_id,
        node_id=node_id,
        checkpoint_id="",
        reason=reason,
    )


async def resume_after_approval(
    checkpoint_id: str,
    version_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Resume workflow after approval.

    Convenience function for resuming.
    """
    workflow = get_streaming_workflow()
    return await workflow.resume_from_checkpoint(checkpoint_id, version_id)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "StreamingWorkflow",
    "StreamHandler",
    "EnhancedCheckpointer",
    "HumanLoop",
    "BreakpointManager",
    # Result types
    "StreamEvent",
    "CheckpointVersion",
    "ApprovalRequest",
    "HumanDecision",
    "BreakpointConfig",
    "StreamingResult",
    # Enums
    "StreamEventType",
    "ApprovalStatus",
    "BreakpointType",
    "CheckpointStrategy",
    # Functions
    "get_streaming_workflow",
    "stream_workflow_execution",
    "create_approval_request",
    "resume_after_approval",
    # Constants
    "STREAMING_AVAILABLE",
    "LANGGRAPH_AVAILABLE",
]
