"""Trust Boundary Enforcement for LangGraph Pipeline.

This module implements trust boundaries between agent tiers to prevent
unauthorized access and enforce the principle of least privilege.

Addresses: RED AGENT CRIT-03 - Missing Trust Boundary Enforcement

Architecture:
    L0 (System)     - Highest privilege, can access everything
    L1 (Executive)  - Can manage L2-L6, access governance
    L2 (VPs)        - Can manage L3-L6, access specs/execution
    L3 (Masters)    - Can manage L4-L6, access assigned domains
    L4 (Squad Lead) - Can manage L5-L6, limited scope
    L5 (Workers)    - Execute only, no management
    L6 (Specialists)- Domain-specific, isolated

Key Features:
- Tier-based access control
- Resource isolation per agent
- Action authorization checks
- Audit logging of boundary crossings

Author: Pipeline Autonomo Team
Version: 1.0.0 (2026-01-16)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# =============================================================================
# VULN-CRIT-001 FIX: Environment-based strict mode enforcement
# =============================================================================

def _is_production_environment() -> bool:
    """Detect if running in production environment.

    Returns True if any of the following conditions are met:
    - ENVIRONMENT or ENV is set to 'production' or 'prod'
    - PIPELINE_ENV is set to 'production' or 'prod'
    - STRICT_MODE_FORCE is set to 'true' or '1'
    """
    env_vars = [
        os.getenv("ENVIRONMENT", "").lower(),
        os.getenv("ENV", "").lower(),
        os.getenv("PIPELINE_ENV", "").lower(),
    ]
    production_values = {"production", "prod"}

    if any(env in production_values for env in env_vars):
        return True

    # Allow explicit forcing of strict mode
    force_strict = os.getenv("STRICT_MODE_FORCE", "").lower()
    if force_strict in ("true", "1", "yes"):
        return True

    return False


def _get_effective_strict_mode(requested_strict_mode: bool) -> bool:
    """Get the effective strict_mode value, enforcing True in production.

    VULN-CRIT-001 FIX: In production environments, strict_mode is ALWAYS True,
    regardless of what was requested. This prevents trust boundary bypass attacks.

    Args:
        requested_strict_mode: The strict_mode value requested by the caller.

    Returns:
        True in production (always), otherwise the requested value.
    """
    if _is_production_environment():
        if not requested_strict_mode:
            logger.warning(
                "SECURITY: strict_mode=False requested in production environment. "
                "Forcing strict_mode=True to prevent trust boundary bypass (VULN-CRIT-001)."
            )
        return True
    return requested_strict_mode


# =============================================================================
# AGENT TIERS
# =============================================================================


class AgentTier(IntEnum):
    """Agent hierarchy tiers with decreasing privilege."""

    L0_SYSTEM = 0       # run_master, pack_driver, ops_ctrl
    L1_EXECUTIVE = 1    # presidente, ceo, arbiter
    L2_VP = 2           # spec_vp, exec_vp
    L3_MASTER = 3       # spec_master, ace_exec, qa_master
    L4_SQUAD_LEAD = 4   # squad_lead_*
    L5_WORKER = 5       # auditor, refinador, developer
    L6_SPECIALIST = 6   # oracle_architect, security_specialist


# Agent to tier mapping
# GAP-001 FIX: Complete mapping of ALL 34+ cerebros across 7 tiers (L0-L6)
# Reference: AUDIT_HIERARCHY_ENFORCEMENT.md, hierarchy.py, cerebro_stacks/*.yaml
AGENT_TIERS: Dict[str, AgentTier] = {
    # =========================================================================
    # L0 - System Controllers (5 cerebros)
    # CONF-008 FIX: Clear authority domains defined:
    #   - pack_driver: Pipeline orchestrator (controls WHAT runs)
    #   - ops_ctrl: Operations watchdog (controls IF it can run - has VETO power)
    #   - run_master: Pipeline caretaker (controls HOW it evolves)
    #   - orchestrator: System orchestrator (LangGraph workflow coordination)
    #   - run_supervisor: System supervisor (monitoring and health)
    # CONFLICT RESOLUTION: ops_ctrl has veto power over pack_driver (safety first)
    # =========================================================================
    "run_master": AgentTier.L0_SYSTEM,
    "pack_driver": AgentTier.L0_SYSTEM,
    "ops_ctrl": AgentTier.L0_SYSTEM,
    "orchestrator": AgentTier.L0_SYSTEM,
    "run_supervisor": AgentTier.L0_SYSTEM,

    # =========================================================================
    # L1 - Executive (3 cerebros)
    # =========================================================================
    "presidente": AgentTier.L1_EXECUTIVE,
    "ceo": AgentTier.L1_EXECUTIVE,
    "arbiter": AgentTier.L1_EXECUTIVE,

    # =========================================================================
    # L2 - VPs (3 cerebros)
    # DRIFT-005 FIX: qa_vp now properly included (was missing in hierarchy.py)
    # =========================================================================
    "spec_vp": AgentTier.L2_VP,
    "exec_vp": AgentTier.L2_VP,
    "qa_vp": AgentTier.L2_VP,

    # =========================================================================
    # L3 - Masters (5 cerebros)
    # CONF-003 FIX: integration_officer is L3 (reports to CEO), NOT L6
    # =========================================================================
    "spec_master": AgentTier.L3_MASTER,
    "ace_exec": AgentTier.L3_MASTER,
    "qa_master": AgentTier.L3_MASTER,
    "sprint_planner": AgentTier.L3_MASTER,
    "external_liaison": AgentTier.L3_MASTER,
    "integration_officer": AgentTier.L3_MASTER,  # CONF-003: Resolved - L3 per hierarchy.py
    "retrospective_master": AgentTier.L3_MASTER,  # Reports to CEO per hierarchy.py

    # =========================================================================
    # L4 - Squad Leads (11 cerebros)
    # CONF-006 FIX: All squad leads from YAMLs now included
    # =========================================================================
    "squad_lead_spec": AgentTier.L4_SQUAD_LEAD,
    "squad_lead_exec": AgentTier.L4_SQUAD_LEAD,
    "squad_lead_qa": AgentTier.L4_SQUAD_LEAD,
    "squad_lead_ops": AgentTier.L4_SQUAD_LEAD,
    "squad_lead_sec": AgentTier.L4_SQUAD_LEAD,
    "squad_lead_p1": AgentTier.L4_SQUAD_LEAD,
    "squad_lead_p2": AgentTier.L4_SQUAD_LEAD,
    "squad_lead_p3": AgentTier.L4_SQUAD_LEAD,
    "squad_lead_p4": AgentTier.L4_SQUAD_LEAD,
    "squad_lead_p5": AgentTier.L4_SQUAD_LEAD,
    "squad_lead_hlx": AgentTier.L4_SQUAD_LEAD,

    # =========================================================================
    # L5 - Workers (9 cerebros)
    # =========================================================================
    "developer": AgentTier.L5_WORKER,
    "auditor": AgentTier.L5_WORKER,
    "refinador": AgentTier.L5_WORKER,
    "tester": AgentTier.L5_WORKER,
    "tech_writer": AgentTier.L5_WORKER,
    "reviewer": AgentTier.L5_WORKER,
    "integrator": AgentTier.L5_WORKER,
    "debugger": AgentTier.L5_WORKER,
    "optimizer": AgentTier.L5_WORKER,

    # =========================================================================
    # L6 - Specialists (10 cerebros)
    # DRIFT-001/DRIFT-002: oracle_architect and security_specialist have
    # elevated capabilities documented as intentional (domain expertise requires it)
    # =========================================================================
    "oracle_architect": AgentTier.L6_SPECIALIST,
    "security_specialist": AgentTier.L6_SPECIALIST,
    "data_scientist": AgentTier.L6_SPECIALIST,
    "performance_analyst": AgentTier.L6_SPECIALIST,
    "ux_specialist": AgentTier.L6_SPECIALIST,
    "devops_engineer": AgentTier.L6_SPECIALIST,
    "ml_engineer": AgentTier.L6_SPECIALIST,
    "knowledge_engineer": AgentTier.L6_SPECIALIST,
    "llm_orchestrator": AgentTier.L6_SPECIALIST,

    # =========================================================================
    # Additional agents from hierarchy.py (workers under various squad leads)
    # =========================================================================
    # QA team workers
    "agente_auditor": AgentTier.L5_WORKER,
    "clean_reviewer": AgentTier.L5_WORKER,
    "edge_case_hunter": AgentTier.L5_WORKER,
    "gap_hunter": AgentTier.L5_WORKER,
    "human_reviewer": AgentTier.L5_WORKER,
    "debt_tracker": AgentTier.L5_WORKER,

    # Spec team workers
    "product_owner": AgentTier.L5_WORKER,
    "project_manager": AgentTier.L5_WORKER,
    "task_decomposer": AgentTier.L5_WORKER,

    # Ops team workers
    "technical_planner": AgentTier.L5_WORKER,
    "dependency_mapper": AgentTier.L5_WORKER,

    # Security team workers
    "red_team_agent": AgentTier.L5_WORKER,

    # Human layer
    "human_approver": AgentTier.L3_MASTER,  # CONF-007: L3, reports to CEO
    "human_layer_specialist": AgentTier.L5_WORKER,

    # Additional L6 specialists from hierarchy.py
    "ui_designer": AgentTier.L6_SPECIALIST,
    "ux_researcher": AgentTier.L6_SPECIALIST,
    "blockchain_engineer": AgentTier.L6_SPECIALIST,
    "data_engineer": AgentTier.L6_SPECIALIST,
    "legal_tech_specialist": AgentTier.L6_SPECIALIST,
    "web3_frontend": AgentTier.L6_SPECIALIST,

    # System support agents
    "resource_optimizer": AgentTier.L0_SYSTEM,
    "system_observer": AgentTier.L0_SYSTEM,
    "ace_orchestration": AgentTier.L4_SQUAD_LEAD,
    "human_layer": AgentTier.L4_SQUAD_LEAD,
}

# =============================================================================
# BLACK-002 FIX: Agent ID Whitelist for Spoofing Prevention
# =============================================================================

# Frozen set of valid agent IDs - only these can be used to access resources
VALID_AGENT_IDS: frozenset[str] = frozenset(AGENT_TIERS.keys())

# Pattern for validating agent ID format
import re
AGENT_ID_PATTERN = re.compile(r'^[a-z][a-z0-9_]{2,50}$')


def validate_agent_id(agent_id: str) -> tuple[bool, str]:
    """BLACK-002 FIX: Validate agent ID against whitelist and format rules.

    This function prevents agent ID spoofing by:
    1. Checking format constraints (lowercase, alphanumeric with underscores)
    2. Validating against the whitelist of known agents
    3. Rejecting IDs that look like system or privileged agents but aren't registered

    Args:
        agent_id: The agent ID to validate.

    Returns:
        Tuple of (is_valid, reason).
    """
    if not agent_id:
        return False, "BLACK-002: Empty agent ID is not allowed"

    # Normalize to lowercase
    agent_id_lower = agent_id.lower()

    # Check format
    if not AGENT_ID_PATTERN.match(agent_id_lower):
        return False, f"BLACK-002: Invalid agent ID format '{agent_id}'"

    # Check against whitelist
    if agent_id_lower not in VALID_AGENT_IDS:
        # Check if it looks like a privileged agent name (spoofing attempt)
        privileged_prefixes = ("run_", "pack_", "ops_", "presidente", "ceo", "arbiter")
        if agent_id_lower.startswith(privileged_prefixes):
            logger.warning(
                f"BLACK-002 SPOOFING ATTEMPT: Agent ID '{agent_id}' looks like a "
                f"privileged agent but is not in whitelist"
            )
            return False, f"BLACK-002: Potential spoofing attempt detected for '{agent_id}'"

        # Unknown agent - not in whitelist
        return False, f"BLACK-002: Unknown agent ID '{agent_id}' not in whitelist"

    return True, "Agent ID validated"


def is_privileged_agent(agent_id: str) -> bool:
    """BLACK-002 FIX: Check if an agent ID corresponds to a privileged tier.

    Args:
        agent_id: The agent ID to check.

    Returns:
        True if the agent is L0, L1, or L2 tier.
    """
    agent_id_lower = agent_id.lower()
    tier = AGENT_TIERS.get(agent_id_lower)
    if tier is None:
        return False
    return tier in (AgentTier.L0_SYSTEM, AgentTier.L1_EXECUTIVE, AgentTier.L2_VP)


# =============================================================================
# RESOURCE TYPES
# =============================================================================


class ResourceType(str):
    """Types of resources that can be accessed."""

    # Configuration
    CONFIG_PIPELINE = "config:pipeline"
    CONFIG_STACKS = "config:stacks"
    CONFIG_AGENTS = "config:agents"

    # Data
    DATA_STATE = "data:state"
    DATA_CHECKPOINTS = "data:checkpoints"
    DATA_EVENTS = "data:events"
    DATA_ARTIFACTS = "data:artifacts"

    # Governance
    GOV_SIGNOFFS = "gov:signoffs"
    GOV_APPROVALS = "gov:approvals"
    GOV_GATES = "gov:gates"

    # Execution
    EXEC_SPRINTS = "exec:sprints"
    EXEC_TASKS = "exec:tasks"
    EXEC_CREWS = "exec:crews"

    # System
    SYS_HALT = "sys:halt"
    SYS_RESUME = "sys:resume"
    SYS_RESET = "sys:reset"


# =============================================================================
# ACTIONS
# =============================================================================


class Action(str):
    """Actions that can be performed on resources."""

    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"
    EXECUTE = "execute"
    APPROVE = "approve"
    REJECT = "reject"


# =============================================================================
# ACCESS POLICIES
# =============================================================================


# Default access policies by tier
# Format: tier -> {resource_type: set of allowed actions}
DEFAULT_ACCESS_POLICIES: Dict[AgentTier, Dict[str, Set[str]]] = {
    AgentTier.L0_SYSTEM: {
        # System tier can do everything
        "*": {Action.READ, Action.WRITE, Action.CREATE, Action.DELETE, Action.EXECUTE, Action.APPROVE, Action.REJECT},
    },
    AgentTier.L1_EXECUTIVE: {
        ResourceType.CONFIG_PIPELINE: {Action.READ},
        ResourceType.CONFIG_STACKS: {Action.READ},
        ResourceType.CONFIG_AGENTS: {Action.READ, Action.WRITE},
        ResourceType.DATA_STATE: {Action.READ, Action.WRITE},
        ResourceType.DATA_CHECKPOINTS: {Action.READ},
        ResourceType.DATA_EVENTS: {Action.READ},
        ResourceType.DATA_ARTIFACTS: {Action.READ},
        ResourceType.GOV_SIGNOFFS: {Action.READ, Action.WRITE, Action.APPROVE},
        ResourceType.GOV_APPROVALS: {Action.READ, Action.WRITE, Action.APPROVE},
        ResourceType.GOV_GATES: {Action.READ},
        ResourceType.EXEC_SPRINTS: {Action.READ, Action.EXECUTE},
        ResourceType.EXEC_TASKS: {Action.READ},
        ResourceType.EXEC_CREWS: {Action.READ},
        ResourceType.SYS_HALT: {Action.EXECUTE},
        ResourceType.SYS_RESUME: {Action.EXECUTE},
    },
    AgentTier.L2_VP: {
        ResourceType.CONFIG_PIPELINE: {Action.READ},
        ResourceType.DATA_STATE: {Action.READ, Action.WRITE},
        ResourceType.DATA_EVENTS: {Action.READ},
        ResourceType.DATA_ARTIFACTS: {Action.READ, Action.WRITE},
        ResourceType.GOV_SIGNOFFS: {Action.READ, Action.WRITE},
        ResourceType.GOV_GATES: {Action.READ},
        ResourceType.EXEC_SPRINTS: {Action.READ, Action.EXECUTE},
        ResourceType.EXEC_TASKS: {Action.READ, Action.WRITE, Action.CREATE},
        ResourceType.EXEC_CREWS: {Action.READ, Action.CREATE},
    },
    AgentTier.L3_MASTER: {
        ResourceType.DATA_STATE: {Action.READ},
        ResourceType.DATA_EVENTS: {Action.READ},
        ResourceType.DATA_ARTIFACTS: {Action.READ, Action.WRITE, Action.CREATE},
        ResourceType.GOV_SIGNOFFS: {Action.READ, Action.WRITE},
        ResourceType.GOV_GATES: {Action.READ, Action.EXECUTE},
        # FIX: ace_exec (L3) needs EXECUTE permission on sprints - that's its primary role
        ResourceType.EXEC_SPRINTS: {Action.READ, Action.EXECUTE},
        ResourceType.EXEC_TASKS: {Action.READ, Action.WRITE, Action.CREATE, Action.EXECUTE},
        ResourceType.EXEC_CREWS: {Action.READ},
    },
    AgentTier.L4_SQUAD_LEAD: {
        ResourceType.DATA_STATE: {Action.READ},
        ResourceType.DATA_EVENTS: {Action.READ},
        ResourceType.DATA_ARTIFACTS: {Action.READ, Action.CREATE},
        ResourceType.GOV_SIGNOFFS: {Action.WRITE},
        ResourceType.EXEC_TASKS: {Action.READ, Action.WRITE, Action.EXECUTE},
    },
    AgentTier.L5_WORKER: {
        ResourceType.DATA_STATE: {Action.READ},
        ResourceType.DATA_EVENTS: {Action.READ},
        ResourceType.DATA_ARTIFACTS: {Action.CREATE},
        ResourceType.EXEC_TASKS: {Action.READ, Action.EXECUTE},
    },
    AgentTier.L6_SPECIALIST: {
        # Specialists have domain-specific access (defined per agent)
        ResourceType.DATA_STATE: {Action.READ},
        ResourceType.DATA_ARTIFACTS: {Action.READ, Action.CREATE},
    },
}


# =============================================================================
# TRUST BOUNDARY VIOLATION
# =============================================================================


@dataclass
class TrustBoundaryViolation:
    """Record of a trust boundary violation."""

    agent_id: str
    agent_tier: AgentTier
    resource: str
    action: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    blocked: bool = True

    def __str__(self) -> str:
        status = "BLOCKED" if self.blocked else "ALLOWED"
        return f"[{status}] Agent '{self.agent_id}' (L{self.agent_tier}) attempted {self.action} on {self.resource}: {self.message}"


@dataclass
class AccessCheckResult:
    """Result of an access check."""

    allowed: bool
    agent_id: str
    resource: str
    action: str
    reason: str
    violation: Optional[TrustBoundaryViolation] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "agent_id": self.agent_id,
            "resource": self.resource,
            "action": self.action,
            "reason": self.reason,
            "violation": str(self.violation) if self.violation else None,
        }


# =============================================================================
# TRUST BOUNDARY ENFORCER
# =============================================================================


class TrustBoundaryEnforcer:
    """Enforces trust boundaries between agent tiers.

    Usage:
        enforcer = TrustBoundaryEnforcer()
        result = enforcer.check_access("developer", "gov:signoffs", "approve")
        if not result.allowed:
            raise PermissionError(result.reason)
    """

    def __init__(
        self,
        policies: Optional[Dict[AgentTier, Dict[str, Set[str]]]] = None,
        agent_tiers: Optional[Dict[str, AgentTier]] = None,
        strict_mode: bool = True,
    ):
        """Initialize the enforcer.

        Args:
            policies: Custom access policies. Uses defaults if None.
            agent_tiers: Custom agent-to-tier mapping. Uses defaults if None.
            strict_mode: If True, unknown agents are denied. If False, warned but allowed.
                        NOTE (VULN-CRIT-001 FIX): In production environments, strict_mode
                        is ALWAYS True regardless of this parameter to prevent bypass attacks.
        """
        self.policies = policies or DEFAULT_ACCESS_POLICIES.copy()
        self.agent_tiers = agent_tiers or AGENT_TIERS.copy()
        # VULN-CRIT-001 FIX: Enforce strict_mode=True in production
        self.strict_mode = _get_effective_strict_mode(strict_mode)
        self._violations: List[TrustBoundaryViolation] = []

    def get_agent_tier(self, agent_id: str) -> Optional[AgentTier]:
        """Get the tier for an agent.

        Args:
            agent_id: Agent identifier.

        Returns:
            AgentTier or None if unknown.
        """
        return self.agent_tiers.get(agent_id.lower())

    def check_access(
        self,
        agent_id: str,
        resource: str,
        action: str,
    ) -> AccessCheckResult:
        """Check if an agent can perform an action on a resource.

        Args:
            agent_id: Agent attempting the action.
            resource: Resource being accessed.
            action: Action being performed.

        Returns:
            AccessCheckResult with allowed status and reason.
        """
        # BLACK-002 FIX: Validate agent ID against whitelist first
        is_valid, validation_reason = validate_agent_id(agent_id)
        if not is_valid and self.strict_mode:
            violation = TrustBoundaryViolation(
                agent_id=agent_id,
                agent_tier=AgentTier.L6_SPECIALIST,
                resource=resource,
                action=action,
                message=validation_reason,
                blocked=True,
            )
            self._violations.append(violation)
            return AccessCheckResult(
                allowed=False,
                agent_id=agent_id,
                resource=resource,
                action=action,
                reason=validation_reason,
                violation=violation,
            )

        agent_id_lower = agent_id.lower()
        tier = self.get_agent_tier(agent_id_lower)

        # Unknown agent handling
        if tier is None:
            if self.strict_mode:
                violation = TrustBoundaryViolation(
                    agent_id=agent_id,
                    agent_tier=AgentTier.L6_SPECIALIST,  # Default to lowest
                    resource=resource,
                    action=action,
                    message="Unknown agent denied in strict mode",
                    blocked=True,
                )
                self._violations.append(violation)
                return AccessCheckResult(
                    allowed=False,
                    agent_id=agent_id,
                    resource=resource,
                    action=action,
                    reason="Unknown agent denied in strict mode",
                    violation=violation,
                )
            else:
                logger.warning(f"Unknown agent '{agent_id}' - allowing in non-strict mode")
                tier = AgentTier.L5_WORKER  # Default to worker tier

        # Get policies for this tier
        tier_policies = self.policies.get(tier, {})

        # Check wildcard first (L0 has wildcard access)
        if "*" in tier_policies:
            allowed_actions = tier_policies["*"]
            if action in allowed_actions:
                return AccessCheckResult(
                    allowed=True,
                    agent_id=agent_id,
                    resource=resource,
                    action=action,
                    reason=f"Tier L{tier} has wildcard access",
                )

        # Check specific resource
        if resource in tier_policies:
            allowed_actions = tier_policies[resource]
            if action in allowed_actions:
                return AccessCheckResult(
                    allowed=True,
                    agent_id=agent_id,
                    resource=resource,
                    action=action,
                    reason=f"Action '{action}' allowed on '{resource}' for tier L{tier}",
                )

        # Access denied
        violation = TrustBoundaryViolation(
            agent_id=agent_id,
            agent_tier=tier,
            resource=resource,
            action=action,
            message=f"Tier L{tier} lacks '{action}' permission on '{resource}'",
            blocked=True,
        )
        self._violations.append(violation)

        return AccessCheckResult(
            allowed=False,
            agent_id=agent_id,
            resource=resource,
            action=action,
            reason=f"Access denied: tier L{tier} cannot '{action}' on '{resource}'",
            violation=violation,
        )

    def require_access(
        self,
        agent_id: str,
        resource: str,
        action: str,
    ) -> None:
        """Check access and raise PermissionError if denied.

        Args:
            agent_id: Agent attempting the action.
            resource: Resource being accessed.
            action: Action being performed.

        Raises:
            PermissionError: If access is denied.
        """
        result = self.check_access(agent_id, resource, action)
        if not result.allowed:
            logger.error(f"Trust boundary violation: {result.reason}")
            raise PermissionError(result.reason)

    def can_manage_agent(self, manager_id: str, subordinate_id: str) -> bool:
        """Check if one agent can manage another.

        A manager can only manage agents at lower tiers.

        Args:
            manager_id: Agent attempting to manage.
            subordinate_id: Agent being managed.

        Returns:
            True if management is allowed.
        """
        manager_tier = self.get_agent_tier(manager_id)
        subordinate_tier = self.get_agent_tier(subordinate_id)

        if manager_tier is None or subordinate_tier is None:
            return False

        # Manager must be at a higher tier (lower number)
        return manager_tier < subordinate_tier

    def get_violations(self) -> List[TrustBoundaryViolation]:
        """Get all recorded violations."""
        return self._violations.copy()

    def clear_violations(self) -> None:
        """Clear recorded violations."""
        self._violations.clear()


# =============================================================================
# SINGLETON ACCESS
# =============================================================================


_enforcer_instance: Optional[TrustBoundaryEnforcer] = None


def get_trust_boundary_enforcer() -> TrustBoundaryEnforcer:
    """Get the default trust boundary enforcer."""
    global _enforcer_instance
    if _enforcer_instance is None:
        _enforcer_instance = TrustBoundaryEnforcer()
    return _enforcer_instance


def check_access(agent_id: str, resource: str, action: str) -> AccessCheckResult:
    """Convenience function to check access."""
    return get_trust_boundary_enforcer().check_access(agent_id, resource, action)


def require_access(agent_id: str, resource: str, action: str) -> None:
    """Convenience function to require access."""
    get_trust_boundary_enforcer().require_access(agent_id, resource, action)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "AgentTier",
    "ResourceType",
    "Action",
    # Types
    "TrustBoundaryViolation",
    "AccessCheckResult",
    # Classes
    "TrustBoundaryEnforcer",
    # Functions
    "get_trust_boundary_enforcer",
    "check_access",
    "require_access",
    # Constants
    "AGENT_TIERS",
    "DEFAULT_ACCESS_POLICIES",
]
