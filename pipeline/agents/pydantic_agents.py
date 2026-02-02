"""Pydantic-AI Agents for Claim Verification.

Pydantic-AI provides type-safe AI agents with dependency injection.
Unlike Instructor (which patches LLM clients for structured output),
Pydantic-AI is designed for building complete agent systems with:
- Type-safe tool definitions
- Dependency injection for runtime context
- Structured result types
- Automatic retry and error handling

Key Differences from Instructor:
- Instructor: Patches existing LLM clients for structured OUTPUT
- Pydantic-AI: Builds complete agents with tools, context, and structured INTERACTIONS

Usage:
    from pipeline.agents.pydantic_agents import (
        claim_verifier_agent,
        run_claim_verification,
        VerificationContext,
    )

    # Create context with dependencies
    context = VerificationContext(
        claim_id="claim_123",
        knowledge_graph_url="http://graphrag:50052",
        llm_guard_url="http://llm-guard:50053",
    )

    # Run verification
    result = await run_claim_verification(
        claim="The sky is blue due to Rayleigh scattering",
        context=context,
    )
"""

from __future__ import annotations

import os
import logging
import asyncio
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
DEFAULT_MODEL = os.getenv("PYDANTIC_AI_MODEL", "claude-sonnet-4-20250514")

# Check if pydantic-ai is available AND API key is set
try:
    from pydantic_ai import Agent, RunContext, Tool
    from pydantic_ai.models.anthropic import AnthropicModel
    if not ANTHROPIC_API_KEY:
        PYDANTIC_AI_AVAILABLE = False
        logger.warning("Pydantic-AI: ANTHROPIC_API_KEY not set. Set the environment variable to enable.")
    else:
        PYDANTIC_AI_AVAILABLE = True
except ImportError:
    PYDANTIC_AI_AVAILABLE = False
    logger.warning("Pydantic-AI not available. Install with: pip install pydantic-ai")


# =============================================================================
# CONTEXT TYPES (Dependency Injection)
# =============================================================================


@dataclass
class VerificationContext:
    """Context for claim verification agent.

    Contains dependencies injected at runtime.
    """

    claim_id: str
    knowledge_graph_url: str = "http://localhost:50052"
    llm_guard_url: str = "http://localhost:50053"
    max_evidence_items: int = 10
    confidence_threshold: float = 0.7
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SourceContext:
    """Context for source evaluation agent."""

    source_id: str
    source_url: Optional[str] = None
    source_name: Optional[str] = None
    historical_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceContext:
    """Context for evidence gathering agent."""

    claim_text: str
    search_depth: int = 3
    include_contradicting: bool = True
    sources_to_check: List[str] = field(default_factory=list)


# =============================================================================
# RESULT TYPES
# =============================================================================


class VerificationVerdict(str, Enum):
    """Possible verdicts for claim verification."""
    TRUE = "TRUE"
    FALSE = "FALSE"
    PARTIALLY_TRUE = "PARTIALLY_TRUE"
    UNVERIFIABLE = "UNVERIFIABLE"
    MISLEADING = "MISLEADING"


class CredibilityLevel(str, Enum):
    """Credibility levels for sources."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class EvidenceType(str, Enum):
    """Types of evidence."""
    SUPPORTING = "supporting"
    CONTRADICTING = "contradicting"
    NEUTRAL = "neutral"


class EvidenceItem(BaseModel):
    """A piece of evidence."""

    text: str = Field(..., description="The evidence text")
    source: str = Field(..., description="Source of the evidence")
    evidence_type: EvidenceType = Field(..., description="Type of evidence")
    relevance: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    credibility: float = Field(..., ge=0.0, le=1.0, description="Source credibility")


class ClaimVerificationResult(BaseModel):
    """Result from claim verification agent."""

    claim_id: str = Field(..., description="ID of the verified claim")
    original_claim: str = Field(..., description="Original claim text")
    verdict: VerificationVerdict = Field(..., description="Verification verdict")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Reasoning for the verdict")
    evidence_used: List[EvidenceItem] = Field(default_factory=list)
    key_findings: List[str] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    verification_time: datetime = Field(default_factory=datetime.utcnow)


class SourceEvaluationResult(BaseModel):
    """Result from source evaluation agent."""

    source_id: str = Field(..., description="ID of the source")
    source_name: str = Field(..., description="Name of the source")
    credibility_score: float = Field(..., ge=0.0, le=1.0)
    credibility_level: CredibilityLevel = Field(...)
    factors: Dict[str, float] = Field(default_factory=dict)
    known_biases: List[str] = Field(default_factory=list)
    reasoning: str = Field(...)
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class EvidenceGatheringResult(BaseModel):
    """Result from evidence gathering agent."""

    claim_text: str = Field(..., description="The claim for which evidence was gathered")
    total_evidence_found: int = Field(..., description="Total pieces of evidence found")
    supporting_evidence: List[EvidenceItem] = Field(default_factory=list)
    contradicting_evidence: List[EvidenceItem] = Field(default_factory=list)
    neutral_evidence: List[EvidenceItem] = Field(default_factory=list)
    search_sources: List[str] = Field(default_factory=list)
    gathering_time: datetime = Field(default_factory=datetime.utcnow)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

if PYDANTIC_AI_AVAILABLE:

    async def search_knowledge_graph(
        ctx: RunContext[VerificationContext],
        query: str,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Search the knowledge graph for relevant information.

        Args:
            ctx: Run context with dependencies
            query: Search query
            top_k: Number of results to return

        Returns:
            Search results from knowledge graph
        """
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{ctx.deps.knowledge_graph_url}/query",
                    json={
                        "query": query,
                        "search_type": "local",
                        "top_k": top_k,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Knowledge graph search failed: {e}")
            return {"error": str(e), "results": []}

    async def check_source_credibility(
        ctx: RunContext[VerificationContext],
        source_name: str,
    ) -> Dict[str, Any]:
        """Check the credibility of a source.

        Args:
            ctx: Run context with dependencies
            source_name: Name of the source to check

        Returns:
            Credibility information for the source
        """
        # This would integrate with a source credibility database
        # For now, return a placeholder
        return {
            "source_name": source_name,
            "credibility_score": 0.5,
            "credibility_level": "medium",
            "known_biases": [],
        }

    async def find_related_claims(
        ctx: RunContext[VerificationContext],
        claim_text: str,
        min_similarity: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Find claims related to the given claim.

        Args:
            ctx: Run context with dependencies
            claim_text: The claim to find related claims for
            min_similarity: Minimum similarity threshold

        Returns:
            List of related claims
        """
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{ctx.deps.knowledge_graph_url}/claims/similar",
                    params={
                        "claim_text": claim_text,
                        "top_k": 10,
                        "min_similarity": min_similarity,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Find related claims failed: {e}")
            return []

    async def validate_security(
        ctx: RunContext[VerificationContext],
        text: str,
    ) -> Dict[str, Any]:
        """Validate text for security issues using LLM Guard.

        Args:
            ctx: Run context with dependencies
            text: Text to validate

        Returns:
            Security scan results
        """
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{ctx.deps.llm_guard_url}/scan/input",
                    json={"text": text, "sanitize": False},
                    timeout=30.0,
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            return {"is_safe": True, "error": str(e)}


# =============================================================================
# AGENT DEFINITIONS
# =============================================================================

if PYDANTIC_AI_AVAILABLE:

    # Claim Verifier Agent
    claim_verifier_agent = Agent(
        model=AnthropicModel(DEFAULT_MODEL),  # Uses ANTHROPIC_API_KEY env var
        deps_type=VerificationContext,
        result_type=ClaimVerificationResult,
        system_prompt="""You are an expert fact-checker and claim verification specialist.
Your role is to carefully analyze claims and determine their veracity.

When verifying a claim:
1. First, search the knowledge graph for relevant information
2. Check the credibility of sources mentioned
3. Look for related claims that have been verified
4. Consider multiple perspectives and potential biases
5. Provide a clear verdict with supporting reasoning

Be thorough but also acknowledge limitations in your verification.
Always explain your reasoning clearly.""",
    )

    # Register tools with the agent
    claim_verifier_agent.tool(search_knowledge_graph)
    claim_verifier_agent.tool(check_source_credibility)
    claim_verifier_agent.tool(find_related_claims)
    claim_verifier_agent.tool(validate_security)

    # Source Evaluator Agent
    source_evaluator_agent = Agent(
        model=AnthropicModel(DEFAULT_MODEL),  # Uses ANTHROPIC_API_KEY env var
        deps_type=SourceContext,
        result_type=SourceEvaluationResult,
        system_prompt="""You are an expert media analyst specializing in source credibility assessment.
Your role is to evaluate news sources and information providers for their reliability.

When evaluating a source:
1. Consider the source's history and track record
2. Identify any known biases or editorial slants
3. Assess transparency and accountability measures
4. Consider funding sources and ownership
5. Look at fact-checking history if available

Provide nuanced assessments that acknowledge complexity.""",
    )

    # Evidence Gatherer Agent
    evidence_gatherer_agent = Agent(
        model=AnthropicModel(DEFAULT_MODEL),  # Uses ANTHROPIC_API_KEY env var
        deps_type=EvidenceContext,
        result_type=EvidenceGatheringResult,
        system_prompt="""You are an expert research analyst specializing in evidence gathering.
Your role is to find and organize evidence related to claims.

When gathering evidence:
1. Search multiple sources for relevant information
2. Classify evidence as supporting, contradicting, or neutral
3. Assess the relevance of each piece of evidence
4. Evaluate the credibility of sources
5. Organize findings in a clear, structured manner

Be thorough and consider multiple perspectives.""",
    )


# =============================================================================
# RUNNER FUNCTIONS
# =============================================================================


async def run_claim_verification(
    claim: str,
    context: Optional[VerificationContext] = None,
) -> Optional[ClaimVerificationResult]:
    """Run claim verification with the Pydantic-AI agent.

    Args:
        claim: The claim to verify
        context: Optional context with dependencies

    Returns:
        ClaimVerificationResult or None if unavailable
    """
    if not PYDANTIC_AI_AVAILABLE:
        logger.warning("Pydantic-AI not available for claim verification")
        return None

    try:
        ctx = context or VerificationContext(claim_id="auto")
        result = await claim_verifier_agent.run(
            f"Verify this claim: {claim}",
            deps=ctx,
        )
        return result.data
    except Exception as e:
        logger.error(f"Claim verification failed: {e}")
        return None


async def run_source_evaluation(
    source_name: str,
    source_url: Optional[str] = None,
    context: Optional[SourceContext] = None,
) -> Optional[SourceEvaluationResult]:
    """Run source evaluation with the Pydantic-AI agent.

    Args:
        source_name: Name of the source to evaluate
        source_url: Optional URL of the source
        context: Optional context with dependencies

    Returns:
        SourceEvaluationResult or None if unavailable
    """
    if not PYDANTIC_AI_AVAILABLE:
        logger.warning("Pydantic-AI not available for source evaluation")
        return None

    try:
        ctx = context or SourceContext(
            source_id="auto",
            source_name=source_name,
            source_url=source_url,
        )
        result = await source_evaluator_agent.run(
            f"Evaluate the credibility of this source: {source_name}"
            + (f" ({source_url})" if source_url else ""),
            deps=ctx,
        )
        return result.data
    except Exception as e:
        logger.error(f"Source evaluation failed: {e}")
        return None


async def run_evidence_gathering(
    claim: str,
    context: Optional[EvidenceContext] = None,
) -> Optional[EvidenceGatheringResult]:
    """Run evidence gathering with the Pydantic-AI agent.

    Args:
        claim: The claim to gather evidence for
        context: Optional context with dependencies

    Returns:
        EvidenceGatheringResult or None if unavailable
    """
    if not PYDANTIC_AI_AVAILABLE:
        logger.warning("Pydantic-AI not available for evidence gathering")
        return None

    try:
        ctx = context or EvidenceContext(claim_text=claim)
        result = await evidence_gatherer_agent.run(
            f"Gather evidence for this claim: {claim}",
            deps=ctx,
        )
        return result.data
    except Exception as e:
        logger.error(f"Evidence gathering failed: {e}")
        return None


# =============================================================================
# TYPE-SAFE AGENT COMMUNICATION
# =============================================================================


class AgentMessage(BaseModel):
    """Type-safe message between agents."""

    sender: str = Field(..., description="Sending agent ID")
    receiver: str = Field(..., description="Receiving agent ID")
    message_type: str = Field(..., description="Type of message")
    payload: Dict[str, Any] = Field(default_factory=dict)
    correlation_id: str = Field(default="", description="For request-response patterns")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AgentResponse(BaseModel):
    """Type-safe response from an agent."""

    request_id: str = Field(..., description="Original request ID")
    status: str = Field(..., description="success, error, pending")
    result: Optional[Any] = Field(None, description="Result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: float = Field(default=0.0)


class AgentCapability(BaseModel):
    """Describes an agent's capabilities."""

    agent_id: str = Field(..., description="Agent identifier")
    capabilities: List[str] = Field(default_factory=list)
    input_types: List[str] = Field(default_factory=list)
    output_types: List[str] = Field(default_factory=list)
    max_concurrent_requests: int = Field(default=5)


class MultiAgentContext(BaseModel):
    """Context for multi-agent coordination."""

    task_id: str = Field(..., description="Task identifier")
    participating_agents: List[str] = Field(default_factory=list)
    shared_state: Dict[str, Any] = Field(default_factory=dict)
    conversation_history: List[AgentMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AgentOrchestrator:
    """Orchestrator for type-safe multi-agent communication.

    Provides:
    - Type-safe message passing between agents
    - Request-response patterns
    - Agent capability discovery
    - Conversation state management
    """

    def __init__(self):
        self._agents: Dict[str, Any] = {}
        self._capabilities: Dict[str, AgentCapability] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._contexts: Dict[str, MultiAgentContext] = {}

        if PYDANTIC_AI_AVAILABLE:
            self._register_builtin_agents()

    def _register_builtin_agents(self):
        """Register built-in agents."""
        self.register_agent(
            "claim_verifier",
            claim_verifier_agent,
            AgentCapability(
                agent_id="claim_verifier",
                capabilities=["verify_claim", "assess_evidence"],
                input_types=["ClaimVerificationRequest"],
                output_types=["ClaimVerificationResult"],
            ),
        )
        self.register_agent(
            "source_evaluator",
            source_evaluator_agent,
            AgentCapability(
                agent_id="source_evaluator",
                capabilities=["evaluate_source", "assess_credibility"],
                input_types=["SourceEvaluationRequest"],
                output_types=["SourceEvaluationResult"],
            ),
        )
        self.register_agent(
            "evidence_gatherer",
            evidence_gatherer_agent,
            AgentCapability(
                agent_id="evidence_gatherer",
                capabilities=["gather_evidence", "search_sources"],
                input_types=["EvidenceGatheringRequest"],
                output_types=["EvidenceGatheringResult"],
            ),
        )

    def register_agent(
        self,
        agent_id: str,
        agent: Any,
        capability: AgentCapability,
    ) -> None:
        """Register an agent with capabilities."""
        self._agents[agent_id] = agent
        self._capabilities[agent_id] = capability
        logger.info(f"Registered agent: {agent_id}")

    def get_agent_capabilities(self, agent_id: str) -> Optional[AgentCapability]:
        """Get capabilities for an agent."""
        return self._capabilities.get(agent_id)

    def find_agents_for_capability(self, capability: str) -> List[str]:
        """Find agents that have a specific capability."""
        return [
            agent_id
            for agent_id, cap in self._capabilities.items()
            if capability in cap.capabilities
        ]

    async def send_message(
        self,
        message: AgentMessage,
        timeout: float = 30.0,
    ) -> AgentResponse:
        """Send a type-safe message to an agent."""
        import uuid

        start_time = datetime.now(timezone.utc)
        request_id = message.correlation_id or str(uuid.uuid4())[:8]

        if message.receiver not in self._agents:
            return AgentResponse(
                request_id=request_id,
                status="error",
                error=f"Unknown agent: {message.receiver}",
            )

        try:
            agent = self._agents[message.receiver]
            prompt = f"[From: {message.sender}] {message.message_type}: {message.payload}"

            # Determine context type
            if message.receiver == "claim_verifier":
                ctx = VerificationContext(claim_id=message.payload.get("claim_id", "auto"))
            elif message.receiver == "source_evaluator":
                ctx = SourceContext(source_id=message.payload.get("source_id", "auto"))
            elif message.receiver == "evidence_gatherer":
                ctx = EvidenceContext(claim_text=message.payload.get("claim_text", ""))
            else:
                ctx = None

            result = await asyncio.wait_for(
                agent.run(prompt, deps=ctx),
                timeout=timeout,
            )

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return AgentResponse(
                request_id=request_id,
                status="success",
                result=result.data.model_dump() if hasattr(result.data, 'model_dump') else result.data,
                processing_time_ms=processing_time,
            )

        except asyncio.TimeoutError:
            return AgentResponse(request_id=request_id, status="error", error=f"Timeout after {timeout}s")
        except Exception as e:
            logger.error(f"Agent message failed: {e}")
            return AgentResponse(request_id=request_id, status="error", error=str(e))

    def create_context(self, task_id: str, agents: List[str]) -> MultiAgentContext:
        """Create a new multi-agent context."""
        context = MultiAgentContext(task_id=task_id, participating_agents=agents)
        self._contexts[task_id] = context
        return context

    def get_context(self, task_id: str) -> Optional[MultiAgentContext]:
        """Get context by task ID."""
        return self._contexts.get(task_id)

    async def coordinate_verification(
        self,
        claim: str,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Coordinate multiple agents for comprehensive claim verification."""
        import uuid

        task_id = task_id or str(uuid.uuid4())[:8]
        context = self.create_context(
            task_id, ["evidence_gatherer", "source_evaluator", "claim_verifier"]
        )

        results = {}
        start_time = datetime.now(timezone.utc)

        # Step 1: Gather evidence
        evidence_msg = AgentMessage(
            sender="orchestrator",
            receiver="evidence_gatherer",
            message_type="gather_evidence",
            payload={"claim_text": claim},
            correlation_id=f"{task_id}_evidence",
        )
        evidence_response = await self.send_message(evidence_msg)
        results["evidence"] = evidence_response

        if evidence_response.status != "success":
            return {"task_id": task_id, "status": "failed", "error": "Evidence gathering failed", "results": results}

        # Step 2: Evaluate sources in parallel
        sources = evidence_response.result.get("search_sources", [])[:3] if evidence_response.result else []
        source_tasks = []
        for source in sources:
            source_msg = AgentMessage(
                sender="orchestrator",
                receiver="source_evaluator",
                message_type="evaluate_source",
                payload={"source_name": source, "source_id": source},
                correlation_id=f"{task_id}_source_{source[:8]}",
            )
            source_tasks.append(self.send_message(source_msg))

        if source_tasks:
            source_responses = await asyncio.gather(*source_tasks, return_exceptions=True)
            results["source_evaluations"] = [
                r.model_dump() if isinstance(r, AgentResponse) else {"error": str(r)}
                for r in source_responses
            ]

        # Step 3: Final verification
        verify_msg = AgentMessage(
            sender="orchestrator",
            receiver="claim_verifier",
            message_type="verify_claim",
            payload={"claim_id": task_id, "claim_text": claim, "evidence": evidence_response.result},
            correlation_id=f"{task_id}_verify",
        )
        verify_response = await self.send_message(verify_msg)
        results["verification"] = verify_response

        total_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return {
            "task_id": task_id,
            "status": "success" if verify_response.status == "success" else "partial",
            "claim": claim,
            "results": results,
            "total_time_ms": total_time,
            "agents_used": context.participating_agents,
        }


# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_agent_orchestrator() -> AgentOrchestrator:
    """Get singleton agent orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


async def coordinate_claim_verification(claim: str) -> Optional[Dict[str, Any]]:
    """Convenience function for coordinated claim verification."""
    if not PYDANTIC_AI_AVAILABLE:
        logger.warning("Pydantic-AI not available for coordinated verification")
        return None

    try:
        orchestrator = get_agent_orchestrator()
        return await orchestrator.coordinate_verification(claim)
    except Exception as e:
        logger.error(f"Coordinated verification failed: {e}")
        return None


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "PYDANTIC_AI_AVAILABLE",
    # Contexts
    "VerificationContext",
    "SourceContext",
    "EvidenceContext",
    # Results
    "ClaimVerificationResult",
    "SourceEvaluationResult",
    "EvidenceGatheringResult",
    "EvidenceItem",
    "VerificationVerdict",
    "CredibilityLevel",
    "EvidenceType",
    # Agent Communication
    "AgentMessage",
    "AgentResponse",
    "AgentCapability",
    "MultiAgentContext",
    "AgentOrchestrator",
    "get_agent_orchestrator",
    # Runner functions
    "run_claim_verification",
    "run_source_evaluation",
    "run_evidence_gathering",
    "coordinate_claim_verification",
]

# Conditional exports for agents and tools
if PYDANTIC_AI_AVAILABLE:
    __all__.extend([
        # Agents
        "claim_verifier_agent",
        "source_evaluator_agent",
        "evidence_gatherer_agent",
        # Tools
        "search_knowledge_graph",
        "check_source_credibility",
        "find_related_claims",
        "validate_security",
    ])
