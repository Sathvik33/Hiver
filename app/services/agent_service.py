import uuid
import time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from app.core.logging import logger
from app.llm.factory import get_llm_provider
from app.services.retrieval_service import get_retriever
from app.services.grounding_service import GroundingGuardrail
from app.services.escalation_service import EscalationEngine, EscalationDecision

class IntentResult(BaseModel):
    label: str
    confidence: float
    reason: Optional[str] = None

class RetrievalResult(BaseModel):
    results: List[Dict[str, Any]]
    top_similarity: float
    method: str

class AgentResult(BaseModel):
    request_id: str
    intent: IntentResult
    retrieval: RetrievalResult
    response: str
    decision: str  # "AUTO_HANDLE" or "ESCALATE"
    escalation_reason: Optional[str] = None
    evidence_ids: List[str] = Field(default_factory=list)
    grounding_violations: List[str] = Field(default_factory=list)
    latency_ms: float

async def handle_customer_message(
    message: str,
    context: Optional[str] = None,
    override_provider = None
) -> AgentResult:
    start_time = time.time()
    req_id = f"req_{uuid.uuid4().hex[:10]}"
    
    # 1. Normalization
    normalized_msg = message.strip()
    
    # 2. Intent Classification
    llm = override_provider or get_llm_provider()
    intent_pred = await llm.classify_intent(normalized_msg, context)
    
    # 3. Retrieval
    retriever = get_retriever()
    retrieval_out = retriever.search(normalized_msg, top_k=4)
    
    # 4. Reply Generation (grounded on top retrieved resolutions)
    gen_result = await llm.generate_grounded_reply(
        customer_message=normalized_msg,
        context=context,
        intent=intent_pred.intent,
        retrieved_evidence=retrieval_out["results"]
    )
    
    # 5. Grounding Guardrails
    is_grounded, violations = GroundingGuardrail.validate(
        gen_result.reply,
        retrieval_out["results"]
    )
    
    # 6. Escalation Policy Evaluation
    escalation_decision = EscalationEngine.evaluate(
        customer_message=normalized_msg,
        intent_label=intent_pred.intent,
        intent_confidence=intent_pred.confidence,
        top_retrieval_similarity=retrieval_out["top_similarity"],
        retrieved_evidence=retrieval_out["results"],
        grounding_violations=violations
    )
    
    # If escalated, ensure response guides user cleanly to human touchpoint
    final_reply = gen_result.reply
    if escalation_decision.decision == "ESCALATE":
        if "human_agent_request" in intent_pred.intent or "human" in normalized_msg.lower():
            final_reply = "We'd be glad to connect you with a team member! Please send us a direct message with your order number so an agent can assist you right away."
        elif violations:
            final_reply = "Thanks for reaching out! To ensure accurate account assistance, please send us a DM with your order details so our specialized team can look into this."

    latency = round((time.time() - start_time) * 1000, 2)
    
    logger.info(
        f"[{req_id}] Intent={intent_pred.intent} ({intent_pred.confidence:.2f}) | "
        f"TopSim={retrieval_out['top_similarity']:.2f} | Decision={escalation_decision.decision} | "
        f"Latency={latency}ms"
    )
    
    return AgentResult(
        request_id=req_id,
        intent=IntentResult(
            label=intent_pred.intent,
            confidence=intent_pred.confidence,
            reason=intent_pred.reason
        ),
        retrieval=RetrievalResult(
            results=retrieval_out["results"],
            top_similarity=retrieval_out["top_similarity"],
            method=retrieval_out["method"]
        ),
        response=final_reply,
        decision=escalation_decision.decision,
        escalation_reason=escalation_decision.reason,
        evidence_ids=gen_result.evidence_ids,
        grounding_violations=violations,
        latency_ms=latency
    )
