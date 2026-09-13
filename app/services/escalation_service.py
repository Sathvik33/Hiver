from typing import Optional, List, Dict, Any
import re
from pydantic import BaseModel
from app.core.config import settings

class EscalationDecision(BaseModel):
    decision: str  # "AUTO_HANDLE" or "ESCALATE"
    reason: Optional[str] = None
    is_safe: bool = True

HUMAN_REQUEST_KEYWORDS = [
    "human", "agent", "person", "representative", "supervisor", "manager",
    "real person", "stop bot", "actual human", "speak to someone", "talk to a person"
]

EMERGENCY_KEYWORDS = [
    "hacked", "stolen", "fraud", "lawyer", "police", "burned", "injury", "lawsuit", "freeze",
    "unauthorized", "compromised", "fire", "smoke", "accident", "counterfeit", "prescription"
]

def check_multi_intent(message: str) -> bool:
    """
    Detects if message combines multiple support intents or disparate workflows.
    e.g., 'cancel order and also return', 'late delivery and double charged'
    """
    lower = message.lower()
    conjunction_patterns = ["and also", "as well as", "plus i also", "and i was also", "and my card"]
    if any(cp in lower for cp in conjunction_patterns):
        return True
    
    # Check for distinct intent keyword co-occurrence
    has_cancel = any(w in lower for w in ["cancel", "cancellation"])
    has_return = any(w in lower for w in ["return", "refund"])
    has_delay = any(w in lower for w in ["late", "delayed", "where is my"])
    has_billing = any(w in lower for w in ["charged twice", "double charge", "billed"])
    has_damaged = any(w in lower for w in ["damaged", "broken", "cracked", "crushed"])
    
    active_intents = sum([has_cancel, has_return, has_delay, has_billing, has_damaged])
    return active_intents >= 2

def check_insufficient_context(message: str, intent_label: str) -> bool:
    """
    Detects vague, underspecified refund, cancellation, or status requests that lack
    order numbers, tracking IDs, or product details (e.g. 'Can you refund me for the bad purchase?')
    """
    lower = message.lower().strip()
    words = lower.split()
    
    # Extremely short inquiries
    if len(words) <= 3:
        return True
        
    # Vague refund requests with no item, order ID, or tracking info
    if "refund" in lower:
        has_identifier = bool(re.search(r'\b\d{3}[-\s]?\d{7}[-\s]?\d{7}\b|\b#?\d{4,}\b', lower))
        has_specific_product = any(w in lower for w in ["shoes", "phone", "book", "tv", "laptop", "shirt", "charger"])
        if "bad purchase" in lower or "the purchase" in lower or "my purchase" in lower or "that thing" in lower:
            if not has_identifier and not has_specific_product:
                return True

    return False

class EscalationEngine:
    """
    Conservative, evidence-based escalation engine implementing clear decision rules:
    ESCALATE if:
    - Intent == 'human_agent_request' or message contains explicit human request
    - Critical safety / fraud / legal / injury emergency keywords detected
    - Multi-intent customer message detected
    - Insufficient context / underspecified request
    - Intent == 'general_inquiry_or_other' (ambiguous or OOD query)
    - Classifier confidence < INTENT_CONFIDENCE_THRESHOLD (default 0.65)
    - Retrieval similarity < RETRIEVAL_SIMILARITY_THRESHOLD (default 0.60)
    - No useful historical resolution exists
    - Grounding validation failed (contains invented claims / unsupported URLs)
    Otherwise:
    - AUTO_HANDLE
    """

    @staticmethod
    def evaluate(
        customer_message: str,
        intent_label: str,
        intent_confidence: float,
        top_retrieval_similarity: float,
        retrieved_evidence: List[Dict[str, Any]],
        grounding_violations: List[str]
    ) -> EscalationDecision:
        lower_msg = customer_message.lower()

        # Rule 1: Explicit customer demand for human assistance
        if intent_label == "human_agent_request" or any(kw in lower_msg for kw in HUMAN_REQUEST_KEYWORDS):
            return EscalationDecision(
                decision="ESCALATE",
                reason="Customer explicitly requested human agent or supervisor.",
                is_safe=True
            )

        # Rule 2: Critical safety / fraud / legal / injury emergency
        if any(ek in lower_msg for ek in EMERGENCY_KEYWORDS):
            return EscalationDecision(
                decision="ESCALATE",
                reason="Customer inquiry involves security breach, financial fraud, physical safety hazard, or legal liability.",
                is_safe=True
            )

        # Rule 3: Multi-intent customer message
        if check_multi_intent(customer_message):
            return EscalationDecision(
                decision="ESCALATE",
                reason="Customer message combines multiple distinct support intents requiring complex agent routing.",
                is_safe=True
            )

        # Rule 4: Insufficient context or underspecified entity
        if check_insufficient_context(customer_message, intent_label):
            return EscalationDecision(
                decision="ESCALATE",
                reason="Customer inquiry is underspecified or missing order/item details required for resolution.",
                is_safe=True
            )

        # Rule 5: Ambiguous or Out-Of-Distribution intent
        if intent_label == "general_inquiry_or_other":
            return EscalationDecision(
                decision="ESCALATE",
                reason="Inquiry is out-of-distribution or intent is ambiguous without clear customer context.",
                is_safe=True
            )

        # Rule 6: Low intent classifier confidence
        if intent_confidence < settings.INTENT_CONFIDENCE_THRESHOLD:
            return EscalationDecision(
                decision="ESCALATE",
                reason=f"Classifier confidence ({intent_confidence:.2f}) is below threshold ({settings.INTENT_CONFIDENCE_THRESHOLD}).",
                is_safe=True
            )

        # Rule 7: Weak retrieval evidence
        if not retrieved_evidence or top_retrieval_similarity < settings.RETRIEVAL_SIMILARITY_THRESHOLD:
            return EscalationDecision(
                decision="ESCALATE",
                reason=f"No sufficiently similar historical resolution was retrieved (top similarity: {top_retrieval_similarity:.2f} < {settings.RETRIEVAL_SIMILARITY_THRESHOLD}).",
                is_safe=True
            )

        # Rule 8: Grounding violations detected in generated reply
        if grounding_violations:
            return EscalationDecision(
                decision="ESCALATE",
                reason=f"Generated reply contained unsupported claims or grounding violations: {'; '.join(grounding_violations)}",
                is_safe=False
            )

        # All conditions satisfied for safe automated assistance
        return EscalationDecision(
            decision="AUTO_HANDLE",
            reason="High intent confidence and strong grounded historical resolution retrieved.",
            is_safe=True
        )
