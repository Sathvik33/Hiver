import pytest
from app.services.escalation_service import EscalationEngine
from app.services.grounding_service import GroundingGuardrail

def test_escalate_on_human_request():
    decision = EscalationEngine.evaluate(
        customer_message="Let me speak to a human representative right now!",
        intent_label="human_agent_request",
        intent_confidence=0.95,
        top_retrieval_similarity=0.85,
        retrieved_evidence=[{"resolution": "test"}],
        grounding_violations=[]
    )
    assert decision.decision == "ESCALATE"
    assert "human" in decision.reason.lower()

def test_escalate_on_fraud_security_keywords():
    decision = EscalationEngine.evaluate(
        customer_message="Someone hacked my account and made fraudulent purchases!",
        intent_label="account_and_login",
        intent_confidence=0.92,
        top_retrieval_similarity=0.88,
        retrieved_evidence=[{"resolution": "Please visit account recovery."}],
        grounding_violations=[]
    )
    assert decision.decision == "ESCALATE"
    assert "security" in decision.reason.lower() or "fraud" in decision.reason.lower()

def test_escalate_on_legal_threat():
    decision = EscalationEngine.evaluate(
        customer_message="I have spoken to my lawyer and we are preparing a lawsuit.",
        intent_label="general_inquiry_or_other",
        intent_confidence=0.90,
        top_retrieval_similarity=0.75,
        retrieved_evidence=[{"resolution": "Contact legal team."}],
        grounding_violations=[]
    )
    assert decision.decision == "ESCALATE"

def test_escalate_on_low_confidence():
    decision = EscalationEngine.evaluate(
        customer_message="Where is my package?",
        intent_label="delivery_delay",
        intent_confidence=0.50,  # below 0.65 threshold
        top_retrieval_similarity=0.85,
        retrieved_evidence=[{"resolution": "test"}],
        grounding_violations=[]
    )
    assert decision.decision == "ESCALATE"
    assert "confidence" in decision.reason.lower()

def test_escalate_on_weak_retrieval():
    decision = EscalationEngine.evaluate(
        customer_message="Where is my order?",
        intent_label="delivery_delay",
        intent_confidence=0.90,
        top_retrieval_similarity=0.30,  # below 0.60 threshold
        retrieved_evidence=[],
        grounding_violations=[]
    )
    assert decision.decision == "ESCALATE"
    assert "similarity" in decision.reason.lower()

def test_escalate_on_grounding_violations():
    decision = EscalationEngine.evaluate(
        customer_message="Where is my refund?",
        intent_label="return_and_refund",
        intent_confidence=0.91,
        top_retrieval_similarity=0.78,
        retrieved_evidence=[{"resolution": "Check returns page."}],
        grounding_violations=["Invented money amount not in evidence: $500"]
    )
    assert decision.decision == "ESCALATE"
    assert decision.is_safe is False
    assert "grounding violations" in decision.reason.lower()

def test_auto_handle_when_grounded_and_confident():
    decision = EscalationEngine.evaluate(
        customer_message="My tracking number hasn't updated for two days.",
        intent_label="delivery_delay",
        intent_confidence=0.89,
        top_retrieval_similarity=0.82,
        retrieved_evidence=[{"customer_problem": "tracking delay", "resolution": "Check amzn.to/track"}],
        grounding_violations=[]
    )
    assert decision.decision == "AUTO_HANDLE"
    assert decision.is_safe is True

def test_grounding_guardrail_unauthorized_url():
    reply = "Please click here to get help: http://phishing-site.ru/refund"
    is_grounded, violations = GroundingGuardrail.validate(reply, [])
    assert not is_grounded
    assert any("unauthorized" in v.lower() for v in violations)

def test_grounding_guardrail_authorized_url():
    reply = "Please check your delivery status here: https://amzn.to/track"
    evidence = [{"customer_problem": "delayed package", "resolution": "check amzn.to/track"}]
    is_grounded, violations = GroundingGuardrail.validate(reply, evidence)
    assert is_grounded
    assert len(violations) == 0

def test_grounding_guardrail_invented_amounts():
    reply = "We have refunded $500 dollars to your bank account."
    evidence = [{"customer_problem": "delayed package", "resolution": "check tracking"}]
    is_grounded, violations = GroundingGuardrail.validate(reply, evidence)
    assert not is_grounded
    assert any("amount" in v.lower() for v in violations)
