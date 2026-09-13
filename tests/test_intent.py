import pytest
import asyncio
from app.llm.base import IntentPrediction
from app.llm.factory import get_llm_provider

TAXONOMY = [
    "delivery_delay",
    "return_and_refund",
    "order_cancellation",
    "damaged_or_defective",
    "wrong_item_received",
    "payment_and_billing",
    "account_and_login",
    "subscription_and_prime",
    "human_agent_request",
    "general_inquiry_or_other"
]

@pytest.mark.asyncio
async def test_classify_valid_delivery_intent():
    llm = get_llm_provider()
    msg = "My order was supposed to arrive yesterday and tracking hasn't updated. Where is it?"
    res = await llm.classify_intent(msg)
    assert isinstance(res, IntentPrediction)
    assert res.intent in TAXONOMY
    assert res.intent == "delivery_delay"
    assert res.confidence >= 0.50

@pytest.mark.asyncio
async def test_classify_human_agent_request():
    llm = get_llm_provider()
    msg = "I need to talk to an actual human representative, not this bot!"
    res = await llm.classify_intent(msg)
    assert isinstance(res, IntentPrediction)
    assert res.intent == "human_agent_request"
    assert res.confidence >= 0.70

@pytest.mark.asyncio
async def test_classify_ambiguous_short_message():
    llm = get_llm_provider()
    msg = "help"
    res = await llm.classify_intent(msg)
    assert isinstance(res, IntentPrediction)
    assert res.intent in TAXONOMY
    # Ambiguous message should either be classified as general_inquiry_or_other or low confidence
    assert res.intent == "general_inquiry_or_other" or res.confidence < 0.70
