from typing import Protocol, Dict, Any, Optional
from pydantic import BaseModel, Field

class IntentPrediction(BaseModel):
    intent: str = Field(description="The predicted intent identifier")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    reason: str = Field(description="Brief rationale for the classification")

class GeneratedReply(BaseModel):
    reply: str = Field(description="Grounded support response")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    evidence_ids: list[str] = Field(default_factory=list, description="IDs of historical conversations used as grounding")
    supported_claims: list[str] = Field(default_factory=list, description="List of claims supported by evidence")

class LLMJudgeEvaluation(BaseModel):
    correctness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    grounding: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    style: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)
    overall: float = Field(ge=1.0, le=5.0)
    reason: str

class LLMProvider(Protocol):
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        ...

    async def classify_intent(self, message: str, context: Optional[str] = None) -> IntentPrediction:
        ...

    async def generate_grounded_reply(
        self,
        customer_message: str,
        context: Optional[str],
        intent: str,
        retrieved_evidence: list[dict]
    ) -> GeneratedReply:
        ...

    async def judge_reply(
        self,
        customer_message: str,
        retrieved_evidence: list[dict],
        generated_reply: str
    ) -> LLMJudgeEvaluation:
        ...
