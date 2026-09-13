from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class AgentRespondRequest(BaseModel):
    message: str = Field(description="Customer incoming message")
    context: Optional[str] = Field(default=None, description="Optional previous conversation context")

class EvidenceItem(BaseModel):
    conversation_id: str
    similarity: float
    customer_problem: str
    resolution: str
    quality: str

class AgentRespondResponse(BaseModel):
    request_id: str
    intent: str
    intent_confidence: float
    reply: str
    decision: str  # "AUTO_HANDLE" or "ESCALATE"
    reason: Optional[str] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    top_similarity: float
    latency_ms: float

class RetrievalSearchRequest(BaseModel):
    query: str
    top_k: int = 4

class RetrievalSearchResponse(BaseModel):
    query: str
    top_similarity: float
    method: str
    results: List[Dict[str, Any]]
