import json
import yaml
from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.schemas.agent import (
    AgentRespondRequest,
    AgentRespondResponse,
    RetrievalSearchRequest,
    RetrievalSearchResponse
)
from app.services.agent_service import handle_customer_message
from app.services.retrieval_service import get_retriever

app = FastAPI(
    title="Hiver AI Customer Support Agent API",
    description="Evaluation-first, retrieval-grounded customer support agent for @AmazonHelp",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "brand": settings.SELECTED_BRAND,
        "llm_provider": settings.LLM_PROVIDER,
        "model": settings.OLLAMA_MODEL if settings.LLM_PROVIDER == "ollama" else settings.GROQ_MODEL
    }

@app.get("/api/brands")
def get_brands():
    return {
        "selected_brand": settings.SELECTED_BRAND,
        "supported_brands": [
            {"name": "Amazon", "handle": "AmazonHelp", "selected": True},
            {"name": "Apple Support", "handle": "AppleSupport", "selected": False},
            {"name": "Uber Support", "handle": "Uber_Support", "selected": False},
            {"name": "Spotify Cares", "handle": "SpotifyCares", "selected": False}
        ]
    }

@app.get("/api/intents")
def get_intents():
    taxonomy_file = PROJECT_ROOT / "data" / "processed" / "intent_taxonomy.yaml"
    if taxonomy_file.exists():
        with open(taxonomy_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data
    raise HTTPException(status_code=404, detail="Intent taxonomy not found")

@app.post("/api/agent/respond", response_model=AgentRespondResponse)
async def respond_to_customer(req: AgentRespondRequest):
    result = await handle_customer_message(message=req.message, context=req.context)
    return AgentRespondResponse(
        request_id=result.request_id,
        intent=result.intent.label,
        intent_confidence=result.intent.confidence,
        reply=result.response,
        decision=result.decision,
        reason=result.escalation_reason,
        evidence=result.retrieval.results,
        top_similarity=result.retrieval.top_similarity,
        latency_ms=result.latency_ms
    )

@app.post("/api/retrieval/search", response_model=RetrievalSearchResponse)
def search_retrieval(req: RetrievalSearchRequest):
    retriever = get_retriever()
    out = retriever.search(req.query, top_k=req.top_k)
    return RetrievalSearchResponse(
        query=req.query,
        top_similarity=out["top_similarity"],
        method=out["method"],
        results=out["results"]
    )

@app.get("/api/evaluation/summary")
def get_evaluation_summary():
    report_file = PROJECT_ROOT / "reports" / "evaluation.json"
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "status": "pending",
        "message": "Benchmark not executed yet. Run python -m evaluation.run to populate."
    }

@app.get("/api/evaluation/results")
def get_evaluation_results():
    golden_file = PROJECT_ROOT / "data" / "golden" / "golden_set.json"
    failures_file = PROJECT_ROOT / "reports" / "failure_examples.json"
    
    golden_data = []
    if golden_file.exists():
        with open(golden_file, "r", encoding="utf-8") as f:
            golden_data = json.load(f)
            
    failures = []
    if failures_file.exists():
        with open(failures_file, "r", encoding="utf-8") as f:
            failures = json.load(f)
            
    return {
        "golden_examples_count": len(golden_data),
        "failures": failures,
        "sample_golden": golden_data[:10]
    }
