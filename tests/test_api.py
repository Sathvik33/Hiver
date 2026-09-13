import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["brand"] == "AmazonHelp"

def test_brands_endpoint():
    resp = client.get("/api/brands")
    assert resp.status_code == 200
    data = resp.json()
    assert data["selected_brand"] == "AmazonHelp"

def test_intents_endpoint():
    resp = client.get("/api/intents")
    assert resp.status_code == 200
    data = resp.json()
    assert "intents" in data
    assert len(data["intents"]) >= 8

def test_retrieval_search_api():
    resp = client.post("/api/retrieval/search", json={"query": "package delayed", "top_k": 2})
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) <= 2
