import pytest
from app.services.retrieval_service import get_retriever

def test_retrieval_ranking():
    retriever = get_retriever()
    query = "Where is my package? It was supposed to be delivered yesterday!"
    out = retriever.search(query, top_k=3)
    assert "results" in out
    assert len(out["results"]) > 0
    assert out["top_similarity"] > 0.0
    # Top result should be relevant to delivery
    top_res = out["results"][0]
    assert "resolution" in top_res
    assert len(top_res["resolution"]) > 0

def test_retrieval_empty_query():
    retriever = get_retriever()
    out = retriever.search("", top_k=3)
    assert "results" in out
    assert isinstance(out["results"], list)

def test_retrieval_nonsense_query():
    retriever = get_retriever()
    query = "xyzqwerty foobar nonsense query 9999"
    out = retriever.search(query, top_k=3)
    assert "results" in out

def test_retrieval_top_k_behavior():
    retriever = get_retriever()
    query = "Can I return an opened item?"
    out_1 = retriever.search(query, top_k=1)
    out_5 = retriever.search(query, top_k=5)
    assert len(out_1["results"]) <= 1
    assert len(out_5["results"]) <= 5
    if len(out_5["results"]) > 0:
        assert out_1["results"][0]["conversation_id"] == out_5["results"][0]["conversation_id"]
