from typing import List, Dict, Any

def compute_retrieval_metrics(
    eval_queries: List[str],
    gold_intents: List[str],
    retriever_instance,
    k_list: List[int] = [1, 3, 5]
) -> Dict[str, Any]:
    """
    Evaluates retrieval quality:
    - Recall@1
    - Recall@3
    - Recall@5
    - MRR (Mean Reciprocal Rank)
    A retrieved example is considered relevant if its historical intent matches the query's gold intent.
    """
    recall_hits = {k: 0 for k in k_list}
    reciprocal_ranks = []

    for query, gold_intent in zip(eval_queries, gold_intents):
        search_res = retriever_instance.search(query, top_k=max(k_list))
        retrieved_items = search_res.get("results", [])

        # Find position of first relevant item
        rr = 0.0
        for rank, item in enumerate(retrieved_items, 1):
            if item.get("intent") == gold_intent:
                if rr == 0.0:
                    rr = 1.0 / rank
                for k in k_list:
                    if rank <= k:
                        recall_hits[k] += 1
                break  # count hit once per query
        reciprocal_ranks.append(rr)

    n = max(len(eval_queries), 1)
    results = {
        f"recall_at_{k}": round(recall_hits[k] / n, 4) for k in k_list
    }
    results["mrr"] = round(sum(reciprocal_ranks) / n, 4)
    results["total_queries_evaluated"] = n
    return results
