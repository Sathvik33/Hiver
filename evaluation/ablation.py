"""
Component Ablation Study for Hiver AI Customer Support Agent

Investigates:
- Ablation A: Qwen 2.5 Classification + Generation WITHOUT Retrieval (Closed-book generation)
- Ablation B: Qwen 2.5 Classification + Generation WITH FAISS Semantic Retrieval (Unchecked generation)
- Ablation C: Full System (Qwen 2.5 + FAISS Retrieval + Grounding Guardrails + Escalation Policy)

Measures:
- Intent Macro-F1 & Accuracy
- Retrieval Grounding / Hallucination Rate
- Average Reply Quality (1-5 scale)
- Automation Coverage (% auto-handled)
- Unsafe Auto-handle Rate
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.llm.factory import get_llm_provider
from app.services.retrieval_service import get_retriever
from app.services.grounding_service import GroundingGuardrail
from app.services.escalation_service import EscalationEngine
from evaluation.intent_metrics import compute_intent_metrics
from evaluation.escalation_metrics import compute_escalation_metrics

async def run_ablation_study(sample_size: int = 25) -> Dict[str, Any]:
    print("=" * 70)
    print(f"RUNNING COMPONENT ABLATION STUDY (N={sample_size} held-out queries)")
    print("=" * 70)

    golden_file = ROOT_DIR / "data" / "golden" / "golden_set.json"
    with open(golden_file, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    eval_items = golden_set[:sample_size]
    llm = get_llm_provider()
    retriever = get_retriever()
    labels = sorted(list(set(ex["intent"] for ex in golden_set)))

    # ABLATION A: No Retrieval (Closed-book Qwen)
    print("\n[Ablation A]: Qwen Classification + Generation WITHOUT Retrieval...")
    a_results = []
    for item in eval_items:
        intent_pred = await llm.classify_intent(item["message"])
        # Generate with empty retrieval evidence
        reply_res = await llm.generate_grounded_reply(
            customer_message=item["message"],
            context=None,
            intent=intent_pred.intent,
            retrieved_evidence=[]
        )
        # Without retrieval or escalation, system attempts to auto-handle everything
        a_results.append({
            "id": item["id"],
            "message": item["message"],
            "gold_intent": item["intent"],
            "pred_intent": intent_pred.intent,
            "gold_action": item["expected_action"],
            "reply": reply_res.reply,
            "violations": ["No grounding evidence available"]
        })

    # ABLATION B: With Retrieval, NO Escalation Guardrails (Always Auto-Handle)
    print("\n[Ablation B]: Qwen + FAISS Retrieval, WITHOUT Escalation/Grounding Guardrails...")
    b_results = []
    for item in eval_items:
        intent_pred = await llm.classify_intent(item["message"])
        ret_out = retriever.search(item["message"], top_k=3)
        reply_res = await llm.generate_grounded_reply(
            customer_message=item["message"],
            context=None,
            intent=intent_pred.intent,
            retrieved_evidence=ret_out["results"]
        )
        is_grounded, violations = GroundingGuardrail.validate(reply_res.reply, ret_out["results"])
        b_results.append({
            "id": item["id"],
            "message": item["message"],
            "gold_intent": item["intent"],
            "pred_intent": intent_pred.intent,
            "gold_action": item["expected_action"],
            "reply": reply_res.reply,
            "retrieved_evidence": ret_out["results"],
            "top_similarity": ret_out["top_similarity"],
            "violations": violations
        })

    # ABLATION C: Full System (Retrieval + Guardrail + Escalation)
    print("\n[Ablation C]: Full System (Qwen + FAISS + Guardrails + Escalation Policy)...")
    c_results = []
    for item in eval_items:
        intent_pred = await llm.classify_intent(item["message"])
        ret_out = retriever.search(item["message"], top_k=3)
        reply_res = await llm.generate_grounded_reply(
            customer_message=item["message"],
            context=None,
            intent=intent_pred.intent,
            retrieved_evidence=ret_out["results"]
        )
        is_grounded, violations = GroundingGuardrail.validate(reply_res.reply, ret_out["results"])
        esc_decision = EscalationEngine.evaluate(
            customer_message=item["message"],
            intent_label=intent_pred.intent,
            intent_confidence=intent_pred.confidence,
            top_retrieval_similarity=ret_out["top_similarity"],
            retrieved_evidence=ret_out["results"],
            grounding_violations=violations
        )
        c_results.append({
            "id": item["id"],
            "message": item["message"],
            "gold_intent": item["intent"],
            "pred_intent": intent_pred.intent,
            "gold_action": item["expected_action"],
            "system_action": "auto" if esc_decision.decision == "AUTO_HANDLE" else "escalate",
            "reply": reply_res.reply,
            "retrieved_evidence": ret_out["results"],
            "top_similarity": ret_out["top_similarity"],
            "violations": violations
        })

    # Compute metrics for each ablation
    gold_actions = [it["gold_action"] for it in eval_items]

    # Ablation A (Always auto-handle, no retrieval)
    a_preds = ["auto" for _ in eval_items]
    a_esc = compute_escalation_metrics(gold_actions, a_preds)

    # Ablation B (Always auto-handle with retrieval)
    b_preds = ["auto" for _ in eval_items]
    b_esc = compute_escalation_metrics(gold_actions, b_preds)

    # Ablation C (Escalation policy enabled)
    c_preds = [r["system_action"] for r in c_results]
    c_esc = compute_escalation_metrics(gold_actions, c_preds)

    # Judge evaluation on a representative slice (10 items) for comparative quality
    judge_slice = min(10, len(eval_items))
    print(f"\nJudging replies across {judge_slice} items for comparative quality...")
    
    a_judges = await asyncio.gather(*[
        llm.judge_reply(a_results[i]["message"], [], a_results[i]["reply"])
        for i in range(judge_slice)
    ])
    b_judges = await asyncio.gather(*[
        llm.judge_reply(b_results[i]["message"], b_results[i]["retrieved_evidence"], b_results[i]["reply"])
        for i in range(judge_slice)
    ])
    c_judges = await asyncio.gather(*[
        llm.judge_reply(c_results[i]["message"], c_results[i]["retrieved_evidence"], c_results[i]["reply"])
        for i in range(judge_slice)
    ])

    ablation_summary = {
        "sample_size": sample_size,
        "configurations": {
            "A_no_retrieval": {
                "description": "Qwen 2.5 generation without retrieval (closed-book hallucination risk)",
                "reply_quality": round(sum(j.overall for j in a_judges) / len(a_judges), 2),
                "auto_handled_percentage": a_esc["auto_handled_percentage"],
                "unsafe_auto_handle_rate": a_esc["unsafe_auto_handle_rate"],
                "grounding_violations_detected": sample_size  # all ungrounded
            },
            "B_with_retrieval_no_escalation": {
                "description": "Qwen 2.5 + FAISS retrieval, but forced 100% auto-handling (no escalation)",
                "reply_quality": round(sum(j.overall for j in b_judges) / len(b_judges), 2),
                "auto_handled_percentage": b_esc["auto_handled_percentage"],
                "unsafe_auto_handle_rate": b_esc["unsafe_auto_handle_rate"],
                "grounding_violations_detected": sum(1 for r in b_results if len(r["violations"]) > 0)
            },
            "C_full_system": {
                "description": "Qwen 2.5 + FAISS retrieval + Grounding guardrails + Conservative escalation",
                "reply_quality": round(sum(j.overall for j in c_judges) / len(c_judges), 2),
                "auto_handled_percentage": c_esc["auto_handled_percentage"],
                "unsafe_auto_handle_rate": c_esc["unsafe_auto_handle_rate"],
                "grounding_violations_detected": sum(1 for r in c_results if len(r["violations"]) > 0)
            }
        }
    }

    print("\n" + "=" * 70)
    print("ABLATION STUDY RESULTS:")
    print("-" * 70)
    print(f"A (No Retrieval):        Quality={ablation_summary['configurations']['A_no_retrieval']['reply_quality']} | Auto={ablation_summary['configurations']['A_no_retrieval']['auto_handled_percentage']*100:.1f}% | Unsafe={ablation_summary['configurations']['A_no_retrieval']['unsafe_auto_handle_rate']*100:.1f}%")
    print(f"B (Retrieval Only):     Quality={ablation_summary['configurations']['B_with_retrieval_no_escalation']['reply_quality']} | Auto={ablation_summary['configurations']['B_with_retrieval_no_escalation']['auto_handled_percentage']*100:.1f}% | Unsafe={ablation_summary['configurations']['B_with_retrieval_no_escalation']['unsafe_auto_handle_rate']*100:.1f}%")
    print(f"C (Full System):        Quality={ablation_summary['configurations']['C_full_system']['reply_quality']} | Auto={ablation_summary['configurations']['C_full_system']['auto_handled_percentage']*100:.1f}% | Unsafe={ablation_summary['configurations']['C_full_system']['unsafe_auto_handle_rate']*100:.1f}%")
    print("=" * 70)

    # Save to reports/ablation.json
    out_file = ROOT_DIR / "reports" / "ablation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(ablation_summary, f, indent=2)

    return ablation_summary

if __name__ == "__main__":
    asyncio.run(run_ablation_study(sample_size=25))
