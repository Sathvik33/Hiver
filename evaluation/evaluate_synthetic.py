"""
Evaluates the system against the 40-case Synthetic Robustness Suite.
Measures:
- Correct Escalation Rate
- Unsafe Auto-Handle Rate (CRITICAL SAFETY METRIC)
- Unnecessary Escalation Rate
- Grounding / Guardrail Violation Rejection Rate
- Breakdown by Adversarial Category

NOTE: This is the SECONDARY safety test suite.
Results are saved to reports/robustness_results.json and kept completely separate
from the Primary Real Twitter Benchmark.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.agent_service import handle_customer_message

async def evaluate_synthetic_robustness() -> Dict[str, Any]:
    print("=" * 70)
    print("RUNNING SYNTHETIC ROBUSTNESS TEST SUITE (N=40 Cases)")
    print("=" * 70)

    cases_file = ROOT_DIR / "data" / "synthetic" / "robustness_cases.json"
    if not cases_file.exists():
        from data.synthetic.build_robustness_set import main as build_synth
        build_synth()

    with open(cases_file, "r", encoding="utf-8") as f:
        robustness_cases = json.load(f)

    print(f"Loaded {len(robustness_cases)} synthetic robustness test cases.")

    results = []
    correct_escalations = 0
    correct_auto_handles = 0
    unsafe_auto_handles = 0
    unnecessary_escalations = 0

    category_stats = {}

    for case in robustness_cases:
        cat = case["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct_escalations": 0, "unsafe_auto_handles": 0}
        category_stats[cat]["total"] += 1

        agent_res = await handle_customer_message(case["message"])
        system_action = "auto" if agent_res.decision == "AUTO_HANDLE" else "escalate"
        expected = case["expected_behavior"]

        is_safe = True
        if expected == "escalate" and system_action == "auto":
            unsafe_auto_handles += 1
            is_safe = False
            category_stats[cat]["unsafe_auto_handles"] += 1
        elif expected == "escalate" and system_action == "escalate":
            correct_escalations += 1
            category_stats[cat]["correct_escalations"] += 1
        elif expected == "auto" and system_action == "auto":
            correct_auto_handles += 1
        elif expected == "auto" and system_action == "escalate":
            unnecessary_escalations += 1

        results.append({
            "case_id": case["case_id"],
            "category": case["category"],
            "message": case["message"],
            "expected_behavior": expected,
            "system_decision": agent_res.decision,
            "system_action": system_action,
            "escalation_reason": agent_res.escalation_reason,
            "intent_predicted": agent_res.intent.label,
            "intent_confidence": agent_res.intent.confidence,
            "top_retrieval_similarity": agent_res.retrieval.top_similarity,
            "grounding_violations": agent_res.grounding_violations,
            "is_safe": is_safe
        })

    n = len(robustness_cases)
    summary = {
        "suite_name": "Synthetic & Manual Robustness Test Suite",
        "total_test_cases": n,
        "metrics": {
            "correct_escalations": correct_escalations,
            "expected_escalations": sum(1 for c in robustness_cases if c["expected_behavior"] == "escalate"),
            "correct_escalation_rate": round(correct_escalations / max(sum(1 for c in robustness_cases if c["expected_behavior"] == "escalate"), 1), 4),
            "unsafe_auto_handles": unsafe_auto_handles,
            "unsafe_auto_handle_rate": round(unsafe_auto_handles / n, 4),
            "unnecessary_escalations": unnecessary_escalations
        },
        "by_category": category_stats,
        "detailed_results": results
    }

    # Save to reports/robustness_results.json
    reports_dir = ROOT_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_file = reports_dir / "robustness_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SYNTHETIC ROBUSTNESS RESULTS:")
    print("-" * 70)
    print(f"Total Cases:             {n}")
    print(f"Correct Escalations:     {correct_escalations} / {summary['metrics']['expected_escalations']} ({summary['metrics']['correct_escalation_rate']*100:.1f}%)")
    print(f"Unsafe Auto-Handles:     {unsafe_auto_handles} (Rate: {summary['metrics']['unsafe_auto_handle_rate']*100:.1f}%)")
    print(f"Unnecessary Escalations: {unnecessary_escalations}")
    print("=" * 70)
    print(f"Saved detailed results to {out_file}")

    return summary

if __name__ == "__main__":
    asyncio.run(evaluate_synthetic_robustness())
