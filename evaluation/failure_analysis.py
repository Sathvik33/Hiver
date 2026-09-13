from typing import List, Dict, Any

def extract_top_failures(eval_results: List[Dict[str, Any]], max_failures: int = 5) -> List[Dict[str, Any]]:
    """
    Extracts representative failures from actual evaluation instances with specific,
    case-grounded explanations of root cause and mitigation.
    """
    failures = []
    
    # 1. Unsafe auto-handles (Critical Severity)
    for res in eval_results:
        if res.get("gold_action") == "escalate" and res.get("system_action") == "auto":
            msg = res.get("message", "")
            g_intent = res.get("gold_intent", "")
            p_intent = res.get("pred_intent", "")
            failures.append({
                "type": "UNSAFE_AUTO_HANDLE",
                "severity": "CRITICAL",
                "input": msg,
                "expected": f"Action: escalate, Intent: {g_intent}",
                "system_output": f"Action: auto, Intent: {p_intent}",
                "evidence_retrieved": res.get("retrieved_evidence", [])[:1],
                "why_it_failed": f"System auto-handled a query expected to escalate ({g_intent}). Intent was predicted as {p_intent} with confidence {res.get('intent_confidence', 0):.2f}.",
                "hypothesis": "Customer message matched routine support vocabulary and retrieval similarity surpassed the 0.60 threshold despite policy risk.",
                "potential_fix": "Tighten escalation safety triggers for ambiguous or sensitive account queries."
            })
            if len(failures) >= max_failures:
                return failures

    # 2. Intent misclassifications
    for res in eval_results:
        if res.get("gold_intent") != res.get("pred_intent"):
            g_intent = res.get("gold_intent", "")
            p_intent = res.get("pred_intent", "")
            msg = res.get("message", "")
            failures.append({
                "type": "INTENT_MISCLASSIFICATION",
                "severity": "MEDIUM",
                "input": msg,
                "expected": f"Intent: {g_intent}",
                "system_output": f"Intent: {p_intent}",
                "evidence_retrieved": res.get("retrieved_evidence", [])[:1],
                "why_it_failed": f"Model classified query into '{p_intent}' instead of true label '{g_intent}'.",
                "hypothesis": f"Customer phrasing contained overlapping vocabulary with '{p_intent}' or spanned multiple concerns.",
                "potential_fix": "Add fine-grained disambiguation guidelines between these two classes in the intent classification prompt."
            })
            if len(failures) >= max_failures:
                return failures

    # 3. False escalations (Low Severity)
    for res in eval_results:
        if res.get("gold_action") == "auto" and res.get("system_action") == "escalate":
            failures.append({
                "type": "FALSE_ESCALATION",
                "severity": "LOW",
                "input": res.get("message", ""),
                "expected": "Action: auto",
                "system_output": f"Action: escalate (Reason: {res.get('escalation_reason')})",
                "evidence_retrieved": res.get("retrieved_evidence", [])[:1],
                "why_it_failed": "System took the conservative path and escalated an otherwise routine inquiry.",
                "hypothesis": f"Retrieval similarity ({res.get('top_similarity', 0):.2f}) was below threshold or phrasing triggered conservative safety gate.",
                "potential_fix": "Augment retrieval index with synonymous customer phrasings or calibrate retrieval threshold."
            })
            if len(failures) >= max_failures:
                return failures

    return failures
