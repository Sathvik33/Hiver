from typing import List, Dict, Any

def compute_escalation_metrics(y_true_action: List[str], y_pred_action: List[str]) -> Dict[str, Any]:
    """
    Computes:
    - Escalation Precision, Recall, F1
    - False-Auto-Handle Rate (Unsafe auto-handle)
    - False-Escalation Rate
    - Overall decision accuracy

    CRITICAL DEFINITION:
    Unsafe auto-handle rate: Proportion of cases where ground truth was 'escalate'
    but the system decided 'auto'.
    """
    tp = 0  # gold=escalate, pred=escalate
    fp = 0  # gold=auto,     pred=escalate (false escalation)
    fn = 0  # gold=escalate, pred=auto     (UNSAFE AUTO-HANDLE)
    tn = 0  # gold=auto,     pred=auto

    for true_act, pred_act in zip(y_true_action, y_pred_action):
        t_esc = true_act.lower() == "escalate"
        p_esc = pred_act.lower() == "escalate"

        if t_esc and p_esc:
            tp += 1
        elif not t_esc and p_esc:
            fp += 1
        elif t_esc and not p_esc:
            fn += 1
        else:
            tn += 1

    total = max(len(y_true_action), 1)
    gold_escalate_total = max(tp + fn, 1)
    gold_auto_total = max(tn + fp, 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-6)

    unsafe_auto_handle_rate = fn / gold_escalate_total
    false_escalation_rate = fp / gold_auto_total
    auto_handled_percentage = (tn + fn) / total

    return {
        "escalation_precision": round(precision, 4),
        "escalation_recall": round(recall, 4),
        "escalation_f1": round(f1, 4),
        "unsafe_auto_handle_rate": round(unsafe_auto_handle_rate, 4),
        "false_escalation_rate": round(false_escalation_rate, 4),
        "auto_handled_percentage": round(auto_handled_percentage, 4),
        "counts": {
            "true_escalate": tp,
            "false_escalate": fp,
            "unsafe_auto_handle": fn,
            "safe_auto_handle": tn,
            "total": total
        }
    }
