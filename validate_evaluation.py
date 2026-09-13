"""
Validation script for Hiver Customer Support Agent evaluation artifacts.
Guarantees:
1. Golden set size and schema validity (N=200, required fields present)
2. Zero data leakage between held-out benchmark and retrieval corpus
3. Zero contamination of synthetic cases into knowledge base
4. Internal consistency across evaluation.json, evaluation.md, robustness_results.json, and HIVER_AGENT_REPORT.md
5. Support counts add up correctly across all evaluated dimensions
"""

import sys
import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def check_leakage():
    golden_file = ROOT_DIR / "data" / "golden" / "golden_twitter_eval.json"
    corpus_file = ROOT_DIR / "data" / "processed" / "conversations_amazonhelp.json"
    synth_file = ROOT_DIR / "data" / "synthetic" / "robustness_cases.json"

    with open(golden_file, "r", encoding="utf-8") as f:
        golden = json.load(f)
    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(synth_file, "r", encoding="utf-8") as f:
        synth = json.load(f)

    print("Checking Data Integrity & Leakage...")
    assert len(golden) == 200, f"Expected 200 golden examples, found {len(golden)}"
    assert len(synth) == 40, f"Expected 40 synthetic robustness cases, found {len(synth)}"

    # Check golden schema
    for idx, ex in enumerate(golden):
        for field in ["id", "conversation_id", "message", "intent", "expected_action"]:
            assert field in ex, f"Missing {field} in golden item {idx}"

    golden_conv_ids = set(ex["conversation_id"] for ex in golden)
    corpus_conv_ids = set(c["conversation_id"] for c in corpus)
    overlap = golden_conv_ids.intersection(corpus_conv_ids)
    assert len(overlap) == 0, f"LEAKAGE DETECTED: {overlap}"

    # Check text overlap
    golden_texts = set(ex["message"].strip() for ex in golden)
    corpus_texts = set(c["problem_text"].strip() for c in corpus)
    text_overlap = golden_texts.intersection(corpus_texts)
    assert len(text_overlap) == 0, f"TEXT LEAKAGE DETECTED: {text_overlap}"

    # Check synthetic cases never entered corpus
    synth_texts = set(s["message"].strip() for s in synth)
    synth_overlap = synth_texts.intersection(corpus_texts)
    assert len(synth_overlap) == 0, f"SYNTHETIC CONTAMINATION DETECTED: {synth_overlap}"

    print("  [PASS] 0 conversation ID overlap between Golden Benchmark and Retrieval Corpus.")
    print("  [PASS] 0 text overlap between Golden Benchmark and Retrieval Corpus.")
    print("  [PASS] 0 synthetic cases in Retrieval Corpus.")

def check_metric_arithmetic():
    print("\nChecking Metric Arithmetic & Sample Sizes...")
    eval_json_path = ROOT_DIR / "reports" / "evaluation.json"
    with open(eval_json_path, "r", encoding="utf-8") as f:
        ej = json.load(f)

    # 1. Intent Confusion Matrix & Macro Arithmetic
    im = ej["main_system"]["intent_metrics"]
    cm = im["confusion_matrix"]
    labels = im["labels"]
    per_intent = im["per_intent"]

    assert len(labels) == 10, f"Expected 10 intent classes, got {len(labels)}"
    total_samples = sum(sum(row) for row in cm)
    assert total_samples == 60, f"Expected 60 total confusion matrix samples, got {total_samples}"

    correct = sum(cm[i][i] for i in range(10))
    computed_acc = round(correct / total_samples, 4)
    assert computed_acc == im["accuracy"], f"Accuracy mismatch: {computed_acc} vs {im['accuracy']}"

    f1s = []
    precs = []
    recs = []
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fn = sum(cm[i]) - tp
        fp = sum(cm[j][i] for j in range(10)) - tp
        sup = sum(cm[i])
        assert sup == 6, f"Expected exactly 6 support per class, got {sup} for {label}"

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
        precs.append(prec)
        recs.append(rec)

        rep = per_intent[label]
        assert round(prec, 4) == rep["precision"], f"Precision mismatch for {label}"
        assert round(rec, 4) == rep["recall"], f"Recall mismatch for {label}"
        assert round(f1, 4) == rep["f1"], f"F1 mismatch for {label}"

    computed_macro_f1 = round(sum(f1s) / len(f1s), 4)
    computed_macro_prec = round(sum(precs) / len(precs), 4)
    computed_macro_rec = round(sum(recs) / len(recs), 4)

    assert computed_macro_f1 == im["macro_f1"], f"Macro-F1 mismatch: {computed_macro_f1} vs {im['macro_f1']}"
    assert computed_macro_prec == im["macro_precision"], f"Macro-Prec mismatch: {computed_macro_prec} vs {im['macro_precision']}"
    assert computed_macro_rec == im["macro_recall"], f"Macro-Rec mismatch: {computed_macro_rec} vs {im['macro_recall']}"
    print("  [PASS] Intent metrics mathematically verified from Confusion Matrix.")

    # 2. Escalation Metric Denominators
    em = ej["main_system"]["escalation_metrics"]
    cnt = em["counts"]
    tp_e = cnt["true_escalate"]
    fp_e = cnt["false_escalate"]
    fn_e = cnt["unsafe_auto_handle"]
    tn_e = cnt["safe_auto_handle"]
    tot_e = cnt["total"]

    assert tot_e == 60, f"Expected 60 escalation total, got {tot_e}"
    assert tp_e + fp_e + fn_e + tn_e == 60

    prec_e = round(tp_e / (tp_e + fp_e), 4)
    rec_e = round(tp_e / (tp_e + fn_e), 4)
    f1_e = round(2 * prec_e * rec_e / (prec_e + rec_e), 4)
    unsafe_rate = round(fn_e / (tp_e + fn_e), 4)
    false_esc = round(fp_e / (tn_e + fp_e), 4)
    auto_cov = round((tn_e + fn_e) / tot_e, 4)

    assert prec_e == em["escalation_precision"]
    assert rec_e == em["escalation_recall"]
    assert f1_e == em["escalation_f1"]
    assert unsafe_rate == em["unsafe_auto_handle_rate"]
    assert false_esc == em["false_escalation_rate"]
    assert auto_cov == em["auto_handled_percentage"]
    print("  [PASS] Escalation metrics & denominators verified from Confusion Matrix.")

    # 3. Retrieval Consistency
    r_faiss = ej["retrieval"]["dense_faiss"]
    r_tfidf = ej["retrieval"]["lexical_tfidf"]
    assert r_faiss["total_queries_evaluated"] == 200
    assert r_tfidf["total_queries_evaluated"] == 200
    assert r_faiss["recall_at_5"] > r_tfidf["recall_at_5"]
    print("  [PASS] Retrieval metrics verified on N=200 queries.")

def check_artifact_consistency():
    print("\nChecking Artifact Consistency...")
    eval_json_path = ROOT_DIR / "reports" / "evaluation.json"
    robust_json_path = ROOT_DIR / "reports" / "robustness_results.json"
    eval_md_path = ROOT_DIR / "reports" / "evaluation.md"
    report_md_path = ROOT_DIR / "reports" / "HIVER_AGENT_REPORT.md"
    readme_path = ROOT_DIR / "README.md"
    decisions_path = ROOT_DIR / "DECISIONS.md"

    with open(eval_json_path, "r", encoding="utf-8") as f:
        ej = json.load(f)
    with open(robust_json_path, "r", encoding="utf-8") as f:
        rj = json.load(f)
    with open(eval_md_path, "r", encoding="utf-8") as f:
        emd = f.read()
    with open(report_md_path, "r", encoding="utf-8") as f:
        rmd = f.read()
    with open(readme_path, "r", encoding="utf-8") as f:
        readme = f.read()
    with open(decisions_path, "r", encoding="utf-8") as f:
        decisions = f.read()

    # 1. Check intent Macro-F1 consistency
    macro_f1_str = f"{ej['main_system']['intent_metrics']['macro_f1']:.4f}"
    assert macro_f1_str in emd, f"evaluation.md missing Macro-F1 {macro_f1_str}"
    assert macro_f1_str in rmd, f"HIVER_AGENT_REPORT.md missing Macro-F1 {macro_f1_str}"
    assert macro_f1_str in readme, f"README.md missing Macro-F1 {macro_f1_str}"
    assert macro_f1_str in decisions, f"DECISIONS.md missing Macro-F1 {macro_f1_str}"

    # 2. Check retrieval recall@5
    recall_5_str = f"{ej['retrieval']['dense_faiss']['recall_at_5']:.4f}"
    assert recall_5_str in emd, f"evaluation.md missing Recall@5 {recall_5_str}"
    assert recall_5_str in rmd, f"HIVER_AGENT_REPORT.md missing Recall@5 {recall_5_str}"
    assert recall_5_str in readme, f"README.md missing Recall@5 {recall_5_str}"
    assert recall_5_str in decisions, f"DECISIONS.md missing Recall@5 {recall_5_str}"

    # 3. Check synthetic robustness numbers
    assert rj["metrics"]["correct_escalations"] == ej["synthetic_robustness"]["correct_escalations"]
    assert rj["metrics"]["unsafe_auto_handles"] == ej["synthetic_robustness"]["unsafe_auto_handles"]
    assert f"{rj['metrics']['correct_escalation_rate']*100:.1f}%" in emd
    assert f"{rj['metrics']['correct_escalation_rate']*100:.1f}%" in rmd
    assert f"{rj['metrics']['correct_escalation_rate']*100:.1f}%" in readme

    # 4. Check for forbidden stale metrics
    STALE_PATTERNS = ["0.9496", "0.9333", "0.6050", "3.86", "41.7%", "92.5%", "0.2254", "38 / 40", "2 / 40"]
    for sp in STALE_PATTERNS:
        for fname, text in [("evaluation.md", emd), ("HIVER_AGENT_REPORT.md", rmd), ("README.md", readme), ("DECISIONS.md", decisions)]:
            assert sp not in text, f"Found stale pattern [{sp}] in {fname}"

    print(f"  [PASS] Intent Macro-F1 ({macro_f1_str}) perfectly consistent across JSON, MD, Report, README, and DECISIONS.")
    print(f"  [PASS] Retrieval Recall@5 ({recall_5_str}) perfectly consistent.")
    print(f"  [PASS] Synthetic robustness numbers perfectly synchronized.")
    print("  [PASS] No forbidden stale metrics found in documentation.")

def main():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE INDEPENDENT EVALUATION AUDIT")
    print("=" * 70)
    try:
        check_leakage()
        check_metric_arithmetic()
        check_artifact_consistency()
        print("\n" + "=" * 70)
        print("ALL AUDIT CHECKS PASSED: Evaluation is trustworthy, reproducible & consistent.")
        print("=" * 70)
    except AssertionError as e:
        print(f"\n[AUDIT FAILURE]: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
