# System Evaluation & Benchmark Results

## 1. Primary Benchmark: Real Twitter Conversations (Held-Out N=200)

Evaluated on held-out genuine `@AmazonHelp` support interactions. Retrieval and baselines evaluated across all $N=200$; main LLM system evaluated on a stratified balanced sample of $N=60$ (6 per intent across all 10 taxonomy classes).

| System / Configuration | Intent Macro-F1 | Intent Accuracy | Retrieval Recall@5 | Reply Quality (1-5) | Unsafe Auto-Handle Rate | Auto-Handled Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Majority Class)** | 0.0182 | 0.1000 | N/A | N/A | 0.0000 | 0.0% |
| **Baseline 2 (TF-IDF + Logistic)** | 0.0715 | 0.1750 | 0.2850 (Lexical) | 3.10 | 0.0222 | 2.5% |
| **Main System (Qwen 2.5 + FAISS)** | **0.6592** | **0.6667** | **0.2950** (Dense) | **3.40** | **0.1429** | **58.3%** |

*Unsafe Auto-Handle Definition: proportion of ground-truth escalation cases mistakenly auto-handled by the system (2 / 14).*

## 2. Independent Retrieval Audit (Real Twitter Data, N=200)
* **Dense Semantic (FAISS)**: Recall@1: 0.2550, Recall@3: 0.2900, Recall@5: 0.2950, MRR: 0.2718
* **Lexical (TF-IDF)**: Recall@1: 0.1900, Recall@3: 0.2550, Recall@5: 0.2850, MRR: 0.2253
* **Advantage of Dense over Lexical**: +1.0% on Recall@5.

## 3. Secondary Safety Suite: Synthetic Robustness Testing (N=40 Edge Cases)
* **Total Robustness Cases**: 40
* **Correct Escalations**: 39 / 40 (97.5%)
* **Unsafe Auto-Handles**: 1 (Rate: 2.5%)
* **Unnecessary Escalations**: 0

## 4. LLM Judge Calibration & Human Agreement (N=25)
* **Pearson Correlation**: -0.0531 (Weak agreement; 7B local model score compression)
* **Within ±1 Point Agreement**: 48.0%
* **Exact Score Agreement**: 44.0%
