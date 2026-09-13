# Hiver SDE Intern Take-Home — AI Customer Support Agent
## Technical & Evaluation Report: `@AmazonHelp` Autonomous Support System

**Candidate**: Implementation Engineer  
**Dataset**: Customer Support on Twitter (`thoughtvector/customer-support-on-twitter`, CC BY-NC-SA 4.0)  
**Target Brand**: `@AmazonHelp` (E-commerce / Retail Support)  
**Core Architecture**: Retrieval-First Grounded Generation with Conservative Escalation  
**Evaluation Strategy**: Two Clearly Separated Evaluation Sources (Real Twitter Benchmark vs. Synthetic Robustness Suite)

---

## 1. Problem Framing & Design Philosophy (~0.75 page)

### Operational Context & Brand Selection
In multi-brand customer support, Twitter presents an adversarial operational environment characterized by character limits, fragmented multi-turn threads, colloquial slang, and high customer frustration. Profiling the 3-million-tweet corpus reveals `@AmazonHelp` as the most actionable, multi-turn enterprise dataset (169,840 brand responses across 134,200 reconstructed threads, with an average conversation depth of 2.8 turns). The operational domain spans ten core e-commerce intents: shipment tracking, returns, broken goods, incorrect shipments, account security, billing disputes, and Prime subscription inquiries.

### Why Retrieval-Grounded Support?
Directly asking a large language model (LLM) to resolve customer inquiries "closed-book" is dangerous: LLMs readily invent refund amounts, cite non-existent warranty windows, and promise unauthorized actions. Instead, this system adopts a **retrieval-first, generation-second** architecture. Prior to generating a single token, the system queries an index of historically resolved support cases. The LLM's role is strictly constrained to synthesizing a response grounded in proven resolutions.

### Why Conservative Escalation?
In customer support, **false automation is exponentially more costly than false escalation**:
* *False Escalation* (routing a solvable question to a human agent) costs incremental agent labor.
* *False Automation* (confidently promising an unauthorized $500 refund, or bot-replying to an active account takeover) destroys customer trust and introduces legal and financial liability.

The system is architected around the business objective: **Maximize safe automation coverage while guaranteeing controlled safety behavior on high-stakes interactions.**

---

## 2. System Architecture & Resolution Representation (~1.0 page)

### Complete Support Resolution Unit: `ResolutionCase`
Rather than treating retrieval as an arbitrary tweet-to-tweet semantic search, our retrieval index structures historical data into discrete **`ResolutionCase`** objects:

```text
ResolutionCase
├── case_id: "tw_745072"
├── customer_problem: "@AmazonHelp had a parcel out for nearly 36 hours 'arriving today'..."
├── conversation_context: "Real Twitter dialogue with @AmazonHelp on Wed Oct 11 15:15:30 2017"
├── historical_agent_resolution: "Is your parcel marked as being out for delivery today: amzn.to/track? We expect it by 21:00."
├── intent: "delivery_delay"
├── resolution_quality: "HIGH"
└── source_tweet_id: "745072"
```

The retriever answers: *"How did `@AmazonHelp` historically resolve this exact class of customer issue?"* rather than *"Which tweet contains similar words?"*

### End-to-End Processing Pipeline

```text
Incoming Customer Tweet + Thread Context
                    │
                    ▼
     [Conversation Normalization]
                    │
                    ▼
    [Intent Classifier: 10 Categories] ─── (Confidence < 0.65 or OOD) ───┐
    (Qwen 2.5 7B Structured JSON)                                        │
                    │                                                   │
                    ▼                                                   │
   [Semantic Retrieval: FAISS + Dense Embeddings]                       │
   (Top-k=4 Historical ResolutionCase objects)                          │
                    │                                                   │
                    ▼                                                   │
     [Evidence Filtering & Sim Scoring] ──── (Top Similarity < 0.60) ────┤
                    │                                                   │
                    ▼                                                   │
      [Grounded Reply Generator]                                        │
     (Conditioned strictly on top ResolutionCase)                       │
                    │                                                   │
                    ▼                                                   │
     [Post-Generation Grounding Guardrail] ── (Hallucinations Detected) ─┤
     (Regex checks: invented $, unauthorized URLs, false promises)      │
                    │                                                   │
                    ▼                                                   │
        [Escalation Policy Engine] ◄─────────────────────────────────────┘
    (Deterministic Rules: AUTO_HANDLE vs ESCALATE)
                    │
                    ▼
        [Structured Agent Result]
```

---

## 3. Evaluation Data Strategy & Methodology (~1.25 pages)

### Two Clearly Separated Evaluation Sources
To guarantee scientific validity, the project operates **two strictly separate evaluation sources**:

| Evaluation Source | Sample Size | Data Origin | Evaluation Purpose | Included in Retrieval Index? |
| :--- | :---: | :--- | :--- | :---: |
| **Primary Benchmark** | 200 cases | Held-out real Twitter conversations (`@AmazonHelp`) | Measure real-world historical support performance | **NO (Strict Zero-Leakage Split)** |
| **Secondary Safety Suite** | 40 cases | Manually crafted adversarial & stress-test edge cases | Probe safety limits (fraud, multi-intent, prompt injection, liability) | **NO (Evaluation-Only)** |

*These datasets are NEVER pooled together. All headline metrics reflect real customer conversations.*

### Explicit Sample Sizes across Evaluation Dimensions
To maintain full methodological transparency, evaluation sample sizes are explicitly declared:
* **Retrieval Evaluation ($N=200$)**: All 200 held-out queries evaluated against the development corpus.
* **Baselines Evaluation ($N=200$)**: Both Majority Class and TF-IDF baselines evaluated across all 200 queries.
* **Main LLM System Evaluation ($N=60$)**: Evaluated on a balanced round-robin stratified slice (exactly 6 items per intent across all 10 taxonomy classes) to allow rapid local inference while maintaining uniform class coverage.
* **Judge and Human Agreement ($N=25$)**: Hand-audited sample of 25 pairs evaluated on a 6-dimension 1–5 rubric.
* **Secondary Synthetic Robustness ($N=40$)**: 40 adversarial and safety edge cases.

### Leakage Audit & Held-Out Golden Set
* **Zero Leakage Confirmed**: Verified programmatically that `set(golden_ids).intersection(set(dev_ids)) == 0`. No golden conversations exist in the FAISS index, TF-IDF matrix, classifier training, or prompt templates.
* **Zero Synthetic Contamination**: The 40 synthetic robustness cases are evaluation-only and never entered the retrieval corpus or development pipeline.

### Per-Intent Classification Breakdown (Real Twitter Benchmark, Stratified N=60)

The main Qwen 2.5 7B classifier achieved **0.6592 Macro-F1** and **66.67% Accuracy** across the balanced 10-class benchmark of real customer tweets:

| Intent Identifier | Precision | Recall | F1-Score | Support (Evaluated) |
| :--- | :---: | :---: | :---: | :---: |
| `account_and_login` | 1.0000 | 0.8333 | 0.9091 | 6 |
| `damaged_or_defective` | 0.6000 | 1.0000 | 0.7500 | 6 |
| `delivery_delay` | 0.5000 | 1.0000 | 0.6667 | 6 |
| `general_inquiry_or_other` | 0.2500 | 0.1667 | 0.2000 | 6 |
| `human_agent_request` | 0.7500 | 0.5000 | 0.6000 | 6 |
| `order_cancellation` | 1.0000 | 0.6667 | 0.8000 | 6 |
| `payment_and_billing` | 0.5556 | 0.8333 | 0.6667 | 6 |
| `return_and_refund` | 1.0000 | 0.6667 | 0.8000 | 6 |
| `subscription_and_prime` | 1.0000 | 0.6667 | 0.8000 | 6 |
| `wrong_item_received` | 0.5000 | 0.3333 | 0.4000 | 6 |
| **Macro Average / Total** | **0.7156** | **0.6667** | **0.6592** | **60 evaluated** |

---

## 4. Experimental Results & Component Ablation (~1.0 page)

### Primary Benchmark Results (Held-Out Real Twitter Data)

| System / Pipeline | Intent Macro-F1 | Intent Accuracy | Retrieval Recall@5 | Reply Quality (1–5) | Unsafe Auto-Handle Rate | Automation Coverage (% Auto-Handled) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1 (Majority Class)** | 0.0182 | 0.1000 | N/A | N/A | 0.0000 | 0.0% |
| **Baseline 2 (TF-IDF + Logistic Reg)** | 0.0715 | 0.1750 | 0.2850 (Lexical) | 3.10 | 0.0222 | 2.5% |
| **Main System (Qwen 2.5 + FAISS)** | **0.6592** | **0.6667** | **0.2950** (Dense) | **3.40 / 5.0** | **0.1429** | **58.3%** |

*Unsafe Auto-Handle Metric Definition: Defined strictly as the proportion of cases requiring escalation where the system mistakenly auto-handled (2 / 14 = 14.29%). Across all 60 evaluated customer cases, 2 were unsafe auto-handles (3.33% total failure rate).*

### Independent Retrieval Audit (Real Twitter Data, N=200)

Evaluating TF-IDF Cosine vs. Dense FAISS on the exact same 200 held-out queries confirms that **Dense Semantic Retrieval provides a modest improvement over Lexical matching on Twitter text**:

| Retrieval Methodology | Recall@1 | Recall@3 | Recall@5 | MRR (Mean Reciprocal Rank) |
| :--- | :---: | :---: | :---: | :---: |
| **Lexical Baseline (TF-IDF Cosine)** | 0.1900 | 0.2550 | 0.2850 | 0.2253 |
| **Dense Semantic (FAISS / MiniLM-L6-v2)** | **0.2550** | **0.2900** | **0.2950** | **0.2718** |
| **Performance Difference ($\Delta$)** | **+6.5%** | **+3.5%** | **+1.0%** | **+0.0465** |

*Finding*: Dense retrieval improved Recall@5 from 28.5% to 29.5%, a modest +1 percentage-point improvement over TF-IDF. The primary benefit of dense retrieval is at top-1 precision (+6.5% Recall@1).

### Secondary Safety Suite: Synthetic Robustness Testing ($N=40$ Edge Cases)

The synthetic suite probes adversarial, multi-intent, and liability edge cases:
* **Total Robustness Cases**: 40
* **Correct Escalation Rate**: **97.5%** (39 / 40 cases correctly escalated)
* **Unsafe Auto-Handles**: 1 / 40 (2.5% unsafe rate under extreme adversarial conditions)
* **Category Breakdown**:
  - *Explicit Human Requests (4 cases)*: 100% escalated
  - *Legal / Safety / Injury (4 cases)*: 100% escalated
  - *Adversarial / Prompt Injections (4 cases)*: 100% escalated
  - *Out-of-Domain (4 cases)*: 100% escalated
  - *Ambiguous Messages (4 cases)*: 100% escalated
  - *Unsupported Policy Guarantees (4 cases)*: 100% escalated
  - *Conflicting Precedents (4 cases)*: 100% escalated
  - *Multi-Intent / Context Detection*: Correctly caught 3 out of 4 complex cases

---

## 5. Root-Cause Failure Analysis (~1.0 page)

Analysis of actual evaluation failures from `failure_examples.json`:

### 1. Non-English Multilingual Tweets in Global Dataset
* **Customer Input**: *"@AmazonHelp In der Suche standen 2,99 Hab aber 3,99 zahlen müssen"*
* **Expected**: `escalate` (General inquiry / German locale)
* **System Output**: `auto` (Classified as `payment_and_billing`, Top similarity: 0.6408)
* **Root Cause**: The raw Kaggle Twitter dump contains multilingual Amazon tweets (German, Japanese, Hindi). English-tuned prompts and keyword rules failed to detect German price dispute nuance.
* **Mitigation**: Add fast language detection before intent classification; immediately escalate non-English inquiries to locale-specialized agents.

### 2. Sarcastic or Indirect Human Agent Requests
* **Customer Input**: *"@AmazonHelp you guys seriously need to get customer service reps w/ some semblance of common sense. You royally screwed up my order!"*
* **Expected**: `escalate` (Human agent demand / escalation)
* **System Output**: `auto` (Classified as `damaged_or_defective`, Top similarity: 0.8351)
* **Root Cause**: Customer expressed frustration about reps indirectly without typing explicit keywords like *"speak to a human"*. Dense retrieval matched general customer service complaints.
* **Mitigation**: Add customer sentiment / frustration scoring to the escalation engine.

### 3. Multi-Issue Overlap (Delivery + Account Credentials)
* **Customer Input**: *"@AmazonHelp my order was supposed to ship on friday today is sunday sill nothing. My account wont even login :("*
* **Expected**: `account_and_login`
* **System Output**: `delivery_delay` (Similarity: 0.7085)
* **Root Cause**: Customer mentioned delivery timing and account login issues in the same tweet. Single-label classifier latched onto "order was supposed to ship".
* **Mitigation**: Multi-intent detection rule now flags messages with multiple disjoint concerns and forces escalation.

### 4. Severe Complaint / Allegation Misclassified as Routine Delivery
* **Customer Input**: *"4 Days still no response from @115850 .. on #Redmi4 fraud did by #Amazon @138673 @AmazonHelp https://t.co/ny1BzUmyKo"*
* **Expected**: `general_inquiry_or_other` (Escalate due to fraud allegation)
* **System Output**: `delivery_delay` (Misclassified)
* **Root Cause**: Customer opened with timing ("4 Days still no response"), triggering the classifier's delivery delay patterns despite the serious fraud claim.
* **Mitigation**: Add emergency / fraud keyword triggers in early classification filtering prior to topic routing.

### 5. Meta-Service Questions Misclassified as Direct Human Agent Requests
* **Customer Input**: *"@AmazonHelp hi, does live chat not work anymore? I have an issue with an order and I can't find it"*
* **Expected**: `general_inquiry_or_other` (System channel status inquiry)
* **System Output**: `human_agent_request` (Misclassified)
* **Root Cause**: The phrase "live chat" triggered the agent request heuristic even though the user was asking about channel operational status.
* **Mitigation**: Distinguish meta-inquiries about support channels from direct requests to speak to a human representative.

---

## 6. Limitations & "What is Misleading About My Headline Number?" (~0.5 page)

An intellectually honest evaluation requires scrutinizing headline metrics:

1. **Unsafe Auto-Handle Rate on Real Twitter Data**:
   On real customer tweets, the system recorded 2 unsafe auto-handles out of 14 expected escalations (14.29% unsafe rate among escalations; 3.33% across all 60 evaluated queries). The system is not perfectly safe and requires continuous human-in-the-loop oversight.
2. **Weak LLM Judge Agreement**:
   Human vs. LLM judge calibration on real Twitter data yielded a low correlation ($r = -0.0531$, 48% within $\pm 1$ point). Local 7B models suffer from score compression and cannot serve as reliable ground truth without human calibration.
3. **Historical Dialogue $\neq$ Current Policy**:
   Customer support tweets represent past conversational turns from 2017, not current Amazon policy. Generation is constrained by historical dialogue, which may contain deprecated instructions.
4. **Retrieval Improvement is Modest**:
   Dense retrieval improved Recall@5 from 28.5% to 29.5% (+1.0 percentage point). While Recall@1 saw a +6.5% gain, dense semantic search does not dramatically outperform lexical TF-IDF on short, noisy support tweets.

---

## 7. One More Week: High-ROI Roadmap (~0.5 page)

If allocated an additional week of engineering time:

1. **Fast Language Detection**: Add language identification to immediately route non-English tweets to native-speaking queues.
2. **Hybrid Dense-Sparse BM25 + FAISS**: Combine MiniLM embeddings with sparse BM25 indexing using Reciprocal Rank Fusion.
3. **Sentiment / Frustration Scorer**: Escalate sarcastic and frustrated customer tweets that bypass keyword filters.
4. **Active Learning Queue**: Route low-confidence predictions into a human supervisor queue to expand the golden benchmark.
