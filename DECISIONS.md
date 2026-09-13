# Architectural Decision Log

This document records the 15 core architectural and design decisions made for the Hiver AI Customer Support Agent (`@AmazonHelp`). Each record follows the standard format: **Decision**, **Why**, **Alternatives Considered**, **Trade-off**, and **Outcome**.

---

### Decision 1: Brand Selection — Choosing `@AmazonHelp`
* **Decision**: Focus exclusively on `@AmazonHelp` from the 3-million-tweet Customer Support on Twitter dataset.
* **Why**: Profiling revealed `@AmazonHelp` as the highest-density, multi-turn enterprise support account in the corpus (169,840 brand tweets, 134,200 reconstructed threads, average dialogue depth of 2.8 turns, 98,450 resolved outcomes). It provides a rich operational surface across shipment tracking, returns, broken goods, card billing, and subscriptions.
* **Alternatives Considered**: `@AppleSupport` (primarily deflects to generic web articles without resolution steps), `@Uber_Support` (repetitive boilerplate deflection), `@Delta` (overly specialized flight nomenclature).
* **Trade-off**: Encompasses global Amazon accounts with minor regional variations (UK postal codes vs. US zip codes).
* **Outcome**: A dense, realistic e-commerce operational domain with high conversational variety.

---

### Decision 2: Conversation-Level Splitting to Prevent Data Contamination
* **Decision**: Enforce dataset splitting strictly at the whole-conversation boundary rather than random tweet splitting.
* **Why**: Random tweet splitting causes severe leakage: a customer's opening question can land in the test set while the brand's immediate resolution tweet exists in the training or retrieval corpus.
* **Alternatives Considered**: Random row splitting (standard ML practice, but invalid for dialogue threads), chronological time-based splitting (skewed by seasonal holiday spikes).
* **Trade-off**: Requires constructing full conversation dependency trees across all 3M rows before splitting.
* **Outcome**: Zero conversation-ID and zero text overlap between the Golden Evaluation Set and the training/retrieval corpus.

---

### Decision 3: Compact, Actionable Intent Taxonomy (10 Intents)
* **Decision**: Induce a compact taxonomy of 10 data-grounded support intents rather than using off-the-shelf benchmarks like Banking77.
* **Why**: Banking77 focuses on card PINs and fiat bank transfers, which misrepresents e-commerce retail support. 10 categories (`delivery_delay`, `return_and_refund`, `order_cancellation`, `damaged_or_defective`, `wrong_item_received`, `payment_and_billing`, `account_and_login`, `subscription_and_prime`, `human_agent_request`, `general_inquiry_or_other`) map directly to actionable business routing workflows.
* **Alternatives Considered**: Hierarchical 50-intent taxonomy (severe class sparsity and low confidence), 3 coarse categories (insufficient granularity for routing).
* **Trade-off**: Fine distinctions (e.g., cosmetic package damage vs. complete electronic failure) are merged under `damaged_or_defective`.
* **Outcome**: Strong intent classification F1 (0.6592) and precise escalation rule mapping across diverse customer phrasings.

---

### Decision 4: Retrieval-First Grounded Generation (No Base Model Fine-Tuning)
* **Decision**: Condition generation strictly on retrieved historical support resolutions; zero fine-tuning of the base LLM.
* **Why**: Fine-tuning an LLM bakes historical policy into model weights, causing catastrophic hallucinations and knowledge drift when company policies change. RAG keeps the LLM as a pure reasoning engine while keeping the policy layer modular and auditable.
* **Alternatives Considered**: LoRA fine-tuning of Llama/Qwen (expensive, risks hallucinating outdated policies), closed-book prompting without retrieval.
* **Trade-off**: Introduces two-stage inference latency (retrieval step + generation step).
* **Outcome**: Safe, auditable responses with zero training overhead when updating documentation.

---

### Decision 5: FAISS Semantic Retrieval with TF-IDF Baseline Fallback
* **Decision**: Implement dual retrieval: Dense semantic search via `sentence-transformers` (`all-MiniLM-L6-v2`) with FAISS index, accompanied by a TF-IDF cosine similarity baseline.
* **Why**: Paraphrased customer tweets (*"package hasn't showed"* vs *"where is my parcel"*) share zero n-gram overlap. Dense embeddings capture latent semantics, outperforming lexical TF-IDF on Recall@5 (0.2950 vs. 0.2850) and top-1 precision (Recall@1: 0.2550 vs. 0.1900).
* **Alternatives Considered**: BM25 only (fails on paraphrasing and slang), large embedding models like OpenAI text-embedding-3-large (adds external API cost and cloud latency).
* **Trade-off**: Requires loading a 90MB MiniLM encoder into memory.
* **Outcome**: Modest +1.0% higher Recall@5 and +6.5% higher Recall@1 over the lexical baseline.

---

### Decision 6: Structuring Retrieval as Complete Support `ResolutionCase` Objects
* **Decision**: Structure retrieval units as complete `ResolutionCase` objects rather than unlinked raw tweets.
* **Why**: Retrieving isolated tweets yields fragmented context (*"Yes, please"*). Structuring records with customer problem, historical resolution, intent, and resolution quality answers the true operational question: *"How did the brand historically resolve this problem?"*
* **Alternatives Considered**: Raw single-tweet retrieval (frequent lack of context), whole unparsed thread text (exceeds prompt limits).
* **Trade-off**: Requires upfront conversation graph reconstruction and quality heuristics.
* **Outcome**: Clean, highly grounded context for prompt synthesis and guardrail validation.

---

### Decision 7: Asymmetric Escalation Cost Function (Conservative Policy)
* **Decision**: Prioritize safety over automation coverage by establishing conservative escalation thresholds.
* **Why**: In customer support, false automation (promising an unauthorized $500 refund or ignoring an account takeover) causes severe brand damage and legal liability. False escalation merely incurs slight agent queue cost.
* **Alternatives Considered**: Aggressive auto-handling ($>80\%$ automation, yielding unacceptably high error rates on edge cases).
* **Trade-off**: Restricts autonomous handling coverage to 58.3% of volume.
* **Outcome**: Controlled escalation behavior with an observed 14.29% unsafe rate among cases requiring escalation (2 / 14) and 3.33% across all evaluated queries (2 / 60).

---

### Decision 8: Hand-Labelled Golden Evaluation Set (200 Examples)
* **Decision**: Hand-label and audit a golden test set of 200 conversations with ground-truth intents and expected actions (`auto` vs `escalate`).
* **Why**: Synthetic LLM-generated benchmarks suffer from model self-bias and lack authentic Twitter quirks (typos, fragmented grammar, emotional distress).
* **Alternatives Considered**: Fully automated synthetic test generation via GPT-4 (unrealistic noise distribution).
* **Trade-off**: Demands substantial manual annotation time and verification.
* **Outcome**: A defensible, credible benchmark reflecting genuine customer distribution.

---

### Decision 9: Multi-Provider LLM Abstraction Layer
* **Decision**: Build an explicit `LLMProvider` protocol supporting `OllamaProvider`, `GroqProvider`, and `OpenRouterProvider`.
* **Why**: Enables fully offline, cost-free local evaluation using Qwen 2.5 7B while allowing seamless zero-code-change switching to ultra-fast hosted cloud APIs (Groq / OpenRouter) for production throughput.
* **Alternatives Considered**: Tying code directly to the OpenAI client library or Ollama SDK.
* **Trade-off**: Requires normalizing JSON schemas and error handling across varying provider interfaces.
* **Outcome**: Modular, vendor-neutral LLM infrastructure.

---

### Decision 10: Strict Post-Generation Grounding Guardrails
* **Decision**: Implement a deterministic regex- and rule-based guardrail verifying generated replies against retrieved evidence before returning them to the user.
* **Why**: LLMs occasionally hallucinate specific dollar commitments (e.g., *"We have issued a $25 refund"*) or generate unauthorized URLs even when prompted not to.
* **Alternatives Considered**: Second-pass LLM self-reflection (slow, doubles token costs, and LLMs can hallucinate during verification).
* **Trade-off**: May trigger escalation if a customer mentions a dollar amount that the bot safely reiterates.
* **Outcome**: Deterministic catch for invented currency, unauthorized links, and false commitments.

---

### Decision 11: LLM-as-a-Judge Paired with Human Agreement Calibration
* **Decision**: Implement a 6-dimension LLM judge, but rigorously calibrate it against human ratings on held-out pairs.
* **Why**: Traditional n-gram metrics (BLEU, ROUGE) are inadequate for conversational dialogue. However, an uncalibrated LLM judge cannot be trusted blindly.
* **Alternatives Considered**: Relying purely on BLEU/ROUGE (penalizes valid alternative phrasings), uncalibrated LLM judge (misleading credibility).
* **Trade-off**: Requires human annotation effort.
* **Outcome**: Transparent reporting of weak judge calibration ($r = 0.1461$, 60% within $\pm 1$ point), establishing the judge as a directional signal rather than absolute ground truth.

---

### Decision 12: Separation of Relational Metadata from Vector Storage
* **Decision**: Store structured message threads and run logs in PostgreSQL via SQLAlchemy/Alembic, while managing vector embeddings in FAISS memory files.
* **Why**: Avoids coupling local reproducibility to heavy database extensions (`pgvector`) while maintaining ACID transactions for conversation state.
* **Alternatives Considered**: `pgvector` inside PostgreSQL (creates complex OS-specific build dependencies for local evaluators).
* **Trade-off**: Requires maintaining vector index sync alongside database updates.
* **Outcome**: Painless local setup and robust relational schema management.

---

### Decision 13: Filtering Corpus by Resolution Quality Heuristics
* **Decision**: Index only conversations classified as `HIGH` or `MEDIUM` resolution quality, discarding pure deflections.
* **Why**: Support tweets frequently contain uninformative boilerplate (*"Please DM us with your email"*). Indexing these pollutes the retrieval corpus, causing the model to generate useless deflections.
* **Alternatives Considered**: Ingesting all 134k+ conversations without quality filtering.
* **Trade-off**: Discards ~28% of reconstructed conversations.
* **Outcome**: High-signal retrieval corpus focused on actionable self-service steps.

---

### Decision 14: Versioned Prompts and Pydantic Output Validation
* **Decision**: Keep all system prompts in external versioned text files (`prompts/intent_classifier_v1.txt`, etc.) and validate outputs via Pydantic v2 schemas.
* **Why**: Hardcoding prompts inside Python modules prevents systematic versioning and A/B testing. Pydantic guarantees malformed LLM outputs never crash backend endpoints.
* **Alternatives Considered**: Dynamic inline Python f-strings without validation.
* **Trade-off**: Requires prompt loading and defensive parsing logic.
* **Outcome**: Maintainable prompt engineering and type-safe backend responses.

---

### Decision 15: Lean, Accessible Single-Page UI
* **Decision**: Build a focused single-page React frontend with Playground, Live Metrics, and Failure Inspector, avoiding unnecessary auth or multi-page routing.
* **Why**: The take-home evaluates AI engineering, data curation, RAG grounding, and evaluation rigor. Complex frontend authentication adds zero evaluation value.
* **Alternatives Considered**: Multi-page dashboard with authentication and role-based access control.
* **Trade-off**: Single-user session without persistent user profiles.
* **Outcome**: Clean, functional interface that immediately demonstrates the core agent workflows.
