from typing import List, Dict, Any, Tuple
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity

class MajorityClassBaseline:
    """
    Baseline 1 (Trivial):
    Always predicts majority intent.
    Always produces generic canned response.
    Always escalates.
    """
    def __init__(self):
        self.majority_intent = "delivery_delay"

    def fit(self, intents: List[str]):
        counts = Counter(intents)
        if counts:
            self.majority_intent = counts.most_common(1)[0][0]

    def predict(self, message: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_intent,
            "confidence": 0.50,
            "reply": "Thanks for reaching out. Please contact support for assistance.",
            "decision": "ESCALATE",
            "escalation_reason": "Baseline 1 policy: always escalate."
        }

class TfidfLogisticBaseline:
    """
    Baseline 2 (Simple):
    Intent: TF-IDF + LogisticRegression trained on training data.
    Response: Retrieves nearest historical response via TF-IDF cosine similarity.
    Escalation: Simple similarity threshold.
    """
    def __init__(self, sim_threshold: float = 0.50):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
        self.clf = LogisticRegression(max_iter=1000)
        self.historical_problems: List[str] = []
        self.historical_resolutions: List[str] = []
        self.hist_tfidf = None
        self.sim_threshold = sim_threshold
        self.is_trained = False

    def fit(self, train_messages: List[str], train_intents: List[str], resolutions: List[str]):
        self.historical_problems = train_messages
        self.historical_resolutions = resolutions
        X = self.vectorizer.fit_transform(train_messages)
        self.clf.fit(X, train_intents)
        self.hist_tfidf = X
        self.is_trained = True

    def predict(self, message: str) -> Dict[str, Any]:
        if not self.is_trained:
            raise RuntimeError("Baseline 2 must be trained before calling predict.")

        vec = self.vectorizer.transform([message])
        intent = self.clf.predict(vec)[0]
        probs = self.clf.predict_proba(vec)[0]
        conf = float(max(probs))

        # Retrieve nearest historical resolution using TF-IDF
        sims = cosine_similarity(vec, self.hist_tfidf).flatten()
        top_idx = int(sims.argmax())
        top_sim = float(sims[top_idx])
        retrieved_resolution = self.historical_resolutions[top_idx]

        # Escalation policy
        if top_sim < self.sim_threshold or conf < 0.40:
            decision = "ESCALATE"
            reason = f"Baseline 2: similarity ({top_sim:.2f}) or confidence ({conf:.2f}) below threshold."
        else:
            decision = "AUTO_HANDLE"
            reason = "Baseline 2: similarity above threshold."

        return {
            "intent": intent,
            "confidence": conf,
            "reply": retrieved_resolution,
            "similarity": top_sim,
            "decision": decision,
            "escalation_reason": reason
        }
