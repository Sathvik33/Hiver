import json
import re
import httpx
from typing import Optional, List, Dict, Any
from pathlib import Path
from app.llm.base import LLMProvider, IntentPrediction, GeneratedReply, LLMJudgeEvaluation
from app.core.config import settings
from app.core.logging import logger

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

class OllamaProvider(LLMProvider):
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.OLLAMA_MODEL
        self._load_prompts()

    def _load_prompts(self):
        with open(PROMPTS_DIR / "intent_classifier_v1.txt", "r", encoding="utf-8") as f:
            self.intent_prompt = f.read()
        with open(PROMPTS_DIR / "reply_generator_v1.txt", "r", encoding="utf-8") as f:
            self.reply_prompt = f.read()
        with open(PROMPTS_DIR / "reply_judge_v1.txt", "r", encoding="utf-8") as f:
            self.judge_prompt = f.read()

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extracts JSON safely from markdown code fences or raw strings."""
        text = text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
        try:
            return json.loads(text)
        except Exception:
            # Try finding first { and last }
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
            raise ValueError(f"Could not parse valid JSON from text: {text}")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {"temperature": temperature}
        }
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()

    async def classify_intent(self, message: str, context: Optional[str] = None) -> IntentPrediction:
        user_text = f"Customer message: \"{message}\"\n"
        if context:
            user_text += f"Context: \"{context}\"\n"
        user_text += "Classify the intent into JSON format with 'intent', 'confidence', 'reason'."

        try:
            raw = await self.generate(user_text, system_prompt=self.intent_prompt, temperature=0.1)
            parsed = self._extract_json(raw)
            return IntentPrediction(
                intent=parsed.get("intent", "general_inquiry_or_other"),
                confidence=float(parsed.get("confidence", 0.7)),
                reason=parsed.get("reason", "Classified by local model")
            )
        except Exception as e:
            logger.warning(f"Ollama intent classification error or JSON parse failure: {e}")
            # Safe conservative fallback
            return IntentPrediction(
                intent="general_inquiry_or_other",
                confidence=0.40,
                reason="Parsing error or fallback triggered"
            )

    async def generate_grounded_reply(
        self,
        customer_message: str,
        context: Optional[str],
        intent: str,
        retrieved_evidence: list[dict]
    ) -> GeneratedReply:
        evidence_text = ""
        for i, ev in enumerate(retrieved_evidence, 1):
            evidence_text += (
                f"\n--- Example {i} (ID: {ev.get('conversation_id')}) ---\n"
                f"Problem: {ev.get('customer_problem')}\n"
                f"Resolution: {ev.get('resolution')}\n"
            )

        user_text = (
            f"Customer Message: \"{customer_message}\"\n"
            f"Context: \"{context or 'None'}\"\n"
            f"Predicted Intent: {intent}\n\n"
            f"Retrieved Historical Examples:\n{evidence_text}\n\n"
            f"Provide your response in the requested JSON structure."
        )

        try:
            raw = await self.generate(user_text, system_prompt=self.reply_prompt, temperature=0.2)
            parsed = self._extract_json(raw)
            return GeneratedReply(
                reply=parsed.get("reply", "Thanks for contacting us. Please DM us your order details so we can investigate."),
                confidence=float(parsed.get("confidence", 0.75)),
                evidence_ids=parsed.get("evidence_ids", [e.get("conversation_id") for e in retrieved_evidence[:1]]),
                supported_claims=parsed.get("supported_claims", [])
            )
        except Exception as e:
            logger.warning(f"Ollama reply generation failed: {e}")
            return GeneratedReply(
                reply="Thanks for contacting @AmazonHelp! Please send us a direct message with your order number so our team can check this.",
                confidence=0.50,
                evidence_ids=[],
                supported_claims=[]
            )

    async def judge_reply(
        self,
        customer_message: str,
        retrieved_evidence: list[dict],
        generated_reply: str
    ) -> LLMJudgeEvaluation:
        evidence_snippet = "\n".join([f"- {e.get('resolution', '')}" for e in retrieved_evidence[:2]])
        user_text = (
            f"Customer Message: \"{customer_message}\"\n"
            f"Retrieved Evidence:\n{evidence_snippet}\n"
            f"Generated Reply: \"{generated_reply}\"\n\n"
            f"Score this reply against the 6 rubric criteria and return the JSON evaluation."
        )
        try:
            raw = await self.generate(user_text, system_prompt=self.judge_prompt, temperature=0.1)
            p = self._extract_json(raw)
            corr = int(p.get("correctness", 4))
            rel = int(p.get("relevance", 4))
            grd = int(p.get("grounding", 4))
            comp = int(p.get("completeness", 4))
            sty = int(p.get("style", 4))
            saf = int(p.get("safety", 5))
            overall = float(p.get("overall", (corr + rel + grd + comp + sty + saf) / 6.0))
            return LLMJudgeEvaluation(
                correctness=corr,
                relevance=rel,
                grounding=grd,
                completeness=comp,
                style=sty,
                safety=saf,
                overall=round(overall, 2),
                reason=p.get("reason", "Graded by local LLM judge")
            )
        except Exception as e:
            logger.warning(f"Ollama judge parsing error: {e}")
            return LLMJudgeEvaluation(
                correctness=3, relevance=3, grounding=3, completeness=3, style=4, safety=4,
                overall=3.33, reason="Fallback judge evaluation due to parser error"
            )
