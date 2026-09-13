import json
import re
import httpx
from typing import Optional, Dict, Any
from pathlib import Path
from app.llm.base import LLMProvider, IntentPrediction, GeneratedReply, LLMJudgeEvaluation
from app.core.config import settings
from app.core.logging import logger

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

class OpenRouterProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.OPENROUTER_API_KEY
        self.model = model or settings.OPENROUTER_MODEL
        self.base_url = "https://openrouter.ai/api/v1"
        self._load_prompts()

    def _load_prompts(self):
        with open(PROMPTS_DIR / "intent_classifier_v1.txt", "r", encoding="utf-8") as f:
            self.intent_prompt = f.read()
        with open(PROMPTS_DIR / "reply_generator_v1.txt", "r", encoding="utf-8") as f:
            self.reply_prompt = f.read()
        with open(PROMPTS_DIR / "reply_judge_v1.txt", "r", encoding="utf-8") as f:
            self.judge_prompt = f.read()

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()
        try:
            return json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
            raise ValueError(f"Could not parse JSON: {text}")

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.2) -> str:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is not configured.")
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/hiver-support-agent",
            "X-Title": "Hiver Support Agent",
            "Content-Type": "application/json"
        }
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"}
        }

        async with httpx.AsyncClient(timeout=35.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def classify_intent(self, message: str, context: Optional[str] = None) -> IntentPrediction:
        user_text = f"Customer message: \"{message}\"\n"
        if context:
            user_text += f"Context: \"{context}\"\n"
        user_text += "Classify intent into JSON format."

        try:
            raw = await self.generate(user_text, system_prompt=self.intent_prompt, temperature=0.1)
            parsed = self._extract_json(raw)
            return IntentPrediction(
                intent=parsed.get("intent", "general_inquiry_or_other"),
                confidence=float(parsed.get("confidence", 0.8)),
                reason=parsed.get("reason", "Classified via OpenRouter")
            )
        except Exception as e:
            logger.warning(f"OpenRouter intent classification error: {e}")
            return IntentPrediction(intent="general_inquiry_or_other", confidence=0.4, reason=str(e))

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
            f"Return the grounded JSON response."
        )
        try:
            raw = await self.generate(user_text, system_prompt=self.reply_prompt, temperature=0.2)
            parsed = self._extract_json(raw)
            return GeneratedReply(
                reply=parsed.get("reply", "Please reach out to us via DM."),
                confidence=float(parsed.get("confidence", 0.85)),
                evidence_ids=parsed.get("evidence_ids", []),
                supported_claims=parsed.get("supported_claims", [])
            )
        except Exception as e:
            logger.warning(f"OpenRouter reply generation error: {e}")
            return GeneratedReply(
                reply="Please send us a direct message with your order details so we can assist.",
                confidence=0.5,
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
            f"Score against 6 rubric criteria and return JSON."
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
                correctness=corr, relevance=rel, grounding=grd, completeness=comp,
                style=sty, safety=saf, overall=round(overall, 2), reason=p.get("reason", "Graded by OpenRouter LLM")
            )
        except Exception as e:
            return LLMJudgeEvaluation(
                correctness=3, relevance=3, grounding=3, completeness=3, style=4, safety=4,
                overall=3.33, reason=f"OpenRouter evaluation error: {e}"
            )
