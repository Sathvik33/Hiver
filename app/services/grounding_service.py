import re
from typing import List, Dict, Any, Tuple

# Suspicious patterns that signal potential hallucination or unauthorized commitments
SUSPICIOUS_AMOUNT_PATTERN = re.compile(r'(\$\s*\d+|\b\d+\s*dollars?\b|\b\d+\s*rupees?\b|£\s*\d+|€\s*\d+)', re.IGNORECASE)
UNAUTHORIZED_URL_PATTERN = re.compile(r'https?://(?:(?!amzn\.to|amazon\.com|theupsstore\.com|kohls\.com)[^\s)]+)', re.IGNORECASE)
UNAUTHORIZED_PROMISES = [
    "guarantee a full refund",
    "will wire the money",
    "compensate you $",
    "gift card of $",
    "credit of $",
    "we have already issued a refund",
    "lawsuit",
    "free prime for life",
    "promise that your package will arrive in 1 hour"
]

class GroundingGuardrail:
    """
    Validates generated support reply against retrieved evidence and safety policies.
    Checks:
    1. Unauthorized URLs not belonging to brand or official partners.
    2. Invented refund amounts or currency promises.
    3. Unsupported guarantee claims.
    """

    @staticmethod
    def validate(reply_text: str, retrieved_evidence: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        violations = []
        lower_reply = reply_text.lower()

        # 1. URL check
        unauth_urls = UNAUTHORIZED_URL_PATTERN.findall(reply_text)
        if unauth_urls:
            violations.append(f"Contains unauthorized or unknown external URL: {unauth_urls[0]}")

        # 2. Check for invented monetary sums not present in evidence
        evidence_str = " ".join([f"{e.get('customer_problem', '')} {e.get('resolution', '')}" for e in retrieved_evidence])
        amounts_in_reply = SUSPICIOUS_AMOUNT_PATTERN.findall(reply_text)
        for amt in amounts_in_reply:
            if amt.lower() not in evidence_str.lower():
                violations.append(f"Invented monetary amount not present in evidence: {amt}")

        # 3. Unauthorized policy promises
        for promise in UNAUTHORIZED_PROMISES:
            if promise in lower_reply:
                violations.append(f"Contains unauthorized commitment: '{promise}'")

        is_grounded = len(violations) == 0
        return is_grounded, violations
