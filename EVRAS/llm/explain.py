import json
import requests
import re
from llm.prompt import SYSTEM_PROMPT


def _extract_json(text: str) -> dict:
    if not text:
        raise ValueError("Empty response from Ollama")

    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in response:\n{text}")

    return json.loads(match.group(0))


def explain_risk_with_ollama(risk_output: dict, model: str = "llama3.1:8b") -> dict:
    """
    LLM must NOT recompute math.
    It only explains the already computed risk score.
    """

    compact_input = {
        "risk_level": risk_output.get("risk_level"),
        "confidence": risk_output.get("confidence"),
        "top_factors": risk_output.get("factors", []),
        "math_transparency": risk_output.get("math_transparency", {})
    }

    user_prompt = f"""
You will be given EVRAS output JSON.

Return ONLY valid JSON.
No markdown. No extra text.

Required output JSON schema:
{{
  "risk_level": "LOW|MEDIUM|HIGH",
  "confidence": 0.0,
  "factors": ["..."],
  "justification": "..."
}}

Rules:
- Use ONLY the provided evidence.
- DO NOT recompute confidence.
- DO NOT output math_transparency.
- risk_level must match the given risk_level.
- factors should be short and based on top_factors.

INPUT JSON:
{json.dumps(compact_input, indent=2)}
"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        "stream": False,
        "options": {"temperature": 0}
    }

    r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=300)
    r.raise_for_status()

    response_text = r.json()["message"]["content"]
    return _extract_json(response_text)