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
    user_prompt = f"""
You will be given a structured JSON from EVRAS.

Return ONLY valid JSON.
Do not include markdown.
Do not include any extra text.

Required output JSON schema:
{{
  "risk_level": "LOW|MEDIUM|HIGH",
  "confidence": 0.0,
  "factors": ["..."],
  "math_transparency": {{
    "formula": "P = 1 - Π(1 - p_i)",
    "factor_probs": [0.0],
    "final_score": 0.0,
    "calculation_steps": ["..."]
  }},
  "justification": "..."
}}

Math transparency rules:
- Use: P = 1 - Π(1 - p_i)
- Show numeric substitution like:
  P = 1 - (1 - 0.54)(1 - 0.75) = 1 - (...) = 0.909
- The final_score must match the computation.
Use the exact probabilities from input JSON math_transparency.per_factor_probs without rounding.

Risk level thresholds (MUST follow exactly):
- LOW    if confidence < 0.35
- MEDIUM if 0.35 <= confidence < 0.70
- HIGH   if confidence >= 0.70

INPUT JSON:
{json.dumps(risk_output, indent=2)}
"""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ],
        "stream": False,
        "options": {
            "temperature": 0
        }
    }

    r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=300)
    r.raise_for_status()

    response_text = r.json()["message"]["content"]
    return _extract_json(response_text)