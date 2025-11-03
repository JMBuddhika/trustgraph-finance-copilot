from __future__ import annotations
from typing import Dict, List
from groq import Groq
from .config import MODELS, GROQ_API_KEY

SYS_JUDGE = """You judge whether an answer is supported by provided evidence.
Return a single JSON object: {"faithfulness": float in [0,1], "notes": "..."}. 
Higher score means stronger support. Penalize any numbers or statements not clearly grounded.
"""

def _client():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")
    return Groq(api_key=GROQ_API_KEY)

def score_faithfulness(question: str, answer_md: str, text_snips: List[str], sql_views: Dict[str,str]) -> float:
    msg = f"""Question: {question}

Answer:
{answer_md}

Evidence (text):
{chr(10).join(text_snips[:8])}

Evidence (SQL previews):
{chr(10).join([f"[{k}]\n{v[:600]}" for k,v in list(sql_views.items())[:3]])}
"""
    try:
        r = _client().chat.completions.create(
            model=MODELS.groq_chat_model,
            messages=[{"role":"system","content":SYS_JUDGE},{"role":"user","content":msg}],
            temperature=0.0,
        )
        content = r.choices[0].message.content.strip()
        import json
        js = json.loads(content)
        return float(js.get("faithfulness", 0.0))
    except Exception:
        return 0.0
