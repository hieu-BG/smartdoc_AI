import json
import re
from typing import Dict

from ..prompt import sufficiency_prompt


def parse_sufficiency(raw: str) -> Dict:
    match = re.search(r"\{.*?\}", raw, re.DOTALL)

    if not match:
        return {
            "sufficient": False,
            "refined_query": "",
            "reason": "no json found"
        }

    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {
            "sufficient": False,
            "refined_query": "",
            "reason": "invalid json"
        }


def check_sufficiency(llm, question: str, context: str) -> Dict:
    prompt = sufficiency_prompt(question, context)
    raw = llm.invoke(prompt).content
    return parse_sufficiency(raw)