"""
utils/llm.py
GPT-4o-mini API 호출 기본 함수 + 호출 수 카운팅
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

# ── 호출 카운터 ──────────────────────────────────────────────
_call_counts = {
    "planner":      0,
    "retrieval":    0,
    "score_1":      0,
    "refine_query": 0,
    "answer":       0,
    "final_verify": 0,
    "logprob":      0,
    "other":        0,
}

def reset_counts():
    for k in _call_counts:
        _call_counts[k] = 0

def get_counts() -> dict:
    return dict(_call_counts)

def get_total() -> int:
    return sum(_call_counts.values())

def increment(label: str):
    if label in _call_counts:
        _call_counts[label] += 1
    else:
        _call_counts["other"] += 1
# ─────────────────────────────────────────────────────────────


def call_llm(messages: list, max_tokens: int = 512, label: str = "other") -> str:
    increment(label)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def call_llm_json(messages: list, label: str = "planner") -> dict:
    import json
    increment(label)
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content.strip()
    return json.loads(text)