"""
utils/llm.py
GPT-4o-mini API 호출 기본 함수
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


def call_llm(messages: list, max_tokens: int = 512) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def call_llm_json(messages: list) -> dict:
    import json
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content.strip()
    return json.loads(text)

