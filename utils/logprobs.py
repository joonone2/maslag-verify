"""
utils/logprobs.py
토큰 확률 추출 함수 - Yes/No logprob 기반 confidence score
"""

import os
import math
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


def get_yesno_prob(prompt: str) -> float:
    """
    Yes/No 질문에 대해 Yes 확률 반환 (0~1)
    top_logprobs에서 Yes/No 찾아서 정규화.
    둘 다 없으면 텍스트 응답으로 fallback (0.85 or 0.15 - 극단값 피함)
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0.0,
        logprobs=True,
        top_logprobs=5,
    )

    choice = response.choices[0]
    top_logprobs = choice.logprobs.content[0].top_logprobs if choice.logprobs else []

    yes_prob = 0.0
    no_prob = 0.0

    for item in top_logprobs:
        token_lower = item.token.strip().lower()
        prob = math.exp(item.logprob)
        if token_lower == "yes":
            yes_prob = prob
        elif token_lower == "no":
            no_prob = prob

    # Yes/No 둘 다 top_logprobs에 없는 경우 - 완화된 fallback
    if yes_prob == 0.0 and no_prob == 0.0:
        top_token = ""
        if choice.logprobs and choice.logprobs.content:
            top_token = choice.logprobs.content[0].token.strip().lower()
        return 0.85 if top_token == "yes" else 0.15

    total = yes_prob + no_prob
    if total == 0.0:
        return 0.15

    return yes_prob / total