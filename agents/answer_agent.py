"""
agents/answer_agent.py
Answer Agent - 검증된 증거를 종합해 최종 답변 생성
"""

from pool.evidence_pool import EvidencePool
from utils.llm import call_llm

_SYSTEM = """You are a precise answer synthesis expert for multi-hop question answering.

## Your task
Synthesize the verified information from each reasoning step
into a single, direct answer to the original question.

## Rules
- Use ONLY the information provided. Never use outside knowledge.
- For yes/no questions: answer ONLY "yes" or "no"
- For factual questions: answer with the minimum necessary words
  (name, place, date, number — no full sentences)
- Steps marked [uncertain] may still contain useful information.
  Use all available information to synthesize the best possible answer.
- If a step contains multiple possible answers, pick the one most
  directly relevant to the original question.
- If information across steps points to a single most specific answer,
  use that. Do not list multiple answers.
- Only say "Not found" if every single step returned "Not found in documents."

## Output
One word, phrase, or "yes"/"no" — nothing more."""


def _format_verified(verified: dict) -> str:
    """필요한 정보만 전달: sub_query + answer + flag + confidence"""
    if not verified:
        return "(No verified information available)"
    parts = []
    for idx in sorted(verified.keys()):
        step = verified[idx]
        flag = step.get("flag", "unknown")
        conf = step.get("confidence")
        conf_str = f", confidence={conf:.2f}" if isinstance(conf, float) else ""
        parts.append(
            f"Step {idx} [{flag}{conf_str}]\n"
            f"Q: {step['sub_query']}\n"
            f"A: {step['intermediate_answer']}"
        )
    return "\n\n".join(parts)


def generate_final_answer(question: str, pool: EvidencePool) -> str:
    verified = pool.get_all_verified()
    context = _format_verified(verified)

    user = (
        f"Original question: {question}\n\n"
        f"Information from each step:\n{context}\n\n"
        "Synthesize a final answer following the rules above. "
        "Pick the single most specific and relevant answer."
    )

    return call_llm(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=64,
        label="answer",
    )