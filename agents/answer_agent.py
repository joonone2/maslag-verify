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
- If a step is marked [uncertain], treat it as low-confidence
  and rely on other steps if possible
- If information is contradictory across steps, use the higher-confidence step
- If no verified information is available, say "Not found"

## Output
One word, phrase, or "yes"/"no" — nothing more."""


def _format_verified(verified: dict) -> str:
    if not verified:
        return "(No verified information available)"
    parts = []
    for idx in sorted(verified.keys()):
        step = verified[idx]
        flag = step.get("flag", "unknown")
        confidence = step.get("confidence")
        conf_str = f", confidence={confidence:.3f}" if confidence is not None else ""
        parts.append(
            f"Step {idx} [{flag}{conf_str}]\n"
            f"Sub-query: {step['sub_query']}\n"
            f"Answer: {step['intermediate_answer']}"
        )
    return "\n\n".join(parts)


def generate_final_answer(question: str, pool: EvidencePool) -> str:
    verified = pool.get_all_verified()
    context = _format_verified(verified)

    user = (
        f"Original question: {question}\n\n"
        f"Verified information from each step:\n{context}\n\n"
        "Synthesize a final answer following the rules above."
    )

    return call_llm(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=64,
    )