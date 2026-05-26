"""
modules/final_verify.py
Final Verify - Not Found 체크만 (룰 기반)
"""

from pool.evidence_pool import EvidencePool
from utils.llm import call_llm

NOT_FOUND_KEYWORDS = [
    "not found", "i don't know", "i do not know",
    "unknown", "no information", "cannot be determined",
    "not available", "not specified"
]


def _is_not_found(text: str) -> bool:
    return any(kw in text.lower() for kw in NOT_FOUND_KEYWORDS)


def _format_verified(verified: dict) -> str:
    if not verified:
        return "(No verified information available)"
    parts = []
    for idx in sorted(verified.keys()):
        step = verified[idx]
        flag = step.get("flag", "unknown")
        parts.append(
            f"Step {idx} [{flag}]\n"
            f"Sub-query: {step['sub_query']}\n"
            f"Answer: {step['intermediate_answer']}"
        )
    return "\n\n".join(parts)


def final_verify(question: str, pool: EvidencePool, final_answer: str) -> str:
    """
    최종 답변 검증:
    - Not Found 류 답변이면 재생성 1회
    - 아니면 그대로 반환
    """
    is_nf = _is_not_found(final_answer)

    pool.steps["final"] = {
        "final_answer": final_answer,
        "is_not_found": is_nf,
        "flag": "regenerated" if is_nf else "passed",
    }

    if not is_nf:
        return final_answer

    verified = pool.get_all_verified()
    all_info = _format_verified(verified)

    system = (
        "The previous answer failed to find the information. "
        "Try to synthesize an answer from the verified information provided. "
        "Answer must be SHORT and DIRECT - only the specific fact."
    )
    user = (
        f"Question: {question}\n"
        f"Verified info:\n{all_info}\n\n"
        "Provide a short direct answer. "
        "If truly not answerable, say 'Not found'."
    )
    corrected = call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=64,
        label="final_verify",
    )

    pool.steps["final"]["regenerated_answer"] = corrected
    return corrected