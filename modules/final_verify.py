"""
modules/final_verify.py
Final Verify - 룰 기반으로 단순화
"Not found" 포함시만 재생성, 나머지는 그대로 반환
"""

from pool.evidence_pool import EvidencePool
from utils.llm import call_llm

NOT_FOUND_KEYWORDS = [
    "not found", "i don't know", "i do not know",
    "unknown", "no information", "cannot be determined",
    "not available", "not specified"
]


def _is_not_found(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in NOT_FOUND_KEYWORDS)


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
    - "Not found" 류 답변이면 재생성
    - 아니면 그대로 반환
    """
    verified = pool.get_all_verified()
    all_info = _format_verified(verified)

    is_not_found = _is_not_found(final_answer)

    pool.steps["final"] = {
        "final_answer": final_answer,
        "is_not_found": is_not_found,
        "flag": "regenerated" if is_not_found else "passed",
    }

    if not is_not_found:
        return final_answer

    # "Not found"이면 재생성 1회
    system = (
        "The previous answer failed to find the information. "
        "Try to synthesize an answer from the verified information provided. "
        "Answer must be SHORT and DIRECT - only the specific fact."
    )
    user = (
        f"Question: {question}\n"
        f"Verified info:\n{all_info}\n\n"
        "Provide a short direct answer. "
        "If truly not answerable from the information, say 'Not found'."
    )
    corrected = call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=64,
    )

    pool.steps["final"]["regenerated_answer"] = corrected
    return corrected