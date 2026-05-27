"""
modules/final_verify.py
Final Verify - Not Found 체크만 (재생성 없음)

Answer Agent가 uncertain 포함 모든 정보를 적극 활용하므로
Final Verify는 단순 체크만 수행.
"""

from pool.evidence_pool import EvidencePool

NOT_FOUND_KEYWORDS = [
    "not found", "i don't know", "i do not know",
    "unknown", "no information", "cannot be determined",
    "not available", "not specified"
]


def _is_not_found(text: str) -> bool:
    return any(kw in text.lower() for kw in NOT_FOUND_KEYWORDS)


def final_verify(question: str, pool: EvidencePool, final_answer: str) -> str:
    """
    최종 답변 검증:
    - Not Found 류 답변이면 flag=regenerated로 기록 (재생성 없음)
    - 아니면 그대로 반환
    """
    is_nf = _is_not_found(final_answer)

    pool.steps["final"] = {
        "final_answer": final_answer,
        "is_not_found": is_nf,
        "flag": "not_found" if is_nf else "passed",
    }

    return final_answer