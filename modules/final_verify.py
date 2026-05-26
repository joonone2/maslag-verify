"""
modules/final_verify.py
Final Verify

bridge 타입:
  - "Not found" 류 답변이면 재생성
  - 아니면 그대로 반환

comparison 타입:
  - Not found 체크 (기본)
  - 추가로 두 스텝 답변이 함께 원래 질문에 답할 수 있는지 로그확률로 검증
"""

from pool.evidence_pool import EvidencePool
from utils.llm import call_llm
from utils.logprobs import get_yesno_prob

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


def _comparison_verify(question: str, verified: dict) -> float:
    """
    comparison 타입 전용 검증.
    각 스텝 답변들을 합쳐서 원래 질문에 답할 수 있는지 로그확률로 판단.
    """
    steps_info = []
    for idx in sorted(verified.keys()):
        step = verified[idx]
        ans = step.get("intermediate_answer", "")
        if _is_not_found(ans):
            return 0.0  # 하나라도 Not found면 실패
        steps_info.append(f"- {ans}")

    if not steps_info:
        return 0.0

    combined = "\n".join(steps_info)
    prompt = (
        f"Original question: {question}\n\n"
        f"Collected information from each step:\n{combined}\n\n"
        "Can the collected information together answer the original question? "
        "(Answer Yes if the information is sufficient to determine the answer, "
        "even if indirectly)\n"
        "Answer Yes or No only."
    )
    return get_yesno_prob(prompt)


def final_verify(question: str, pool: EvidencePool, final_answer: str) -> str:
    verified = pool.get_all_verified()
    all_info = _format_verified(verified)
    is_not_found = _is_not_found(final_answer)
    task_type = pool.task_type

    # comparison 타입: 추가 검증
    comparison_score = None
    if task_type == "comparison" and not is_not_found:
        comparison_score = _comparison_verify(question, verified)
        # comparison 검증 실패 (0.5 미만) → 재생성
        if comparison_score < 0.5:
            is_not_found = True  # 재생성 트리거

    pool.steps["final"] = {
        "final_answer": final_answer,
        "is_not_found": _is_not_found(final_answer),
        "comparison_score": comparison_score,
        "flag": "regenerated" if is_not_found else "passed",
        "task_type": task_type,
    }

    if not is_not_found:
        return final_answer

    # 재생성 1회
    if task_type == "comparison" and comparison_score is not None:
        system = (
            "The collected information is insufficient to answer the question. "
            "Try to synthesize an answer using all available step information. "
            "Answer must be SHORT and DIRECT."
        )
    else:
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