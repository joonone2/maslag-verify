"""
agents/query_refiner.py
동적 쿼리 조정 - 이전 스텝 결과가 Not Found일 때 다음 서브쿼리 재생성
"""

from utils.llm import call_llm

NOT_FOUND_KEYWORDS = [
    "not found", "i don't know", "i do not know",
    "unknown", "no information", "cannot be determined"
]


def is_not_found(text: str) -> bool:
    return any(kw in text.lower() for kw in NOT_FOUND_KEYWORDS)


def refine_next_query(
    original_query: str,
    prev_sub_query: str,
    prev_answer: str,
    question: str,
) -> str:
    """
    이전 스텝이 Not Found일 때 다음 서브쿼리를 동적으로 재생성.

    Parameters
    ----------
    original_query : 원래 계획된 서브쿼리
    prev_sub_query : 이전 스텝 서브쿼리
    prev_answer    : 이전 스텝 답변 (Not Found)
    question       : 원래 질문

    Returns
    -------
    재생성된 서브쿼리
    """
    prompt = (
        f"Original question: {question}\n\n"
        f"Previous sub-query: {prev_sub_query}\n"
        f"Previous answer: {prev_answer} (information not found)\n\n"
        f"Next planned sub-query: {original_query}\n\n"
        "The previous step failed to find information. "
        "Rewrite the next sub-query to be more specific and self-contained, "
        "without relying on the previous step's answer.\n"
        "Output ONLY the rewritten sub-query."
    )
    return call_llm(
        [{"role": "user", "content": prompt}],
        max_tokens=128,
    )