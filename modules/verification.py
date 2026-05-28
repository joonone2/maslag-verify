"""
modules/verification.py
Verification Module - Blind Acceptance 방지

bridge     타입: 이전 답변과 현재 답변의 일관성 검증
comparison 타입: 각 스텝 답변이 원래 질문에 관련성/구체성 있는지 검증
"""

from pool.evidence_pool import EvidencePool
from retriever.dense_retriever import DenseRetriever
from utils.logprobs import get_yesno_prob
from utils.llm import call_llm
from agents.planner import replan_failed_step


# ─────────────────────────────────────────────
# 포맷 헬퍼
# ─────────────────────────────────────────────

def _format_docs(docs: list) -> str:
    if not docs:
        return "(No documents retrieved)"
    return "\n\n".join(
        f"[{i}] Title: {d['title']}\n{d['text']}"
        for i, d in enumerate(docs, 1)
    )


def _format_context(prev_context: dict) -> str:
    if not prev_context:
        return "(No previous verified context)"
    parts = []
    for idx in sorted(prev_context.keys()):
        step = prev_context[idx]
        parts.append(
            f"Step {idx} [{step.get('flag', 'unknown')}] "
            f"(Q: {step['sub_query']})\nA: {step['intermediate_answer']}"
        )
    return "\n\n".join(parts)


# ─────────────────────────────────────────────
# 메인 verify - 타입에 따라 분기
# ─────────────────────────────────────────────

def verify(
    step_idx: int,
    pool: EvidencePool,
    retriever: DenseRetriever,
    question: str = "",
) -> dict:
    """
    bridge     타입: _verify_bridge() 호출
    comparison 타입: _verify_comparison() 호출
    """
    if pool.task_type == "comparison":
        return _verify_comparison(step_idx, pool, retriever, question)
    else:
        return _verify_bridge(step_idx, pool, retriever)


# ─────────────────────────────────────────────
# Bridge 검증 - 이전 답변과 일관성
# ─────────────────────────────────────────────

def _verify_bridge(step_idx: int, pool: EvidencePool, retriever: DenseRetriever) -> dict:
    """
    score_1: 이전 정보 포함 여부
    score_2: 의미론적 일관성
    """
    prev_answer = pool.get_latest_answer(step_idx)
    curr_answer = pool.steps[step_idx]["intermediate_answer"]

    if not prev_answer:
        pool.update_verification(step_idx, 1.0, "skipped")
        return {"flag": "skipped"}

    sub_query = pool.steps[step_idx].get("sub_query", "")
    score_1 = _compute_bridge_score(prev_answer, curr_answer, sub_query)
    score_2 = None
    # score_1만으로 판단, Not Found → 즉시 uncertain
    confidence = score_1

    if _is_not_found(pool.steps[step_idx].get("intermediate_answer", "")):
        # Not Found → 1회 재검색 시도
        result = _refine_and_retry_bridge(step_idx, pool, retriever, max_retry=1)
        result["score_1_initial"] = score_1
        result["score_2_initial"] = score_2
        result["verify_type"] = "bridge"
        result["not_found_retry"] = True
        return result

    if confidence >= 0.5:
        pool.update_verification(step_idx, confidence, "verified", score_1, score_2)
        return {"flag": "verified", "confidence": confidence, "score_1": score_1, "score_2": score_2, "verify_type": "bridge"}
    else:
        result = _refine_and_retry_bridge(step_idx, pool, retriever)
        result["score_1_initial"] = score_1
        result["score_2_initial"] = score_2
        result["verify_type"] = "bridge"
        return result


NOT_FOUND_KEYWORDS = ["not found", "i don't know", "i do not know",
                      "unknown", "no information", "cannot be determined"]


def _is_not_found(text: str) -> bool:
    return any(kw in text.lower() for kw in NOT_FOUND_KEYWORDS)


def _compute_bridge_score(prev_answer: str, curr_answer: str, sub_query: str = "") -> float:
    """score_2 제거. Not Found → 0.0, 아니면 로그확률."""
    if _is_not_found(curr_answer):
        return 0.0
    prompt_1 = (
        f"Sub-query: {sub_query}\n"
        f"Previous step found: {prev_answer}\n"
        f"Current answer: {curr_answer}\n\n"
        "Does the current answer address the sub-query "
        "in the context of the previous finding?\n"
        "Note: pronouns like 'she', 'he', 'it', 'they' likely refer to the previous finding.\n"
        "Note: a partial or indirect answer still counts as Yes.\n"
        "Answer Yes or No only."
    )
    return get_yesno_prob(prompt_1)


# ─────────────────────────────────────────────
# Comparison 검증 - 관련성 + 구체성
# ─────────────────────────────────────────────

def _verify_comparison(
    step_idx: int,
    pool: EvidencePool,
    retriever: DenseRetriever,
    question: str,
) -> dict:
    """
    score_1: 원래 질문에 대한 관련성
    score_2: 답변의 구체성 (vague/not found 아닌가)
    """
    curr_answer = pool.steps[step_idx]["intermediate_answer"]
    sub_query = pool.steps[step_idx]["sub_query"]

    score_1 = _compute_comparison_score(question, sub_query, curr_answer)
    score_2 = None
    # score_1만으로 판단, Not Found → 즉시 uncertain
    confidence = score_1

    if _is_not_found(pool.steps[step_idx].get("intermediate_answer", "")):
        # Not Found → 1회 재검색 시도
        result = _refine_and_retry_comparison(step_idx, pool, retriever, question, max_retry=1)
        result["score_1_initial"] = score_1
        result["score_2_initial"] = score_2
        result["verify_type"] = "comparison"
        result["not_found_retry"] = True
        pool.steps[step_idx]["score_1_initial"] = score_1
        pool.steps[step_idx]["score_2_initial"] = score_2
        return result

    if confidence >= 0.5:
        pool.update_verification(step_idx, confidence, "verified", score_1, score_2)
        return {"flag": "verified", "confidence": confidence, "score_1": score_1, "score_2": score_2, "verify_type": "comparison"}
    else:
        result = _refine_and_retry_comparison(step_idx, pool, retriever, question)
        result["score_1_initial"] = score_1
        result["score_2_initial"] = score_2
        result["verify_type"] = "comparison"
        pool.steps[step_idx]["score_1_initial"] = score_1
        pool.steps[step_idx]["score_2_initial"] = score_2
        return result


def _compute_comparison_score(question: str, sub_query: str, curr_answer: str) -> float:
    """score_2 제거. Not Found → 0.0, 아니면 로그확률."""
    if _is_not_found(curr_answer):
        return 0.0
    prompt_1 = (
        f"Original question: {question}\n"
        f"Sub-query: {sub_query}\n"
        f"Answer: {curr_answer}\n\n"
        "Does this answer contain a specific fact "
        "(name, place, date, number, or yes/no) "
        "that directly contributes to answering the original question?\n"
        "Answer Yes if there is ANY concrete information, even partial.\n"
        "Answer No only if the answer is completely unrelated or empty.\n"
        "Answer Yes or No only."
    )
    return get_yesno_prob(prompt_1)


def _classify_comparison_failure(score_1: float, score_2: float) -> str:
    """
    comparison 실패 유형:
    IRRELEVANT  - 관련성 낮음 (score_1 < 0.7)
    VAGUE       - 구체성 낮음 (score_2 < 0.7)
    BOTH        - 둘 다 낮음
    """
    if score_2 is None: return "IRRELEVANT"
    if score_1 < 0.7 and score_2 < 0.7:
        return "BOTH"
    elif score_1 < 0.7:
        return "IRRELEVANT"
    else:
        return "VAGUE"


def _generate_comparison_refined_query(
    sub_query: str,
    question: str,
    curr_answer: str,
    failure_type: str,
) -> str:
    prompts = {
        "IRRELEVANT": (
            f"Original question: {question}\n"
            f"Sub-query: {sub_query}\n"
            f"Current answer: {curr_answer}\n\n"
            "The answer is not relevant to the original question. "
            "Rewrite the sub-query to explicitly connect it to the original question.\n"
            "Output ONLY the refined query."
        ),
        "VAGUE": (
            f"Sub-query: {sub_query}\n"
            f"Current answer: {curr_answer}\n\n"
            "The answer is too vague or not found. "
            "Rewrite the sub-query to be more specific and targeted.\n"
            "Output ONLY the refined query."
        ),
        "BOTH": (
            f"Original question: {question}\n"
            f"Sub-query: {sub_query}\n"
            f"Current answer: {curr_answer}\n\n"
            "The answer is both irrelevant and vague. "
            "Rewrite the sub-query to be specific and directly relevant to the original question.\n"
            "Output ONLY the refined query."
        ),
    }
    return call_llm(
        [{"role": "user", "content": prompts.get(failure_type, prompts["BOTH"])}],
        max_tokens=128,
    )


def _refine_and_retry_comparison(
    step_idx: int,
    pool: EvidencePool,
    retriever: DenseRetriever,
    question: str,
    max_retry: int = 2,
) -> dict:
    final_confidence = 0.0
    final_s1 = 0.0
    final_s2 = 0.0

    for attempt in range(max_retry):
        successful_steps = {
            k: v for k, v in pool.steps.items()
            if isinstance(v, dict)
            and k != step_idx
            and v.get("flag") in ("skipped", "verified", "refined")
        }
        refined_query = replan_failed_step(
            question=question,
            failed_sub_query=pool.steps[step_idx]["sub_query"],
            failed_answer=pool.steps[step_idx]["intermediate_answer"],
            successful_steps=successful_steps,
        )

        new_docs = retriever.search(refined_query, top_k=5)
        new_answer = _generate_answer(
            refined_query=refined_query,
            docs=new_docs,
            prev_context={},  # comparison은 이전 컨텍스트 불필요
        )

        pool.steps[step_idx]["intermediate_answer"] = new_answer
        pool.steps[step_idx]["refined_query"] = refined_query
        pool.steps[step_idx]["refined_titles"] = [d["title"] for d in new_docs]

        score_1 = _compute_comparison_score(
            question,
            pool.steps[step_idx]["sub_query"],
            new_answer,
        )
        score_2 = None
        final_confidence = score_1
        final_s1 = score_1
        final_s2 = score_2

        if final_confidence >= 0.5:
            pool.update_verification(step_idx, final_confidence, "refined", score_1, score_2)
            return {
                "flag": "refined",
                "confidence": final_confidence,
                "score_1": score_1,
                "score_2": score_2,
                "attempts": attempt + 1,
                "refined_query": refined_query,
            }

    pool.update_verification(step_idx, final_confidence, "uncertain", final_s1, final_s2)
    return {
        "flag": "uncertain",
        "confidence": final_confidence,
        "score_1": final_s1,
        "score_2": final_s2,
        "attempts": max_retry,
    }


# ─────────────────────────────────────────────
# Bridge 실패 분류 + Refinement
# ─────────────────────────────────────────────

def classify_failure(prev_answer: str, curr_answer: str) -> str:
    prompt = (
        f"Previous information: {prev_answer}\n"
        f"Current answer: {curr_answer}\n\n"
        "Classify the failure:\n"
        "A: DILUTION - previous info partially present but weakened\n"
        "B: IGNORED - previous info completely absent\n"
        "C: CONTRADICTION - current answer contradicts previous\n\n"
        "Output ONLY one letter: A, B, or C"
    )
    response = call_llm([{"role": "user", "content": prompt}], max_tokens=1)
    letter = response.strip().upper()
    return letter if letter in {"A", "B", "C"} else "B"


def _generate_bridge_refined_query(
    sub_query: str,
    prev_answer: str,
    curr_answer: str,
    failure_type: str,
) -> str:
    prompts = {
        "A": (
            f"Original query: {sub_query}\n"
            f"Key info to preserve: {prev_answer}\n\n"
            "Rewrite the query to explicitly require preservation of the key information.\n"
            "Output ONLY the refined query."
        ),
        "B": (
            f"Original query: {sub_query}\n"
            f"Ignored information: {prev_answer}\n\n"
            "Rewrite the query to explicitly incorporate the ignored information as context.\n"
            "Output ONLY the refined query."
        ),
        "C": (
            f"Original query: {sub_query}\n"
            f"Established info: {prev_answer}\n"
            f"Contradicting answer: {curr_answer}\n\n"
            "Rewrite the query to verify the established facts and exclude contradicting information.\n"
            "Output ONLY the refined query."
        ),
    }
    return call_llm(
        [{"role": "user", "content": prompts.get(failure_type, prompts["B"])}],
        max_tokens=128,
    )


def _generate_answer(refined_query: str, docs: list, prev_context: dict) -> str:
    system = (
        "You are a precise question answering agent.\n"
        "Answer ONLY based on the provided documents.\n"
        "Be concise and specific.\n"
        'If not found, say "Not found in documents".'
    )
    user = (
        f"Sub-query: {refined_query}\n\n"
        f"Previous verified context:\n{_format_context(prev_context)}\n\n"
        f"Retrieved documents:\n{_format_docs(docs)}\n\n"
        "Answer the sub-query concisely."
    )
    return call_llm(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=256,
    )


def _refine_and_retry_bridge(
    step_idx: int,
    pool: EvidencePool,
    retriever: DenseRetriever,
    max_retry: int = 2,
) -> dict:
    final_confidence = 0.0
    final_s1 = 0.0
    final_s2 = 0.0

    for attempt in range(max_retry):
        # 성공한 스텝 수집
        successful_steps = {
            k: v for k, v in pool.steps.items()
            if isinstance(v, dict)
            and k != step_idx
            and v.get("flag") in ("skipped", "verified", "refined")
        }
        refined_query = replan_failed_step(
            question=pool.question if hasattr(pool, "question") else "",
            failed_sub_query=pool.steps[step_idx]["sub_query"],
            failed_answer=pool.steps[step_idx]["intermediate_answer"],
            successful_steps=successful_steps,
        )

        new_docs = retriever.search(refined_query, top_k=5)
        new_answer = _generate_answer(
            refined_query=refined_query,
            docs=new_docs,
            prev_context=pool.get_previous_verified(step_idx),
        )

        pool.steps[step_idx]["intermediate_answer"] = new_answer
        pool.steps[step_idx]["refined_query"] = refined_query
        pool.steps[step_idx]["refined_titles"] = [d["title"] for d in new_docs]

        prev_answer = pool.get_latest_answer(step_idx)
        sub_query_r = pool.steps[step_idx].get("sub_query", "")
        score_1 = _compute_bridge_score(prev_answer, new_answer, sub_query_r)
        score_2 = None
        final_confidence = score_1
        final_s1 = score_1
        final_s2 = score_2

        if final_confidence >= 0.5:
            pool.update_verification(step_idx, final_confidence, "refined", score_1, score_2)
            return {
                "flag": "refined",
                "confidence": final_confidence,
                "score_1": score_1,
                "score_2": score_2,
                "attempts": attempt + 1,
                "refined_query": refined_query,
            }

    pool.update_verification(step_idx, final_confidence, "uncertain", final_s1, final_s2)
    return {
        "flag": "uncertain",
        "confidence": final_confidence,
        "score_1": final_s1,
        "score_2": final_s2,
        "attempts": max_retry,
    }