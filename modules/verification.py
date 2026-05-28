"""
modules/verification.py
Verification Module - LLM-as-a-Judge 기반 자가 검증 (Self-Reflective Verification)

bridge     타입: 이전 답변과 현재 답변의 일관성 및 맥락 포함 여부 검증
comparison 타입: 원래 질문에 대한 관련성 및 구체적 사실 포함 여부 검증
"""

from pool.evidence_pool import EvidencePool
from retriever.dense_retriever import DenseRetriever
from utils.llm import call_llm, call_llm_json
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
    if pool.task_type == "comparison":
        return _verify_comparison(step_idx, pool, retriever, question)
    else:
        return _verify_bridge(step_idx, pool, retriever)


# ─────────────────────────────────────────────
# LLM-as-a-Judge 채점 로직 (Logprob 대체)
# ─────────────────────────────────────────────

NOT_FOUND_KEYWORDS = ["not found", "i don't know", "i do not know", "unknown", "no information", "cannot be determined"]

def _is_not_found(text: str) -> bool:
    return any(kw in text.lower() for kw in NOT_FOUND_KEYWORDS)

def _compute_bridge_score(prev_answer: str, curr_answer: str, sub_query: str = "") -> float:
    if _is_not_found(curr_answer):
        return 0.0
    
    prompt = f"""Evaluate if the 'Current answer' logically addresses the 'Sub-query' based on the context of the 'Previous step found'.
    Previous step found: {prev_answer}
    Sub-query: {sub_query}
    Current answer: {curr_answer}

    Evaluation Rubric (1-5 scale):
    1: Completely fails, contradicts, or is totally irrelevant.
    2: Mentions keywords but lacks any specific answer.
    3: Partially answers the query but lacks exact specific details.
    4: Good, specific answer, but slightly ambiguous or verbose.
    5: Perfect, specific, and unambiguous fact.

    Output ONLY valid JSON:
    {{"reasoning": "Brief explanation of the score", "support_score": <int 1-5>}}"""
    
    try:
        res = call_llm_json([{"role": "user", "content": prompt}], label="score_1")
        score_int = res.get("support_score", 1)
        return score_int / 5.0  # 1~5점을 0.2 ~ 1.0으로 정규화
    except Exception:
        return 0.2

def _compute_comparison_score(question: str, sub_query: str, curr_answer: str) -> float:
    if _is_not_found(curr_answer):
        return 0.0
    
    prompt = f"""Evaluate if the 'Current answer' provides a specific concrete fact to answer the 'Sub-query' in the context of the 'Original question'.
    Original question: {question}
    Sub-query: {sub_query}
    Current answer: {curr_answer}

    Evaluation Rubric (1-5 scale):
    1: Completely fails or is irrelevant.
    2: Vague answer, missing the specific comparative attribute (e.g. name, date, country).
    3: Contains some relevant info but misses the core specific fact.
    4: Specific fact is present but mixed with unnecessary information.
    5: Perfect, concise, and exact specific fact found.

    Output ONLY valid JSON:
    {{"reasoning": "Brief explanation of the score", "support_score": <int 1-5>}}"""
    
    try:
        res = call_llm_json([{"role": "user", "content": prompt}], label="score_1")
        score_int = res.get("support_score", 1)
        return score_int / 5.0  # 1~5점을 0.2 ~ 1.0으로 정규화
    except Exception:
        return 0.2


# ─────────────────────────────────────────────
# Bridge / Comparison 검증 흐름
# ─────────────────────────────────────────────

def _verify_bridge(step_idx: int, pool: EvidencePool, retriever: DenseRetriever) -> dict:
    prev_answer = pool.get_latest_answer(step_idx)
    curr_answer = pool.steps[step_idx]["intermediate_answer"]

    if not prev_answer:
        pool.update_verification(step_idx, 1.0, "skipped")
        return {"flag": "skipped"}

    sub_query = pool.steps[step_idx].get("sub_query", "")
    score_1 = _compute_bridge_score(prev_answer, curr_answer, sub_query)
    score_2 = None
    confidence = score_1

    # ESWA 기준: Threshold 0.6 (3점 이상)으로 강건함 확보
    if confidence >= 0.6 and not _is_not_found(curr_answer):
        pool.update_verification(step_idx, confidence, "verified", score_1, score_2)
        return {"flag": "verified", "confidence": confidence, "score_1": score_1, "score_2": score_2, "verify_type": "bridge"}
    else:
        result = _refine_and_retry_bridge(step_idx, pool, retriever, max_retry=1)
        result["score_1_initial"] = score_1
        result["score_2_initial"] = score_2
        result["verify_type"] = "bridge"
        return result

def _verify_comparison(step_idx: int, pool: EvidencePool, retriever: DenseRetriever, question: str) -> dict:
    curr_answer = pool.steps[step_idx]["intermediate_answer"]
    sub_query = pool.steps[step_idx]["sub_query"]

    score_1 = _compute_comparison_score(question, sub_query, curr_answer)
    score_2 = None
    confidence = score_1

    if confidence >= 0.6 and not _is_not_found(curr_answer):
        pool.update_verification(step_idx, confidence, "verified", score_1, score_2)
        return {"flag": "verified", "confidence": confidence, "score_1": score_1, "score_2": score_2, "verify_type": "comparison"}
    else:
        result = _refine_and_retry_comparison(step_idx, pool, retriever, question, max_retry=1)
        result["score_1_initial"] = score_1
        result["score_2_initial"] = score_2
        result["verify_type"] = "comparison"
        pool.steps[step_idx]["score_1_initial"] = score_1
        pool.steps[step_idx]["score_2_initial"] = score_2
        return result


# ─────────────────────────────────────────────
# Retry / Refine 로직
# ─────────────────────────────────────────────

def _generate_answer(refined_query: str, docs: list, prev_context: dict) -> str:
    system = (
        "You are a precise question answering agent.\n"
        "Answer ONLY based on the provided documents.\n"
        "Be concise and specific.\n"
        "If not found, say 'Not found in documents'."
    )
    user = (
        f"Sub-query: {refined_query}\n\n"
        f"Previous verified context:\n{_format_context(prev_context)}\n\n"
        f"Retrieved documents:\n{_format_docs(docs)}\n\n"
        "Answer the sub-query concisely."
    )
    return call_llm([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=128)


def _refine_and_retry_bridge(step_idx: int, pool: EvidencePool, retriever: DenseRetriever, max_retry: int = 1) -> dict:
    final_confidence = 0.0
    final_s1 = 0.0

    for attempt in range(max_retry):
        successful_steps = {
            k: v for k, v in pool.steps.items() 
            if isinstance(v, dict) and k != step_idx and v.get("flag") in ("skipped", "verified", "refined")
        }
        refined_query = replan_failed_step(
            question=pool.question if hasattr(pool, "question") else "",
            failed_sub_query=pool.steps[step_idx]["sub_query"],
            failed_answer=pool.steps[step_idx]["intermediate_answer"],
            successful_steps=successful_steps
        )

        new_docs = retriever.search(refined_query, top_k=5)
        new_answer = _generate_answer(refined_query, new_docs, pool.get_previous_verified(step_idx))

        pool.steps[step_idx]["intermediate_answer"] = new_answer
        pool.steps[step_idx]["refined_query"] = refined_query
        pool.steps[step_idx]["refined_titles"] = [d["title"] for d in new_docs]

        prev_answer = pool.get_latest_answer(step_idx)
        score_1 = _compute_bridge_score(prev_answer, new_answer, pool.steps[step_idx].get("sub_query", ""))
        
        final_confidence = score_1
        final_s1 = score_1

        if final_confidence >= 0.6:
            pool.update_verification(step_idx, final_confidence, "refined", score_1, None)
            return {
                "flag": "refined", "confidence": final_confidence, "score_1": score_1, 
                "score_2": None, "attempts": attempt + 1, "refined_query": refined_query
            }

    pool.update_verification(step_idx, final_confidence, "uncertain", final_s1, None)
    return {
        "flag": "uncertain", "confidence": final_confidence, "score_1": final_s1, 
        "score_2": None, "attempts": max_retry
    }


def _refine_and_retry_comparison(step_idx: int, pool: EvidencePool, retriever: DenseRetriever, question: str, max_retry: int = 1) -> dict:
    final_confidence = 0.0
    final_s1 = 0.0

    for attempt in range(max_retry):
        successful_steps = {
            k: v for k, v in pool.steps.items() 
            if isinstance(v, dict) and k != step_idx and v.get("flag") in ("skipped", "verified", "refined")
        }
        refined_query = replan_failed_step(question, pool.steps[step_idx]["sub_query"], pool.steps[step_idx]["intermediate_answer"], successful_steps)
        
        new_docs = retriever.search(refined_query, top_k=5)
        new_answer = _generate_answer(refined_query, new_docs, {})

        pool.steps[step_idx]["intermediate_answer"] = new_answer
        pool.steps[step_idx]["refined_query"] = refined_query
        pool.steps[step_idx]["refined_titles"] = [d["title"] for d in new_docs]

        score_1 = _compute_comparison_score(question, pool.steps[step_idx]["sub_query"], new_answer)
        final_confidence = score_1
        final_s1 = score_1

        if final_confidence >= 0.6:
            pool.update_verification(step_idx, final_confidence, "refined", score_1, None)
            return {
                "flag": "refined", "confidence": final_confidence, "score_1": score_1, 
                "score_2": None, "attempts": attempt + 1, "refined_query": refined_query
            }

    pool.update_verification(step_idx, final_confidence, "uncertain", final_s1, None)
    return {
        "flag": "uncertain", "confidence": final_confidence, "score_1": final_s1, 
        "score_2": None, "attempts": max_retry
    }