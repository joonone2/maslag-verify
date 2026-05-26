"""
pipeline.py
MAS+RAG Verification Pipeline 전체 연결
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import traceback

from pool.evidence_pool import EvidencePool
from retriever.dense_retriever import ContextRetriever
from agents.planner import plan
from agents.retrieval import retrieve_and_answer
from agents.answer_agent import generate_final_answer
from modules.verification import verify
from modules.final_verify import final_verify
from agents.query_refiner import refine_next_query, is_not_found as _is_not_found


def run_pipeline(
    question: str,
    gold_answer: str,
    retriever: ContextRetriever,
    context: dict = None,  # HotpotQA distractor context
) -> dict:
    start_time = time.time()
    pool = EvidencePool()
    error_msg = None
    final_answer = ""
    sub_queries = []
    task_type = "bridge"

    try:
        # ── 0. distractor context 설정 ───────────────
        if context is not None:
            retriever.set_context(context)

        # ── 1. Planner ──────────────────────────────
        plan_result = plan(question)
        task_type = plan_result["type"]
        sub_queries = plan_result["sub_queries"]
        pool.task_type = task_type
        print(f"  [Planner] type={task_type}, {len(sub_queries)} sub-queries: {sub_queries}")

        # ── 2. 각 서브쿼리 처리 ─────────────────────
        for step_idx, sub_query in enumerate(sub_queries):

            # 동적 쿼리 조정: 이전 스텝이 Not Found면 현재 쿼리 재생성
            if step_idx > 0:
                prev_step = pool.steps.get(step_idx - 1, {})
                prev_answer = prev_step.get("intermediate_answer", "")
                prev_sub_query = prev_step.get("sub_query", "")
                if _is_not_found(prev_answer):
                    refined = refine_next_query(
                        original_query=sub_query,
                        prev_sub_query=prev_sub_query,
                        prev_answer=prev_answer,
                        question=question,
                    )
                    print(f"  [Step {step_idx}] 쿼리 동적 조정: {sub_query!r}")
                    print(f"              → {refined!r}")
                    sub_query = refined

            print(f"  [Step {step_idx}] Retrieve & Answer: {sub_query!r}")

            retrieve_and_answer(sub_query, pool, retriever, step_idx)

            if step_idx == 0:
                pool.update_verification(step_idx, 1.0, "skipped")
                print(f"  [Step {step_idx}] Verification → skipped (first step)")
            else:
                v_result = verify(step_idx, pool, retriever, question=question)
                _print_verify_result(step_idx, v_result)

        # ── 3. Answer Agent ─────────────────────────
        print("  [Answer Agent] 최종 답변 생성 중...")
        final_answer = generate_final_answer(question, pool)

        # ── 4. Final Verify (Not Found 체크) ─────────
        print("  [Final Verify] 최종 답변 검증 중...")
        final_answer = final_verify(question, pool, final_answer)

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"  [Pipeline ERROR] {e}")

    elapsed = time.time() - start_time

    return {
        "question": question,
        "gold_answer": gold_answer,
        "task_type": task_type,
        "sub_queries": sub_queries,
        "steps": pool.to_log(),
        "final_answer": final_answer,
        "elapsed_sec": round(elapsed, 2),
        "error": error_msg,
    }


def _print_verify_result(step_idx: int, v_result: dict):
    flag = v_result["flag"]
    verify_type = v_result.get("verify_type", "")
    conf = v_result.get("confidence")
    s1 = v_result.get("score_1") or v_result.get("score_1_initial")
    s2 = v_result.get("score_2") or v_result.get("score_2_initial")
    failure_type = v_result.get("failure_type", "")

    if conf is not None and s1 is not None:
        print(
            f"  [Step {step_idx}] Verification({verify_type}) → "
            f"flag={flag}, confidence={conf:.3f} "
            f"(score_1={s1:.3f}, score_2={s2:.3f})"
            + (f", failure={failure_type}" if failure_type else "")
        )
    else:
        print(f"  [Step {step_idx}] Verification({verify_type}) → flag={flag}")