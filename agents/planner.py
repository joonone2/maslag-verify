"""
agents/planner.py
Planner Agent - 멀티홉 질문을 타입 분류 + 서브쿼리로 분해
+ 실패 스텝에 대한 동적 쿼리 재작성
"""

from utils.llm import call_llm_json, call_llm

_SYSTEM = """You are an expert query planner for multi-hop question answering.

## Step 1: Classify the question type
- "bridge": later sub-queries depend on answers from earlier ones
  Example: "Where was the director of Inception born?"
  → Step1 finds director → Step2 uses that answer to find birthplace
- "comparison": each sub-query independently gathers facts to compare
  Example: "Were A and B born in the same country?"
  → Step1 finds A's country → Step2 independently finds B's country

## Step 2: Decompose into sub-queries
Rules:
- Minimum sub-queries necessary (2-3 preferred, 4 max)
- Each sub-query must be independently searchable as a keyword query
- Sub-queries must NOT overlap or repeat information
- Stop decomposing when the next step's answer would directly be the final answer
- Do NOT include information already stated in the original question
- Do NOT answer — only plan

## Step 3: Output format
Output ONLY valid JSON:
{
  "type": "bridge" or "comparison",
  "sub_queries": ["sub-query 1", "sub-query 2", ...]
}"""

_REPLAN_SYSTEM = """You are an expert query rewriter for multi-hop question answering.

A sub-query has failed to find relevant information.
Your task is to rewrite ONLY the failed sub-query to improve search results.

Rules:
- Use the successful steps' results as context
- Make the failed sub-query more specific and self-contained
- Do NOT rewrite successful steps
- Output ONLY the rewritten sub-query as plain text"""


def plan(question: str) -> dict:
    """
    질문을 타입 분류 + 서브쿼리 리스트로 분해.
    """
    user = f"Question: {question}"

    try:
        result = call_llm_json([
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ])

        task_type = result.get("type", "bridge")
        sub_queries = result.get("sub_queries", [])

        if task_type not in ("bridge", "comparison"):
            task_type = "bridge"

        if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
            return {"type": task_type, "sub_queries": sub_queries}
    except Exception:
        pass

    return {"type": "bridge", "sub_queries": [question]}


def replan_failed_step(
    question: str,
    failed_sub_query: str,
    failed_answer: str,
    successful_steps: dict,
) -> str:
    """
    실패한 스텝의 서브쿼리를 Planner가 재작성.

    Parameters
    ----------
    question        : 원래 질문
    failed_sub_query: 실패한 스텝의 서브쿼리
    failed_answer   : 실패한 답변 (Not Found 등)
    successful_steps: 성공한 스텝들 {step_idx: {sub_query, intermediate_answer}}

    Returns
    -------
    재작성된 서브쿼리
    """
    # 성공한 스텝 컨텍스트 구성
    context_parts = []
    for idx in sorted(successful_steps.keys()):
        step = successful_steps[idx]
        context_parts.append(
            f"Step {idx} (succeeded): {step['sub_query']}\n"
            f"  → Found: {step['intermediate_answer'][:100]}"
        )
    context = "\n".join(context_parts) if context_parts else "No successful steps yet."

    user = (
        f"Original question: {question}\n\n"
        f"Successful steps:\n{context}\n\n"
        f"Failed sub-query: {failed_sub_query}\n"
        f"Failed answer: {failed_answer}\n\n"
        "Rewrite the failed sub-query to be more specific and searchable. "
        "Use the successful steps' findings as context if helpful.\n"
        "Output ONLY the rewritten sub-query."
    )

    return call_llm(
        [
            {"role": "system", "content": _REPLAN_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=128,
        label="planner",
    )