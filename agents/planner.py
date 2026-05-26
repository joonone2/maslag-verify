"""
agents/planner.py
Planner Agent - 멀티홉 질문을 타입 분류 + 서브쿼리로 분해
"""

from utils.llm import call_llm_json

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


def plan(question: str) -> dict:
    """
    질문을 타입 분류 + 서브쿼리 리스트로 분해.

    Returns
    -------
    {
        "type": "bridge" | "comparison",
        "sub_queries": list[str]
    }
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