"""
agents/retrieval.py
Retrieval Agent - 서브쿼리 검색 + 중간 답변 생성 + Pool 저장
"""

from pool.evidence_pool import EvidencePool
from retriever.dense_retriever import ContextRetriever
from utils.llm import call_llm

_SYSTEM = """You are a precise information extraction agent for multi-hop question answering.

## Your task
Extract the specific answer to the sub-query from the retrieved documents.

## Rules
- Answer ONLY based on the retrieved documents. Never use outside knowledge.
- Be as specific and concise as possible (name, date, place, number, yes/no)
- If the answer spans multiple documents, synthesize them
- Use the previous verified context to guide your extraction
  (e.g. if previous step found "Christopher Nolan", focus on information about him)
- If the answer is truly not found in the documents, say exactly: "Not found in documents."
- Do NOT explain or elaborate. Give only the fact.

## Output format
One concise phrase or sentence containing the specific answer."""


def format_docs(docs: list) -> str:
    if not docs:
        return "(No documents retrieved)"
    return "\n\n".join(
        f"[{i}] Title: {d['title']}\n{d['text']}"
        for i, d in enumerate(docs, 1)
    )


def format_context(prev_context: dict) -> str:
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


def _rewrite_query_with_context(sub_query: str, prev_answers: list) -> str:
    """이전 답변의 핵심 엔티티를 서브 쿼리의 대명사와 치환하여 독립적인 검색 쿼리 생성"""
    context_str = " | ".join(prev_answers)
    prompt = (
        f"Context from previous steps: {context_str}\n"
        f"Original Query: {sub_query}\n\n"
        "Rewrite the Original Query to be completely self-contained for a search engine. "
        "Replace ambiguous pronouns (he, she, it, they, that person, the director, etc.) "
        "with the exact specific entities found in the Context. "
        "Do NOT add unnecessary words, just output the clean rewritten query."
    )
    return call_llm([{"role": "user", "content": prompt}], max_tokens=64, label="refine_query")


def retrieve_and_answer(
    sub_query: str,
    pool: EvidencePool,
    retriever: ContextRetriever,
    step_idx: int,
) -> dict:
    # 이전 컨텍스트
    prev_context = pool.get_previous_verified(step_idx)

    # 검색 쿼리 확장 (bridge 타입, Step 0 성공시 이전 답변을 쿼리에 추가)
    search_query = sub_query
    if prev_context and pool.task_type == "bridge" and step_idx > 0:
        prev_answers = [
            v.get("intermediate_answer", "")
            for v in prev_context.values()
            if isinstance(v, dict)
            and v.get("intermediate_answer")
            and not any(kw in v.get("intermediate_answer", "").lower()
                       for kw in ["not found", "i don't know", "unknown"])
        ]
        if prev_answers:
            # === 변경된 부분: 단순 결합 대신 LLM 기반 쿼리 재작성 ===
            search_query = _rewrite_query_with_context(sub_query, prev_answers)

    docs = retriever.search(search_query, top_k=5)

    user = (
        f"Sub-query: {sub_query}\n\n"
        f"Previous verified context:\n{format_context(prev_context)}\n\n"
        f"Retrieved documents:\n{format_docs(docs)}\n\n"
        "Extract the specific answer to the sub-query from the documents above."
    )

    answer = call_llm(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=128,
        label="retrieval",
    )

    pool.add(
        step_idx=step_idx,
        sub_query=sub_query,
        retrieved_titles=[d["title"] for d in docs],
        intermediate_answer=answer,
    )

    return {"answer": answer, "docs": docs}