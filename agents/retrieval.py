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


def retrieve_and_answer(
    sub_query: str,
    pool: EvidencePool,
    retriever: ContextRetriever,
    step_idx: int,
) -> dict:
    # 검색
    docs = retriever.search(sub_query, top_k=5)

    # 이전 컨텍스트
    prev_context = pool.get_previous_verified(step_idx)

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