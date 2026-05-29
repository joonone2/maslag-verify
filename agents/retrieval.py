"""
agents/retrieval.py
Retrieval Agent - 서브쿼리 검색 + 중간 답변 생성 + Pool 저장

(업데이트) Zero-shot CoT 기반 범용적 엔티티 추출기 적용: 
단어의 인접성에 속지 않고, 질문이 요구하는 엔티티 범주(Category)와 추출된 정답의 범주가 
일치하는지 스스로 검열한 뒤 정답을 반환합니다.
"""

from pool.evidence_pool import EvidencePool
from retriever.dense_retriever import ContextRetriever
# 🚨 변경점: JSON 형식으로 강제하기 위해 call_llm_json을 추가로 임포트합니다.
from utils.llm import call_llm, call_llm_json

# 🚨 변경점: 텍스트 추출에서 JSON 기반 4단계 사고 사슬(CoT) 프롬프트로 전면 교체
_SYSTEM = """You are an expert reading comprehension and entity extraction agent for multi-hop question answering.
Your task is to read the provided documents and extract the precise answer to the sub-query.

CRITICAL RULE: You MUST strictly align the semantic category of your extracted answer with the semantic category requested by the sub-query. 
For example, if the sub-query asks for an 'animal', you must not extract a 'person's name' even if it appears nearby in the text.

You MUST process your extraction step-by-step and output ONLY valid JSON in the following format:
{
    "expected_entity_type": "Abstractly define the type of entity requested by the sub-query (e.g., a person, a geographic location, a date, an animal, a specific object, a boolean yes/no, a numerical value).",
    "extracted_candidate": "Extract the best candidate answer from the documents. If not found, write 'Not found'.",
    "candidate_entity_type": "Abstractly define the actual semantic category of your 'extracted_candidate'.",
    "type_alignment_check": "Does the 'candidate_entity_type' perfectly match the 'expected_entity_type'? (Yes or No)",
    "intermediate_answer": "If type_alignment_check is 'Yes', output the extracted_candidate. If 'No', or if no valid candidate was found, strictly output 'Not found in documents'."
}"""


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
            search_query = _rewrite_query_with_context(sub_query, prev_answers)

    docs = retriever.search(search_query, top_k=5)

    user = (
        f"Sub-query: {sub_query}\n\n"
        f"Previous verified context:\n{format_context(prev_context)}\n\n"
        f"Retrieved documents:\n{format_docs(docs)}\n\n"
        "Extract the specific answer to the sub-query from the documents above following the strict JSON format."
    )

    # 🚨 변경점: call_llm -> call_llm_json 으로 변경하여 파싱
    try:
        res = call_llm_json(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            label="retrieval",
        )
        # JSON에서 안전하게 정답만 추출 (실패나 불일치 시 Not found 처리됨)
        answer = res.get("intermediate_answer", "Not found in documents.")
        
        # (선택) 디버깅 및 논문 증빙을 위해 추론 과정을 Evidence Pool에 기록할 수도 있습니다.
        reasoning = f"[Expected]: {res.get('expected_entity_type')} | [Extracted]: {res.get('extracted_candidate')} | [Align]: {res.get('type_alignment_check')}"
        
    except Exception:
        answer = "Not found in documents."
        reasoning = "JSON Parsing Error"

    pool.add(
        step_idx=step_idx,
        sub_query=sub_query,
        retrieved_titles=[d["title"] for d in docs],
        intermediate_answer=answer,
    )
    # 추론 로그를 따로 남기고 싶다면 풀이나 리턴값에 추가 가능합니다. (현재는 기존 로직을 해치지 않게 유지)
    
    return {"answer": answer, "docs": docs}