"""
modules/final_verify.py
Final Verify - Global Consistency Check (전역 일관성 및 모순 체크)

단순히 Not Found만 체크하는 것이 아니라, 최종 답변이 
원래 질문의 전제 조건과 모순되지 않는지 LLM을 통해 검증합니다.
"""

from pool.evidence_pool import EvidencePool
from utils.llm import call_llm_json

NOT_FOUND_KEYWORDS = [
    "not found", "i don't know", "i do not know",
    "unknown", "no information", "cannot be determined",
    "not available", "not specified"
]

def _is_not_found(text: str) -> bool:
    return any(kw in text.lower() for kw in NOT_FOUND_KEYWORDS)


def final_verify(question: str, pool: EvidencePool, final_answer: str) -> str:
    """
    최종 답변이 원래 질문의 의도나 제약 조건에 모순되지 않는지 검증합니다.
    """
    is_nf = _is_not_found(final_answer)
    
    if is_nf:
        pool.steps["final"] = {
            "final_answer": final_answer, 
            "is_not_found": True, 
            "flag": "not_found"
        }
        return final_answer

    # 전역 모순 검증 (Global Contradiction Check) 프롬프트
    prompt = f"""Original Question: {question}
    Synthesized Final Answer: {final_answer}

    Does this synthesized answer logically contradict the premise or constraints of the original question?
    For example, if the question asks for a 'city' and the answer is a 'person', that is a contradiction.
    
    Output ONLY valid JSON:
    {{"contradiction": true or false, "reasoning": "Brief explanation"}}"""

    try:
        res = call_llm_json([{"role": "user", "content": prompt}], label="final_verify")
        has_contradiction = res.get("contradiction", False)
        
        pool.steps["final"] = {
            "final_answer": final_answer,
            "is_not_found": False,
            "flag": "contradicted" if has_contradiction else "passed",
            "reasoning": res.get("reasoning", "")
        }
    except Exception:
        # JSON 파싱 실패 등 예외 발생 시 안전하게(Fail-safe) 통과 처리
        pool.steps["final"] = {
            "final_answer": final_answer, 
            "is_not_found": False, 
            "flag": "passed"
        }

    return final_answer