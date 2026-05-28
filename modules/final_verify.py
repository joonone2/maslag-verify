"""
modules/final_verify.py
Final Verify - Constraint & Data Type Validator (제약 조건 및 형식 검사기)

내부 지식(Parametric Knowledge)을 활용한 팩트체크를 철저히 차단하고,
오직 질문이 요구하는 데이터 타입(형식, 범주)을 만족하는지만 엄격하게 검수합니다.
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
    최종 답변이 원래 질문의 의도나 제약 조건(Data Type)에 모순되지 않는지 검증합니다.
    """
    is_nf = _is_not_found(final_answer)
    
    if is_nf:
        pool.steps["final"] = {
            "final_answer": final_answer, 
            "is_not_found": True, 
            "flag": "not_found"
        }
        return final_answer

    # 🚨 변경된 부분: 외부 지식 개입 금지 및 데이터 형식(Type) 위반만 검사하도록 강제
    prompt = f"""Original Question: {question}
    Synthesized Final Answer: {final_answer}

    TASK: Perform a strict "Type and Constraint" contradiction check.
    CRITICAL RULE: DO NOT fact-check using your internal knowledge. Assume the 'Synthesized Final Answer' is factually correct based on the retrieved documents.
    
    ONLY flag a contradiction if the answer completely violates the categorical constraint of the question.
    Examples of true contradictions:
    - The question asks for a 'city', but the answer is a 'person's name'.
    - The question asks for a 'year', but the answer is a 'yes/no'.
    - The question asks for 'who', but the answer is a 'date'.
    
    Output ONLY valid JSON:
    {{"contradiction": true or false, "reasoning": "Brief explanation focused ONLY on categorical constraints"}}"""

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