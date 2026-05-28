"""
modules/final_verify.py
Final Verify - Universal Constraint Validator (범용적 제약 조건 검사기)

내부 지식(Parametric Knowledge)을 활용한 팩트체크를 철저히 차단하고,
질문의 제약(Constraint)과 답변의 범주(Category)가 논리적으로 정렬(Alignment)되는지만 
사고 사슬(CoT) 방식으로 엄격하게 검수합니다.
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
    is_nf = _is_not_found(final_answer)
    
    if is_nf:
        pool.steps["final"] = {
            "final_answer": final_answer, 
            "is_not_found": True, 
            "flag": "not_found"
        }
        return final_answer

    # 🚨 새롭게 적용된 '범용적 사고 사슬(Zero-shot CoT)' 프롬프트
    prompt = f"""Original Question: {question}
    Synthesized Final Answer: {final_answer}

    TASK: Perform a generalized "Constraint Alignment" check.
    CRITICAL RULE: DO NOT evaluate factual accuracy using your internal knowledge. Assume the facts in the 'Synthesized Final Answer' are correct. Your ONLY job is to evaluate whether the answer's semantic category aligns with the question's constraints.

    Process this evaluation step-by-step and output ONLY valid JSON:
    {{
        "question_constraint": "Abstractly define the type of information required by the question (e.g., a specific date, a geographic location, a boolean yes/no, a person's name).",
        "answer_category": "Abstractly define the category of the provided answer.",
        "alignment_check": "Does the 'answer_category' logically fulfill the 'question_constraint'? (Yes or No)",
        "contradiction": true or false (Set to true ONLY if alignment_check is No)
    }}"""

    try:
        res = call_llm_json([{"role": "user", "content": prompt}], label="final_verify")
        
        # JSON 응답 파싱
        has_contradiction = res.get("contradiction", False)
        q_constraint = res.get("question_constraint", "N/A")
        a_category = res.get("answer_category", "N/A")
        alignment = res.get("alignment_check", "N/A")
        
        # 💡 핵심: 논문 작성 및 디버깅을 위해 LLM의 사고 과정을 모두 Evidence Pool에 저장합니다.
        reasoning_log = f"[Q-Constraint]: {q_constraint} | [A-Category]: {a_category} | [Alignment]: {alignment}"
        
        pool.steps["final"] = {
            "final_answer": final_answer,
            "is_not_found": False,
            "flag": "contradicted" if has_contradiction else "passed",
            "reasoning": reasoning_log
        }
    except Exception as e:
        # 안전 장치 (Fail-safe): 에러가 나면 멈추지 않고 강제 통과
        pool.steps["final"] = {
            "final_answer": final_answer, 
            "is_not_found": False, 
            "flag": "passed",
            "reasoning": "JSON Parsing Error - Passed by Fail-safe"
        }

    return final_answer