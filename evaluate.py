import json
import re
import string
from collections import Counter
import argparse

def normalize_answer(s: str) -> str:
    """
    NLP QA 논문 표준 정규화 함수 (SQuAD style)
    - 소문자화, 구두점 제거, 관사(a, an, the) 제거, 다중 공백 제거
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """정규화 후 정확히 일치하는지 확인"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def f1_score(prediction: str, ground_truth: str) -> float:
    """단어(Token) 레벨의 겹침(Overlap)을 측정하여 F1 Score 계산"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    # 두 답변 모두 공백인 경우
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

def evaluate_logs(log_path: str):
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {log_path} 파일을 찾을 수 없습니다.")
        return

    total_samples = len(data)
    if total_samples == 0:
        print("로그 파일이 비어 있습니다.")
        return

    em_count = 0
    f1_sum = 0.0
    
    yesno_total = 0
    yesno_correct = 0
    
    total_llm_calls = 0
    total_elapsed_sec = 0.0
    
    final_flags = Counter()
    task_types = Counter()

    # 에러가 발생하여 정상 처리되지 않은 샘플 추적
    error_count = 0

    for item in data:
        if item.get("error"):
            error_count += 1
            continue

        pred = item.get("final_answer", "")
        gold = item.get("gold_answer", "")
        
        # 1. EM & F1
        if exact_match_score(pred, gold):
            em_count += 1
        f1_sum += f1_score(pred, gold)
        
        # 2. Yes/No Accuracy
        gold_norm = normalize_answer(gold)
        if gold_norm in ['yes', 'no']:
            yesno_total += 1
            if normalize_answer(pred) == gold_norm:
                yesno_correct += 1

        # 3. Efficiency Metrics (Calls & Time)
        total_llm_calls += item.get("llm_calls_total", 0)
        total_elapsed_sec += item.get("elapsed_sec", 0.0)

        # 4. Final Verify Flags & Task Types
        steps = item.get("steps", {})
        final_step = steps.get("final", {})
        flag = final_step.get("flag", "unknown")
        final_flags[flag] += 1
        
        task_types[item.get("task_type", "unknown")] += 1

    valid_samples = total_samples - error_count

    # 결과 출력
    print("=" * 50)
    print("🚀 MAS-RAG Evaluation Report 🚀")
    print("=" * 50)
    print(f"Total Samples    : {total_samples}")
    if error_count > 0:
        print(f"Errors Occurred  : {error_count} (Excluded from metric calc)")
    
    print("-" * 50)
    print("[1] Accuracy Metrics")
    print(f"  Exact Match (EM) : {em_count / valid_samples * 100:.2f}% ({em_count}/{valid_samples})")
    print(f"  F1 Score         : {f1_sum / valid_samples * 100:.2f}%")
    if yesno_total > 0:
        print(f"  Yes/No Accuracy  : {yesno_correct / yesno_total * 100:.2f}% ({yesno_correct}/{yesno_total})")
    
    print("-" * 50)
    print("[2] Efficiency Metrics")
    print(f"  Avg LLM Calls    : {total_llm_calls / valid_samples:.2f} calls/query")
    print(f"  Avg Elapsed Time : {total_elapsed_sec / valid_samples:.2f} sec/query")
    
    print("-" * 50)
    print("[3] System Behavior (Final Flags)")
    for flag, cnt in sorted(final_flags.items()):
        print(f"  - {flag:<12} : {cnt}")
        
    print("-" * 50)
    print("[4] Task Distribution")
    for t_type, cnt in sorted(task_types.items()):
        print(f"  - {t_type:<12} : {cnt}")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="logs/results_0527_최종.json", help="Path to the log JSON file")
    args = parser.parse_args()
    
    evaluate_logs(args.log)