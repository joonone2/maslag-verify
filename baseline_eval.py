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
    if not s:
        return ""
    
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

def evaluate_early_logs(log_path: str):
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: '{log_path}' 파일을 찾을 수 없습니다.")
        return

    total_samples = len(data)
    if total_samples == 0:
        print("로그 파일이 비어 있습니다.")
        return

    em_count = 0
    f1_sum = 0.0
    
    yesno_total = 0
    yesno_correct = 0
    
    empty_answer_count = 0
    total_steps = 0

    for item in data:
        pred = item.get("final_answer", "").strip()
        gold = item.get("gold_answer", "").strip()
        steps = item.get("steps", [])
        
        # 스텝 수 누적
        total_steps += len(steps)
        
        # 응답 실패 (빈 문자열) 체크
        if not pred:
            empty_answer_count += 1
            continue  # 빈 답변은 EM/F1 계산에서 0점 처리됨과 동일하므로 넘어감 (F1 계산 시 오류 방지)
            
        # 1. EM & F1
        if exact_match_score(pred, gold):
            em_count += 1
        f1_sum += f1_score(pred, gold)
        
        # 2. Yes/No Accuracy
        gold_norm = normalize_answer(gold)
        if gold_norm in ['yes', 'no']:
            yesno_total += 1
            # 예측 답변에 yes 또는 no가 명확히 포함되어 있는지 확인 (초기 모델은 문장형으로 답하므로 in 사용)
            pred_norm = normalize_answer(pred)
            if gold_norm == "yes" and "yes" in pred_norm.split():
                yesno_correct += 1
            elif gold_norm == "no" and "no" in pred_norm.split():
                yesno_correct += 1

    # 결과 출력
    print("=" * 50)
    print("🛠️ Early MA-RAG (Baseline) Evaluation Report 🛠️")
    print("=" * 50)
    print(f"Total Samples      : {total_samples}")
    print(f"Empty Answers      : {empty_answer_count} ({(empty_answer_count/total_samples)*100:.1f}% Failure Rate)")
    
    print("-" * 50)
    print("[1] Accuracy Metrics (Strict String Match)")
    print(f"  Exact Match (EM) : {(em_count / total_samples) * 100:.2f}% ({em_count}/{total_samples})")
    print(f"  Avg F1 Score     : {(f1_sum / total_samples) * 100:.2f}%")
    if yesno_total > 0:
        print(f"  Yes/No Accuracy  : {(yesno_correct / yesno_total) * 100:.2f}% ({yesno_correct}/{yesno_total})")
    
    print("-" * 50)
    print("[2] Search Behavior")
    print(f"  Avg Search Steps : {total_steps / total_samples:.2f} steps/query")
    print("=" * 50)
    print("* Note: EM score might be artificially low due to the model's verbose, sentence-style answers.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, default="bs_logs.json", help="Path to the early log JSON file")
    args = parser.parse_args()
    
    evaluate_early_logs(args.log)