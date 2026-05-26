import os
import ssl

# 학교 네트워크 SSL 우회 (반드시 다른 import보다 먼저)
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["PYTHONHTTPSVERIFY"] = "0"
ssl._create_default_https_context = ssl._create_unverified_context

# httpx 패치
import httpx
_orig_client_init = httpx.Client.__init__
_orig_async_init  = httpx.AsyncClient.__init__
def _patched_client(self, *a, **kw):
    kw["verify"] = False
    _orig_client_init(self, *a, **kw)
def _patched_async(self, *a, **kw):
    kw["verify"] = False
    _orig_async_init(self, *a, **kw)
httpx.Client.__init__      = _patched_client
httpx.AsyncClient.__init__ = _patched_async

# requests 패치
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
_orig_session_init = requests.Session.__init__
def _patched_session(self, *a, **kw):
    _orig_session_init(self, *a, **kw)
    self.verify = False
requests.Session.__init__ = _patched_session
requests.packages.urllib3.disable_warnings()

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
run.py
실험 실행 및 로그 저장

사용법:
    python run.py                          # 기본 100개
    python run.py --n 20                   # 샘플 수 지정
    python run.py --n 20 --offset 100      # 101~120번째 샘플
    python run.py --build-index            # 인덱스 먼저 빌드
"""

import os
import json
import argparse
from tqdm import tqdm

from datasets import load_dataset
from pipeline import run_pipeline
from retriever.dense_retriever import DenseRetriever

INDEX_PATH = "index/hotpot.index"
DOCS_PATH  = "index/hotpot_docs.pkl"
LOG_PATH   = "logs/results.json"


def main():
    parser = argparse.ArgumentParser(description="MAS+RAG Verification Experiment")
    parser.add_argument("--n",           type=int, default=100, help="실행할 샘플 수")
    parser.add_argument("--offset",      type=int, default=0,   help="시작 인덱스")
    parser.add_argument("--build-index",     action="store_true", help="인덱스 빌드 후 종료")
    parser.add_argument("--download-model", action="store_true", help="모델 로컬 다운로드 후 종료")
    parser.add_argument("--log-path",    type=str, default=LOG_PATH)
    parser.add_argument("--dataset",     type=str, default="hotpotqa",
                        choices=["hotpotqa", "musique"],
                        help="사용할 데이터셋 (hotpotqa / musique)")
    args = parser.parse_args()

    # ── 인덱스 빌드 모드 ─────────────────────────
    if args.download_model:
        from retriever.dense_retriever import download_model
        download_model()
        return

    if args.build_index:
        from retriever.dense_retriever import build_index
        build_index()
        return

    # ── 인덱스 존재 확인 ─────────────────────────
    if not os.path.exists(INDEX_PATH) or not os.path.exists(DOCS_PATH):
        print("인덱스 파일이 없습니다. 먼저 빌드하세요:")
        print("  python run.py --build-index")
        return

    os.makedirs("logs", exist_ok=True)

    # ── 데이터셋 로드 ────────────────────────────
    if args.dataset == "musique":
        print("[run] MuSiQue validation 로드 중...")
        raw = load_dataset("dgslibisey/MuSiQue", split="validation")
        dataset = [item for item in raw if item["answerable"]]
    else:
        print("[run] HotpotQA validation 로드 중...")
        dataset = list(load_dataset("hotpot_qa", "distractor", split="validation"))

    samples = dataset[args.offset : args.offset + args.n]
    print(f"[run] 처리할 샘플: {len(samples)}개 (offset={args.offset}, dataset={args.dataset})")

    # ── 검색기 초기화 (distractor 설정) ─────────
    from retriever.dense_retriever import ContextRetriever
    retriever = ContextRetriever()

    # ── 기존 결과 로드 (이어쓰기 지원) ──────────
    if os.path.exists(args.log_path):
        with open(args.log_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"[run] 기존 결과 {len(results)}개 로드 (이어쓰기 모드)")
    else:
        results = []

    # ── 실행 ─────────────────────────────────────
    for i, item in enumerate(tqdm(samples, desc="Processing")):
        question    = item["question"]
        gold_answer = item["answer"]
        context     = item["paragraphs"] if args.dataset == "musique" else item["context"]

        print(f"\n[{i+1}/{len(samples)}] Q: {question}")
        result = run_pipeline(
            question=question,
            gold_answer=gold_answer,
            retriever=retriever,
            context=context,
        )
        print(f"  → Predicted: {result['final_answer']}")
        print(f"  → Gold:      {gold_answer}")
        print(f"  → Time:      {result['elapsed_sec']}s")

        results.append(result)

        # 중간 저장 (매 샘플마다)
        with open(args.log_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 요약 통계 ────────────────────────────────
    _print_summary(results[-len(samples):])
    print(f"\n[run] 완료: 총 {len(results)}개 → {args.log_path}")


def _print_summary(results: list[dict]):
    """간단한 실험 요약 출력."""
    total = len(results)
    errors = sum(1 for r in results if r.get("error"))

    # Verification flag 분포
    flag_counts: dict[str, int] = {}
    for r in results:
        for step in r.get("steps", {}).values():
            if isinstance(step, dict):
                flag = step.get("flag", "unknown")
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

    print("\n" + "="*50)
    print(f"실험 요약 (n={total})")
    print(f"  에러 발생: {errors}건")
    print("  Verification flag 분포:")
    for flag, cnt in sorted(flag_counts.items()):
        print(f"    {flag:12s}: {cnt}")
    print("="*50)


if __name__ == "__main__":
    main()