"""
export_data.py
HotpotQA validation 셋에서 N개를 추출해서
MA-RAG(pilot_dense.py)가 읽을 수 있는 JSON 형식으로 저장.

사용법:
    python export_data.py --n 100 --output pilot_data/hotpotqa_pilot_100.json
"""

# ── SSL 우회 ──────────────────────────────────────────────────────────────────
import ssl, os, warnings
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"]     = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_DATASETS_OFFLINE"] = "0"
warnings.filterwarnings("ignore")

import requests
_orig = requests.Session.request
def _no_verify(self, *a, **kw):
    kw["verify"] = False
    return _orig(self, *a, **kw)
requests.Session.request = _no_verify

import httpx
_orig_init = httpx.Client.__init__
def _unsafe_init(self, *a, **kw):
    kw["verify"] = False
    _orig_init(self, *a, **kw)
httpx.Client.__init__ = _unsafe_init

_orig_async = httpx.AsyncClient.__init__
def _unsafe_async(self, *a, **kw):
    kw["verify"] = False
    _orig_async(self, *a, **kw)
httpx.AsyncClient.__init__ = _unsafe_async
# ─────────────────────────────────────────────────────────────────────────────

import json
import argparse
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=100, help="추출할 샘플 수")
    parser.add_argument("--offset", type=int, default=0,   help="시작 인덱스")
    parser.add_argument("--output", type=str, default="pilot_data/hotpotqa_pilot_100.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    print("[export] HotpotQA validation 로드 중...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    samples = list(dataset)[args.offset : args.offset + args.n]
    print(f"[export] 추출: {len(samples)}개 (offset={args.offset})")

    # MA-RAG 형식으로 변환
    # pilot_bm25.py가 기대하는 필드:
    # question, answer, supporting_facts(title, sent_id), context(title, sentences)
    export = []
    for item in samples:
        export.append({
            "question": item["question"],
            "answer":   item["answer"],
            "supporting_facts": {
                "title":   item["supporting_facts"]["title"],
                "sent_id": item["supporting_facts"]["sent_id"],
            },
            "context": {
                "title":     item["context"]["title"],
                "sentences": item["context"]["sentences"],
            },
            "type":   item.get("type", ""),
            "level":  item.get("level", ""),
        })

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)

    print(f"[export] 저장 완료 → {args.output}")
    print(f"[export] 샘플 예시:")
    print(f"  Q: {export[0]['question']}")
    print(f"  A: {export[0]['answer']}")
    print(f"  context 문서 수: {len(export[0]['context']['title'])}")


if __name__ == "__main__":
    main()