"""
test_retrieval.py
ContextRetriever 검색 품질 확인
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from retriever.dense_retriever import ContextRetriever

def main():
    print("=" * 60)
    print("ContextRetriever 검색 품질 테스트")
    print("=" * 60)

    os.environ["HF_DATASETS_OFFLINE"] = "0"
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    retriever = ContextRetriever()
    hit = 0
    total = 5

    for item in list(dataset)[:total]:
        question = item["question"]
        gold_titles = item["supporting_facts"]["title"]
        context = item["context"]

        retriever.set_context(context)
        results = retriever.search(question, top_k=5)
        retrieved_titles = [r["title"] for r in results]

        found = any(t in retrieved_titles for t in gold_titles)
        status = "✅ HIT " if found else "❌ MISS"
        if found:
            hit += 1

        print(f"\n{status} | Q: {question}")
        print(f"       Gold titles:      {gold_titles}")
        print(f"       Retrieved titles: {retrieved_titles}")

    print("\n" + "=" * 60)
    print(f"결과: {hit}/{total} 히트")
    print("=" * 60)

if __name__ == "__main__":
    main()