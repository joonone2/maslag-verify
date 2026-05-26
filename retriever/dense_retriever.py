"""
retriever/dense_retriever.py
FAISS + sentence-transformers (gte-multilingual-base) 기반 Dense Retriever

두 가지 모드:
1. ContextRetriever: distractor 설정 - 질문별 제공된 문서 안에서 검색
2. DenseRetriever: fullwiki 설정 - 전체 인덱스 검색 (나중에 사용)
"""

import os
import ssl
import pickle
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

LOCAL_MODEL_DIR = "models/gte-multilingual-base"
_DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

_model = None  # 싱글톤 - 한 번만 로드

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[Model] 로드: {LOCAL_MODEL_DIR}")
        _model = SentenceTransformer(LOCAL_MODEL_DIR, trust_remote_code=True, device=_DEVICE)
    return _model


# ─────────────────────────────────────────────
# ContextRetriever - distractor 설정
# ─────────────────────────────────────────────

class ContextRetriever:
    """
    HotpotQA distractor 설정용.
    질문마다 제공된 context 문서(10개) 안에서만 검색.
    매 질문마다 set_context()로 문서 설정 후 search().
    """

    def __init__(self):
        self.model = _get_model()
        self.docs: list = []
        self.index = None
        print(f"[ContextRetriever] 준비 완료 — device: {_DEVICE}")

    def set_context(self, context):
        """
        HotpotQA 또는 MuSiQue context를 설정.

        HotpotQA: {"title": [...], "sentences": [[...], ...]}
        MuSiQue:  [{"idx", "title", "paragraph_text", "is_supporting"}, ...]
        """
        self.docs = []

        # MuSiQue 형식 (리스트)
        if isinstance(context, list):
            for para in context:
                self.docs.append({
                    "title": para["title"],
                    "text": para["title"] + "\n" + para["paragraph_text"],
                    "is_supporting": para.get("is_supporting", False),
                })
        # HotpotQA 형식 (dict)
        else:
            titles = context["title"]
            sentences_list = context["sentences"]
            for title, sentences in zip(titles, sentences_list):
                text = " ".join(sentences)
                self.docs.append({"title": title, "text": text})

        # 문서 임베딩 + 소형 FAISS 인덱스 빌드
        texts = [d["text"] for d in self.docs]
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        dim = vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)

    def search(self, query: str, top_k: int = 5) -> list:
        if self.index is None or len(self.docs) == 0:
            return []

        top_k = min(top_k, len(self.docs))
        q_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        distances, indices = self.index.search(q_vec, top_k)
        return [self.docs[i] for i in indices[0] if 0 <= i < len(self.docs)]


# ─────────────────────────────────────────────
# DenseRetriever - fullwiki 설정 (나중에 사용)
# ─────────────────────────────────────────────

class DenseRetriever:
    def __init__(self, index_path: str, docs_path: str):
        print(f"[DenseRetriever] 인덱스 로드: {index_path}")
        self.index = faiss.read_index(index_path)

        print(f"[DenseRetriever] 문서 로드: {docs_path}")
        with open(docs_path, "rb") as f:
            self.docs: list = pickle.load(f)

        self.model = _get_model()
        print(f"[DenseRetriever] 준비 완료 — 문서 수: {len(self.docs):,}, "
              f"인덱스 벡터 수: {self.index.ntotal:,}, device: {_DEVICE}")

    def _embed(self, texts: list) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

    def search(self, query: str, top_k: int = 5) -> list:
        q_vec = self._embed([query])
        distances, indices = self.index.search(q_vec, top_k)
        return [self.docs[i] for i in indices[0] if 0 <= i < len(self.docs)]


# ─────────────────────────────────────────────
# 인덱스 빌드 (fullwiki용, 나중에 사용)
# ─────────────────────────────────────────────

def build_index(output_dir: str = "index", batch_size: int = 64):
    os.environ["HF_DATASETS_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, "hotpot.index")
    docs_path  = os.path.join(output_dir, "hotpot_docs.pkl")

    print("[build_index] HotpotQA validation 로드...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")

    seen_titles: set = set()
    docs: list = []

    for item in dataset:
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            if title in seen_titles:
                continue
            seen_titles.add(title)
            docs.append({"title": title, "text": " ".join(sentences)})

    print(f"[build_index] 수집된 고유 문서 수: {len(docs):,}")

    model = _get_model()
    all_vecs = model.encode(
        [d["text"] for d in docs],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(all_vecs.shape[1])
    index.add(all_vecs)
    faiss.write_index(index, index_path)
    print(f"[build_index] 저장: {index_path}")

    with open(docs_path, "wb") as f:
        pickle.dump(docs, f)
    print(f"[build_index] 저장: {docs_path}")
    print("[build_index] 완료!")