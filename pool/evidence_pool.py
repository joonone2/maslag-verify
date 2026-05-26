"""
pool/evidence_pool.py
Shared Evidence Pool
"""

import json
from typing import Optional


class EvidencePool:
    def __init__(self):
        self.steps: dict = {}
        self.task_type: str = "bridge"  # bridge | comparison

    def add(self, step_idx: int, sub_query: str, retrieved_titles: list, intermediate_answer: str):
        self.steps[step_idx] = {
            "sub_query": sub_query,
            "retrieved_titles": retrieved_titles,
            "intermediate_answer": intermediate_answer,
            "confidence": None,
            "score_1": None,
            "score_2": None,
            "flag": None,
        }

    def update_verification(
        self,
        step_idx: int,
        confidence: float,
        flag: str,
        score_1: float = None,
        score_2: float = None,
    ):
        if step_idx not in self.steps:
            raise KeyError(f"step_idx {step_idx} not in pool")
        self.steps[step_idx]["confidence"] = confidence
        self.steps[step_idx]["flag"] = flag
        if score_1 is not None:
            self.steps[step_idx]["score_1"] = score_1
        if score_2 is not None:
            self.steps[step_idx]["score_2"] = score_2

    def get_previous_verified(self, current_step_idx: int) -> dict:
        """verified / refined / skipped 만 반환 (bridge 컨텍스트용)"""
        valid_flags = {"verified", "refined", "skipped"}
        return {
            idx: step
            for idx, step in self.steps.items()
            if idx < current_step_idx and step.get("flag") in valid_flags
        }

    def get_latest_answer(self, current_step_idx: int) -> Optional[str]:
        prev_idx = current_step_idx - 1
        if prev_idx < 0 or prev_idx not in self.steps:
            return None
        return self.steps[prev_idx]["intermediate_answer"]

    def get_all_verified(self) -> dict:
        """
        Answer Agent용 - uncertain 포함 전부 반환.
        uncertain 제외시 정보 손실 발생.
        """
        valid_flags = {"verified", "refined", "skipped", "uncertain"}
        return {
            idx: step
            for idx, step in self.steps.items()
            if step.get("flag") in valid_flags
        }

    def to_log(self) -> dict:
        return dict(self.steps)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.steps, f, indent=2, ensure_ascii=False)