from __future__ import annotations
from collections import Counter
from typing import List


def vote_single_label(candidates: List[List[str]]) -> str:
    """
    candidates: list of runs, each like ["label"] (single-label mode)
    Returns the label with majority vote (break ties by first-most-common).
    """
    flat = [c[0] for c in candidates if c]
    if not flat:
        raise ValueError("No candidate labels to vote on.")
    return Counter(flat).most_common(1)[0][0]


def average_confidences(conf_lists: List[List[float]]) -> List[float]:
    """
    Average confidence lists element-wise.
    Assumes all lists have the same length and alignment.
    """
    if not conf_lists:
        return []
    n = len(conf_lists)
    lens = {len(x) for x in conf_lists}
    if len(lens) != 1:
        raise ValueError("Confidence lists must be the same length to average.")
    return [sum(vals) / n for vals in zip(*conf_lists)]
