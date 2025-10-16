from textwrap import dedent
from typing import List

SYSTEM = dedent(
    """
    You are a precise data labeling assistant. You will classify short texts into one or more labels
    from a fixed candidate set. Follow the rules strictly and return ONLY valid JSON per the schema.

    Rules:
    - Use ONLY labels from the provided list; do NOT invent new labels.
    - If multi_label is false, return exactly one label (the best fit).
    - If confidences are requested, provide a float in [0,1] per label, same order, summing to 1.0 for single-label.
    - Be concise; keep rationale to <= 70 words.
    """
)


def user_prompt(text: str, labels: List[str], multi_label: bool, require_confidence: bool) -> str:
    return dedent(
        f"""
        Text to label:
        ---
        {text}
        ---

        Candidate labels: {labels}
        multi_label: {multi_label}
        require_confidence: {require_confidence}

        Return JSON with keys: predicted_labels (list[str]), confidences (list[float]|null), rationale (str|null).
        Example outputs:
        Single-label:
        {{"predicted_labels": ["delays"], "confidences": [1.0], "rationale": "..."}}
        Multi-label:
        {{"predicted_labels": ["service_quality","cleanliness"], "confidences": [0.6,0.4], "rationale": "..."}}
        """
    ).strip()


def dual_user_prompt(
    text: str,
    labels1: List[str],
    labels2: List[str],
    multi1: bool,
    multi2: bool,
) -> str:
    """Prompt for single-call, dual-label extraction.

    Asks the model to return JSON with keys: label_1 (list[str]) and label_2 (list[str]).
    For single-label cases, return a 1-length list.
    """
    return dedent(
        f"""
        You will assign two sets of labels to the same review in a single response.

        Review:
        ---
        {text}
        ---

        First label set (label_1): {labels1}
        - multi_label: {multi1}

        Second label set (label_2): {labels2}
        - multi_label: {multi2}

        Return ONLY JSON with keys label_1 and label_2 as lists of strings from the given sets.
        If multi_label is false, each list must contain exactly one label.
        Example (single label each):
        {{"label_1": ["Customer Service"], "label_2": ["Complaint"]}}
        Example (multi-label for label_1):
        {{"label_1": ["Customer Service","Baggage Issues"], "label_2": ["Suggestion"]}}
        """
    ).strip()
