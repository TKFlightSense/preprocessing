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
