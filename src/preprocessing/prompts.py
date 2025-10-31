from textwrap import dedent
from typing import List

SYSTEM = dedent(
    """
    You are a precise data labeling assistant. Use ONLY labels from the provided list.
    Follow instructions strictly. Do not add extra text beyond the requested output.
    """
)


def user_prompt(
    text: str, labels: List[str], multi_label: bool, require_confidence: bool
) -> str:
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


# Removed dual-label and segmentation prompts; the project supports only
# single-label and multi-label classification modes.


def labels_only_prompt(text: str, labels: List[str]) -> str:
    """Strict, minimal prompt that requests only the best single label.

    The model must return exactly one label string from the provided list,
    with no quotes, punctuation, or extra text.
    """
    return dedent(
        f"""
        Review:
        ---
        {text}
        ---

        Candidate labels (choose exactly one):
        {labels}

        Rules:
        - Return ONLY the single best label EXACTLY as written above.
        - No explanations, no JSON, no quotes, no punctuation, no extra words.
        - Output must be a single line with only the label text.
        """
    ).strip()


def labels_only_prompt_multi(text: str, labels: List[str]) -> str:
    """Strict prompt requesting a comma-separated list of labels (up to 3).

    The model must return only labels from the list, separated by commas on a
    single line, with no explanations or extra characters.
    """
    return dedent(
        f"""
        Review:
        ---
        {text}
        ---

        Candidate labels (return one or more that apply, up to 3):
        {labels}

        Rules:
        - Return ONLY the applicable labels from the list, comma-separated, one line only.
        - Most reviews map to a single best label; return two or three labels only if clearly relevant.
        - No explanations, no JSON, no quotes, no punctuation other than commas.
        - Examples:
          baggage_issues
          baggage_issues,inflight_experience
          baggage_issues,inflight_experience,customer_service

        """
    ).strip()
