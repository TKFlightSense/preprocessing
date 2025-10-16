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
    - 
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


def segmentation_user_prompt(
    text: str,
    labels1: List[str],
    labels2: List[str],
) -> str:
    """Prompt for single-call segmentation + dual labeling with strict spans.

    Returns JSON: {"segments": [{"start": int, "end": int, "label_1": str, "label_2": str}, ...]}
    - start/end are 0-based character offsets (Python slicing, end exclusive) into the ORIGINAL text.
    - label_1 must be from labels1; label_2 must be from labels2.
    - Use exact contiguous spans from the original text (no paraphrasing, no summarizing).
    - Always return at least one segment; if the review fits a single category, one segment covering the relevant part (possibly the whole text) is acceptable.
    """
    return dedent(
        f"""
        You will segment the following review into one or more labeled spans and also assign a sentiment/intent label per span.

        Review (index characters from 0):
        ---
        {text}
        ---

        Primary label set (label_1): {labels1}
        Secondary label set (label_2): {labels2}

        Rules:
        - Return ONLY JSON per schema.
        - Segments MUST be exact substrings using 0-based character offsets from the original review (Python slicing: [start, end)).
        - Do NOT paraphrase or summarize; pick contiguous spans from the original text.
        - label_1 must be one of the primary labels; label_2 must be one of the secondary labels.
        - Segments should be minimal spans that clearly correspond to the assigned primary label.
        - Segments may not overlap unless necessary; prefer separate segments for different categories.
        - Always return at least one segment. If only one category is relevant, return a single segment covering that part (or the whole text).

        JSON schema:
        {{
          "segments": [
            {{"start": int, "end": int, "label_1": str, "label_2": str}},
            ...
          ]
        }}

        Example output:
        {{"segments": [{{"start": 0, "end": 58, "label_1": "Customer Service", "label_2": "Complaint"}}]}}
        """
    ).strip()


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
