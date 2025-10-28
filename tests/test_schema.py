from preprocessing.schema import LabelInstruction, LLMResponse


def test_label_instruction_bounds():
    li = LabelInstruction(labels=["a", "b", "c"], multi_label=False)
    assert li.labels[0] == "a"
    assert not li.multi_label


def test_llm_response_model():
    data = {"predicted_labels": ["a"], "confidences": [1.0], "rationale": "ok"}
    r = LLMResponse(**data)
    assert r.predicted_labels == ["a"]
    assert r.confidences == [1.0]
