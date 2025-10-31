from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from .schema import LabelInstruction, ModelConfig, LabeledRow, LLMResponse
from .prompts import SYSTEM, user_prompt
from .llm_client import OpenAIClient, prompt_hash
from .self_consistency import vote_single_label


class Labeler:
    def __init__(
        self, label_instr: LabelInstruction, model_cfg: ModelConfig, debug: bool = False
    ):
        self.label_instr = label_instr
        self.model_cfg = model_cfg
        self.debug = debug
        if model_cfg.provider == "openai":
            self.client = OpenAIClient(model=model_cfg.model)
        else:
            raise NotImplementedError(f"Provider {model_cfg.provider} not implemented")
        # simple in-memory cache keyed by prompt hash
        self.cache: Dict[str, Dict[str, Any]] = {}

    def _ask_once(
        self, text: str
    ) -> Tuple[LLMResponse, Dict[str, Any], str, Dict[str, Any]]:
        up = user_prompt(
            text,
            self.label_instr.labels,
            self.label_instr.multi_label,
            self.label_instr.require_confidence,
        )
        ph = prompt_hash(SYSTEM, up, self.model_cfg.model)
        if ph in self.cache:
            cached = self.cache[ph]
            return LLMResponse(**cached["parsed"]), cached["raw"], ph, {"cached": True}

        res = self.client.chat(
            system=SYSTEM,
            user=up,
            temperature=self.model_cfg.temperature,
            top_p=self.model_cfg.top_p,
            seed=self.model_cfg.seed,
        )
        content = res.content.strip()
        if self.debug:
            print("[DEBUG][LLM][labeler] raw:")
            print(content)
        # Handle accidental code fences
        if content.startswith("```"):
            content = content.strip("`\n ")
            if content.startswith("json"):
                content = content[4:].lstrip()

        data = json.loads(content)
        parsed = LLMResponse(**data)
        meta = {
            "usage": res.usage,
            "provider": self.model_cfg.provider,
            "model": self.model_cfg.model,
        }
        self.cache[ph] = {"parsed": parsed.model_dump(), "raw": data, "meta": meta}
        return parsed, data, ph, meta

    def label_text(self, text: str, self_consistency: int = 1) -> LabeledRow:
        runs: List[LLMResponse] = []
        raws: List[Dict[str, Any]] = []
        metas: List[Dict[str, Any]] = []

        k = max(1, int(self_consistency))
        for _ in range(k):
            parsed, raw, ph, meta = self._ask_once(text)
            runs.append(parsed)
            raws.append(raw)
            metas.append(meta)

        if k == 1:
            final = runs[0]
            raw0 = raws[0]
        else:
            if not self.label_instr.multi_label:
                # vote on the single label across runs
                voted = vote_single_label([r.predicted_labels for r in runs])
                # pick the first run that matches the vote to keep its confidences/rationale
                chosen_idx = next(
                    i
                    for i, r in enumerate(runs)
                    if r.predicted_labels and r.predicted_labels[0] == voted
                )
                final = runs[chosen_idx]
                raw0 = raws[chosen_idx]
            else:
                # Simple multi-label aggregation:
                # choose the most common exact label set; keep the first run with that set
                from collections import Counter

                tuples = [tuple(r.predicted_labels) for r in runs]
                best_tuple = Counter(tuples).most_common(1)[0][0]
                chosen_idx = next(i for i, t in enumerate(tuples) if t == best_tuple)
                final = runs[chosen_idx]
                raw0 = raws[chosen_idx]

        return LabeledRow(
            idx=-1, text=text, response=final, raw_json=raw0, prompt_hash=ph
        )

    def label_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str,
        self_consistency: int = 1,
        max_rows: int | None = None,
    ) -> pd.DataFrame:
        if text_col not in df.columns:
            raise ValueError(
                f"Text column '{text_col}' not found in dataset. Have {list(df.columns)}"
            )

        it = df.itertuples(index=True)
        if max_rows is not None:
            it = list(it)[:max_rows]

        out_rows: List[Dict[str, Any]] = []
        total = len(df) if max_rows is None else min(max_rows, len(df))
        for row in tqdm(it, total=total):
            text = str(getattr(row, text_col))
            lr = self.label_text(text, self_consistency=self_consistency)
            out_rows.append(
                {
                    "idx": row.Index,
                    "text": text,
                    "predicted_labels": lr.response.predicted_labels,
                    "confidences": lr.response.confidences,
                    "rationale": lr.response.rationale,
                    "raw_json": lr.raw_json,
                    "prompt_hash": lr.prompt_hash,
                }
            )

        return pd.DataFrame(out_rows)
