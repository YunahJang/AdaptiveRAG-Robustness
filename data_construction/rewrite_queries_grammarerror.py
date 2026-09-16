
import os as _os
# Project root. Override with:  export QR_ROOT=/path/to/QueryRobustness
QR_ROOT = _os.environ.get(
    "QR_ROOT",
    _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")),
)

import json
import re
from typing import List, Dict, Optional

import language_tool_python

from prompts import PROMPTS
from models import ParallelGPT


# -------------------------
# Normalization & checks
# -------------------------
_ws_re = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = _ws_re.sub(" ", s)
    return s


def grammar_error_count(tool: language_tool_python.LanguageTool, text: str) -> int:
    return len(tool.check(text))


def force_make_different(original: str, candidate: str) -> str:
    """
    Guarantee output differs from original (deterministic fallback).
    """
    original_n = normalize_text(original)
    cand_n = normalize_text(candidate)

    if cand_n and cand_n != original_n:
        return candidate.strip()

    base = cand_n if cand_n else original_n
    # deterministic change: ensures different from original
    return f"the the {base}".strip()


# -------------------------
# LLM generation
# -------------------------
def generate_queries(
    model: ParallelGPT,
    queries: List[str],
    system_prompt: str,
    temperature: float = 1.0,  # must stay 1.0
) -> List[str]:
    prompts = [f"Original Query: {q}\nNew Query:" for q in queries]
    outputs = model.generate(
        prompts,
        system_prompt=system_prompt,
        temperature=temperature,
        num_return_sequences=1,
    )["responses"]
    return [out[0].strip() for out in outputs]


def ensure_queries(
    tool: language_tool_python.LanguageTool,
    model: ParallelGPT,
    original_queries: List[str],
    base_system_prompt: str,
    max_retry: int = 30,
    grammar_attempts: int = 15,
) -> List[str]:
    """
    Attempt 1..grammar_attempts: accept only if LanguageTool finds >=1 error.
    Attempt grammar_attempts+1..max_retry: accept if output differs from original.
    Temperature is fixed to 1.0 for all attempts.

    After retries, force outputs to differ from original deterministically.
    """
    n = len(original_queries)
    final_queries: List[Optional[str]] = [None] * n
    remaining_indices = list(range(n))
    last_try: Dict[int, str] = {}

    # Stronger prompt for later attempts (still temperature=1.0)
    # This helps break plateaus without changing temperature.
    diff_prompt = (
        base_system_prompt
        + "\n\nADDITIONAL REQUIREMENTS:\n"
          "- The new query MUST be different from the original (change wording or introduce a small mistake).\n"
          "- If you struggle to change it, insert a duplicated word like 'the the' somewhere.\n"
    )

    for attempt in range(1, max_retry + 1):
        if not remaining_indices:
            break

        if attempt <= grammar_attempts:
            sys_prompt = base_system_prompt
            mode = "LT(>=1 error)"
        else:
            sys_prompt = diff_prompt
            mode = "DIFF(original)"

        print(f"[Attempt {attempt}/{max_retry}] Remaining: {len(remaining_indices)} | Criterion: {mode}")

        batch_indices = remaining_indices[:]  # freeze order
        batch_queries = [original_queries[i] for i in batch_indices]

        generated = generate_queries(
            model=model,
            queries=batch_queries,
            system_prompt=sys_prompt,
            temperature=1.0,  # fixed
        )

        next_remaining = []
        for idx, gen_q in zip(batch_indices, generated):
            gen_q = (gen_q or "").strip()
            last_try[idx] = gen_q

            orig = original_queries[idx]
            candidate = gen_q if gen_q else orig

            if attempt <= grammar_attempts:
                if grammar_error_count(tool, candidate) > 0:
                    final_queries[idx] = candidate
                else:
                    next_remaining.append(idx)
            else:
                if normalize_text(candidate) != normalize_text(orig):
                    final_queries[idx] = candidate
                else:
                    next_remaining.append(idx)

        remaining_indices = next_remaining

    # Fallback: force any still unresolved to be different from original
    if remaining_indices:
        print(f"[Fallback] Forcing DIFFERENT output for {len(remaining_indices)} samples still unresolved after {max_retry} tries.")
    for idx in remaining_indices:
        orig = original_queries[idx]
        cand = last_try.get(idx) or ""
        final_queries[idx] = force_make_different(orig, cand)

    # Safety fill
    for i in range(n):
        if final_queries[i] is None:
            final_queries[i] = force_make_different(original_queries[i], "")

    return final_queries


# -------------------------
# Main
# -------------------------
def main():
    data_list = [
        "2wikimultihopqa",
        "natural_questions",
        "hotpotqa",
        "musique",
        "squad",
        "triviaqa",
    ]

    MODEL_NAME = "gpt-5-mini-2025-08-07"  # any chat model works here
    STYLE = "grammar_error2"
    MAX_RETRY = 30
    GRAMMAR_ATTEMPTS = 15

    tool = language_tool_python.LanguageTool("en-US")
    try:
        for DATASET_NAME in data_list:
            dataset_path = f"{QR_ROOT}/processed_data/{DATASET_NAME}/test_subsampled.jsonl"
            query_output_path = f"{QR_ROOT}/processed_data/{DATASET_NAME}/test_subsampled_{STYLE}.jsonl"

            print(f"\n=== Dataset: {DATASET_NAME} ===")
            print("Loading:", dataset_path)

            with open(dataset_path, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            original_queries = [item["question_text"] for item in data]

            base_prompt = "Given the original query generate one new query following the instructions.\n"
            base_prompt += PROMPTS["grammar_error"]["p1"]

            print("Initializing model:", MODEL_NAME)
            model = ParallelGPT(MODEL_NAME)

            noisy_queries = ensure_queries(
                tool=tool,
                model=model,
                original_queries=original_queries,
                base_system_prompt=base_prompt,
                max_retry=MAX_RETRY,
                grammar_attempts=GRAMMAR_ATTEMPTS,
            )

            # Attach & save
            for i, item in enumerate(data):
                item[f"{STYLE}_variation"] = noisy_queries[i]

            # Sanity check: all must differ from original after this pipeline
            same_cnt = sum(
                1 for orig, newq in zip(original_queries, noisy_queries)
                if normalize_text(orig) == normalize_text(newq)
            )
            if same_cnt > 0:
                raise RuntimeError(f"[{DATASET_NAME}] {same_cnt} queries are still identical to original (unexpected).")

            print("Saving:", query_output_path)
            with open(query_output_path, "w", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"Done. Saved to {query_output_path}")

    finally:
        # Explicit cleanup to avoid __del__ errors
        try:
            tool.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
