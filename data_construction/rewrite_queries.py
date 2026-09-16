"""Generate query variations for a QA dataset.

Three families of variations are produced here (grammar errors live in
`rewrite_queries_grammarerror.py`, which additionally validates the output with
LanguageTool):

  style     : formality, readability, politeness   (LLM rewrite, prompts.py)
  sentence  : declarative, imperative              (LLM rewrite, prompts.py)
  error     : keyboard_error, ocr_error, spelling_error   (nlpaug, no LLM)

Input  : {QR_ROOT}/processed_data/{dataset}/test_subsampled.jsonl
Output : {QR_ROOT}/processed_data/{dataset}/test_subsampled_{variation}.jsonl
         (same records, plus a `{variation}_variation` field)

Examples
--------
    export OPENAI_API_KEY=...
    python rewrite_queries.py --dataset triviaqa --variations formality readability
    python rewrite_queries.py --dataset triviaqa --variations declarative imperative
    python rewrite_queries.py --dataset triviaqa --variations spelling_error ocr_error
"""
import os as _os

# Project root. Override with:  export QR_ROOT=/path/to/QueryRobustness
QR_ROOT = _os.environ.get(
    "QR_ROOT",
    _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")),
)

import argparse
import json

from models import ParallelGPT
from prompts import PROMPTS

STYLE_VARIATIONS = ["formality", "readability", "politeness"]
SENTENCE_VARIATIONS = ["declarative", "imperative"]
ERROR_VARIATIONS = ["keyboard_error", "ocr_error", "spelling_error"]
ALL_VARIATIONS = STYLE_VARIATIONS + SENTENCE_VARIATIONS + ERROR_VARIATIONS

DEFAULT_MODEL = "gpt-5-mini-2025-08-07"


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def llm_rewrite(data, system_prompt, model_name):
    """One LLM rewrite per question (temperature=1, single sample)."""
    queries = [f"Original Query: {item['question_text']}\nNew Query:" for item in data]
    model = ParallelGPT(model_name)
    output = model.generate(
        queries,
        system_prompt=system_prompt,
        temperature=1,
        num_return_sequences=1,
    )["responses"]
    return [out[0] for out in output]


def make_style(data, variation, model_name):
    prompt = "Given the original query generate one new query following the instructions.\n"
    prompt += PROMPTS[variation]["p1"]
    return llm_rewrite(data, prompt, model_name)


def make_sentence_type(data, variation, model_name):
    prompt = PROMPTS["sentence_type"]["p1"].format(sentence_type=variation)
    return llm_rewrite(data, prompt, model_name)


def make_error(data, variation):
    """Character/word level noise injected with nlpaug (no LLM call)."""
    # imported lazily: only the error variations need nlpaug
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw

    if variation == "keyboard_error":
        aug = nac.KeyboardAug()
    elif variation == "ocr_error":
        aug = nac.OcrAug()
    elif variation == "spelling_error":
        aug = naw.SpellingAug()
    else:
        raise ValueError(f"unknown error variation: {variation}")
    return aug.augment([item["question_text"] for item in data])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="squad / natural_questions / triviaqa / hotpotqa / "
                         "2wikimultihopqa / musique")
    ap.add_argument("--variations", nargs="+", default=ALL_VARIATIONS,
                    choices=ALL_VARIATIONS)
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"OpenAI model used for LLM rewrites (default: {DEFAULT_MODEL})")
    ap.add_argument("--data_dir", default=f"{QR_ROOT}/processed_data")
    args = ap.parse_args()

    dataset_path = f"{args.data_dir}/{args.dataset}/test_subsampled.jsonl"
    print(f"Processing dataset: {args.dataset}  ({dataset_path})")

    for variation in args.variations:
        data = read_jsonl(dataset_path)
        print(f"[{variation}] {len(data)} queries")

        if variation in STYLE_VARIATIONS:
            rewritten = make_style(data, variation, args.model)
        elif variation in SENTENCE_VARIATIONS:
            rewritten = make_sentence_type(data, variation, args.model)
        else:
            rewritten = make_error(data, variation)

        for idx, item in enumerate(data):
            item[f"{variation}_variation"] = rewritten[idx]

        out_path = f"{args.data_dir}/{args.dataset}/test_subsampled_{variation}.jsonl"
        write_jsonl(out_path, data)
        print(f"[{variation}] example: {rewritten[0]}")
        print(f"[{variation}] wrote {out_path}")


if __name__ == "__main__":
    main()
