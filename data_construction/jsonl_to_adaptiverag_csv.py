"""Convert variation JSONL files into the CSV layout the baselines read.

Input  : {QR_ROOT}/processed_data/{dataset}/test_subsampled_{variation}.jsonl
Output : {QR_ROOT}/baselines/AdaRAGUE/data/adaptive_rag_{dataset}/test_{name}.csv

The variation text (`{variation}_variation`) becomes the `question_text` column,
so every runner sees the rewritten query while gold answers and contexts stay
untouched.

Examples
--------
    python jsonl_to_adaptiverag_csv.py --variations grammar_error
    python jsonl_to_adaptiverag_csv.py --datasets squad --variations formality readability
"""
import os as _os

# Project root. Override with:  export QR_ROOT=/path/to/QueryRobustness
QR_ROOT = _os.environ.get(
    "QR_ROOT",
    _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")),
)

import argparse
import csv
import json
import os

DATASETS = ["squad", "triviaqa", "natural_questions", "hotpotqa",
            "2wikimultihopqa", "musique"]

VARIATIONS = ["formality", "readability", "declarative", "imperative",
              "grammar_error", "spelling_error", "keyboard_error", "ocr_error"]

# variation -> csv suffix expected by the baseline configs
CSV_NAME = {"grammar_error": "grammar"}

DESIRED_COLUMNS = [
    "qid",
    "question_text",
    "answer_objects",
    "context",
    "reference",
    "rag_one_retrieve",
    "no_retrieve",
]


def to_serializable(obj):
    """Serialize for CSV while preserving non-ASCII characters."""
    return json.dumps(obj, ensure_ascii=False)


def extract_reference(record):
    """Fall back to answers_objects[*].spans when the record has no reference key."""
    if "reference" in record and record["reference"] not in (None, ""):
        return record["reference"]
    # note: the input JSON key is answers_objects
    spans = []
    for ans in record.get("answers_objects", []):
        for s in ans.get("spans", []):
            spans.append(s)
    return spans


def normalize_row(record, variation):
    """Convert one input JSON record into one CSV row."""
    reference = extract_reference(record)
    return {
        "qid": record.get("question_id", ""),
        "question_text": record.get(f"{variation}_variation", ""),
        # the CSV column is answer_objects while the input key is answers_objects
        "answer_objects": to_serializable(record.get("answers_objects", [])),
        "context": to_serializable(record.get("contexts", [])),
        "reference": (reference if isinstance(reference, str)
                      else to_serializable(reference)),
        "rag_one_retrieve": record.get("rag_one_retrieve", ""),
        "no_retrieve": record.get("no_retrieve", ""),
    }


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def convert(dataset, variation, data_dir, out_dir):
    src = f"{data_dir}/{dataset}/test_subsampled_{variation}.jsonl"
    dst = f"{out_dir}/adaptive_rag_{dataset}/test_{CSV_NAME.get(variation, variation)}.csv"
    if not os.path.exists(src):
        print(f"[skip] {src} not found")
        return
    rows = [normalize_row(rec, variation) for rec in load_jsonl(src)]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # write the CSV (commas/newlines inside fields are quoted)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DESIRED_COLUMNS,
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {dataset}/{variation}: {len(rows)} rows -> {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--variations", nargs="+", default=VARIATIONS, choices=VARIATIONS)
    ap.add_argument("--data_dir", default=f"{QR_ROOT}/processed_data")
    ap.add_argument("--out_dir", default=f"{QR_ROOT}/baselines/AdaRAGUE/data")
    args = ap.parse_args()

    for dataset in args.datasets:
        for variation in args.variations:
            convert(dataset, variation, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
