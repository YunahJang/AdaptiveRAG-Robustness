"""
Build the public release of the QueryRobustness query set.

Sources = exactly the files consumed by the experiment configs
(baselines/*/configs*/**.json -> data_path), i.e.
  baselines/AdaRAGUE/data/adaptive_rag_{dataset}/
      test.csv              original queries (500)
      test_formality.csv    \
      test_readability.csv   |
      test_declarative.csv   |  LLM-generated variations (500 each)
      test_imperative.csv    |
      test_spelling_error.csv|
      test_grammar.csv      /   (nlpaug grammar errors)
      test_human_100.csv        human rewrites (100 qids, 1-4 annotators each)

Output: hf_release/final_{dataset}.jsonl  +  hf_release/final_all.jsonl
"""

import os as _os
# Project root. Override with:  export QR_ROOT=/path/to/QueryRobustness
QR_ROOT = _os.environ.get(
    "QR_ROOT",
    _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")),
)

import ast, json, os
import pandas as pd

ROOT = f"{QR_ROOT}"
SRC  = os.path.join(ROOT, "baselines/AdaRAGUE/data")
OUT  = os.path.join(ROOT, "hf_release")

DATASETS = ["squad", "natural_questions", "triviaqa",
            "2wikimultihopqa", "hotpotqa", "musique"]

# variation name -> source csv  (order = paper order)
VARIATIONS = {
    "formality":      "test_formality.csv",
    "readability":    "test_readability.csv",
    "declarative":    "test_declarative.csv",
    "imperative":     "test_imperative.csv",
    "spelling_error": "test_spelling_error.csv",
    "grammar_error":  "test_grammar.csv",
}
ORIGINAL_FILE = "test.csv"
HUMAN_FILE    = "test_human_100.csv"


def read_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df


def parse_reference(x):
    """'[\"a\", \"b\"]' -> ['a','b']"""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return [str(v) for v in x]
    try:
        v = ast.literal_eval(str(x))
    except (ValueError, SyntaxError):
        return [str(x).strip()]
    if isinstance(v, (list, tuple, set)):
        return [str(i) for i in v]
    return [str(v)]


def clean(x):
    return None if pd.isna(x) else str(x).strip()


def build_dataset(ds):
    d = os.path.join(SRC, f"adaptive_rag_{ds}")

    orig = read_csv(os.path.join(d, ORIGINAL_FILE))
    records = {}
    for _, r in orig.iterrows():
        records[r["qid"]] = {
            "dataset": ds,
            "qid": r["qid"],
            "question": clean(r["question_text"]),
            "answers": parse_reference(r["reference"]),
            "variations": {},
            "human_rewrites": [],
            "has_human": False,
        }

    for var, fname in VARIATIONS.items():
        df = read_csv(os.path.join(d, fname))
        assert set(df["qid"]) == set(records), f"{ds}/{fname}: qid mismatch"
        for _, r in df.iterrows():
            records[r["qid"]]["variations"][var] = clean(r["question_text"])

    hum = read_csv(os.path.join(d, HUMAN_FILE))
    hum = hum.drop_duplicates(subset=["qid", "question_text"])  # a few exact re-entries
    for _, r in hum.iterrows():
        qid = r["qid"]
        assert qid in records, f"{ds}: human qid {qid} not in original"
        ann = r.get("annotator")
        records[qid]["human_rewrites"].append({
            "annotator_id": None if pd.isna(ann) else str(int(float(ann))),
            "question": clean(r["question_text"]),
        })
        records[qid]["has_human"] = True

    # stable order: original csv order, annotators sorted
    out = []
    for qid in orig["qid"]:
        rec = records[qid]
        rec["variations"] = {v: rec["variations"][v] for v in VARIATIONS}
        rec["human_rewrites"].sort(key=lambda h: (h["annotator_id"] is None,
                                                  h["annotator_id"] or ""))
        rec["num_human_rewrites"] = len(rec["human_rewrites"])
        out.append(rec)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    for ds in DATASETS:
        rows = build_dataset(ds)
        p = os.path.join(OUT, f"final_{ds}.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_h = sum(r["has_human"] for r in rows)
        n_r = sum(r["num_human_rewrites"] for r in rows)
        print(f"{ds:18s} {len(rows):4d} questions | human qids {n_h:3d} | human rewrites {n_r:3d} -> {p}")
        all_rows.extend(rows)

    p = os.path.join(OUT, "final_all.jsonl")
    with open(p, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{'ALL':18s} {len(all_rows):4d} questions -> {p}")


if __name__ == "__main__":
    main()
