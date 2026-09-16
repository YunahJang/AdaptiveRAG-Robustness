"""Turn an annotator TSV export of human query rewrites into csv + jsonl.

The annotation sheet holds one row per question with a `human_text` column; that
column replaces `question_text` (the original is kept as `question_text_orig`)
so the result can be fed to the baselines like any other variation.

Edit SAMPLE_PATH / CLEAN_CSV_PATH / JSONL_PATH below for the dataset you are
converting.
"""

import os as _os
# Project root. Override with:  export QR_ROOT=/path/to/QueryRobustness
QR_ROOT = _os.environ.get(
    "QR_ROOT",
    _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")),
)

import pandas as pd
import re, json, io, os, textwrap

SAMPLE_PATH = f"{QR_ROOT}/baselines/AdaRAGUE/data/adaptive_rag_2wikimultihopqa/test_2wiki - test_2wiki.tsv"
CLEAN_CSV_PATH = f"{QR_ROOT}/baselines/AdaRAGUE/data/adaptive_rag_2wikimultihopqa/test_human.csv"
JSONL_PATH = f"{QR_ROOT}/baselines/AdaRAGUE/data/adaptive_rag_2wikimultihopqa/test_human.jsonl"

def to_valid_json(s):
    if s is None:
        return s
    s = str(s)
    # array([...], dtype=object) -> [...]
    s = re.sub(r"array\(\s*(\[[\s\S]*?\])\s*,\s*dtype=object\s*\)", r"\1", s)
    # \' -> '
    s = s.replace("\\'", "'")
    # ' -> "
    s = s.replace("'", '"')
    # Python literals -> JSON
    s = s.replace("None", "null").replace("True", "true").replace("False", "false")
    # normalize smart quotes
    s = s.replace("“", '"').replace("”", '"')
    return s

def safe_json_loads(s):
    if pd.isna(s):
        return None
    s2 = to_valid_json(s)
    try:
        return json.loads(s2)
    except Exception:
        return s  # keep the raw text when it is not valid JSON

# 1) read the source file
df = pd.read_csv(SAMPLE_PATH, sep="\t", engine="python")

# 2) parse the JSON-like columns
json_like_cols = [c for c in ["answer_objects", "context", "reference"] if c in df.columns]
for c in json_like_cols:
    df[c] = df[c].apply(safe_json_loads)

# 3) move human_text into question_text (keeping the original)
if "question_text" in df.columns and "human_text" in df.columns:
    # keep the original question
    insert_at = df.columns.get_loc("question_text") + 1
    df.insert(insert_at, "question_text_original", df["question_text"])
    # replace question_text whenever human_text is non-empty
    def coalesce_human_into_question(row):
        ht = row["human_text"]
        if pd.isna(ht):
            return row["question_text"]
        s = str(ht).strip()
        return s if s else row["question_text"]
    df["question_text"] = df.apply(coalesce_human_into_question, axis=1)
else:
    # guard against a missing column
    pass

# 4) write the CSV
df.to_csv(CLEAN_CSV_PATH, index=False)

# 5) write the JSONL (with the updated question_text)
with open(JSONL_PATH, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        rec = {}
        for k, v in row.to_dict().items():
            try:
                json.dumps(v)  # check that the value is serializable
                rec[k] = v
            except TypeError:
                rec[k] = str(v)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# (optional) preview
# print(df.head(3)[["qid","question_text","question_text_original","human_text"]])
