# Contriever-based RAG configurations

Config files for the **dense-retrieval** variant of the experiments: wo-RAG (NR),
SR-RAG (SR) and DRAGIN (LB) served by a Contriever FAISS index instead of the
BM25/Elasticsearch index used in the main configs
(`dragin/Llama_3.1_8B_Instruct/`, `dragin/qwq32b/`).

Both **Llama-3.1-8B-Instruct** and **QwQ-32B** are supported.

## Contents

| File | Purpose |
| --- | --- |
| `generate_configs.py` | Generates one JSON config per (model, dataset, variation, method). |
| `run_batch.sh` | Runs inference for a dataset/method (or everything) over all variations. |
| `run_single_example.sh` | Runs one config and evaluates the resulting output directory. |
| `{DRAGIN,SR-RAG,wo-RAG}_original.json` | Reference templates showing every field. |

The FAISS index itself is built by `../src/build_contriever_index.py` and is
expected at:

```
<QR_ROOT>/indexes/contriever_psgs_w100_fp16_stream/index.faiss
<QR_ROOT>/indexes/contriever_psgs_w100_fp16_stream/meta.jsonl
```

## Usage

### Step 0 — generate the configs (once)

```bash
cd <QR_ROOT>/baselines/AdaRAGUE/dragin/configs_contriever

python generate_configs.py                                   # all models / datasets / variations
python generate_configs.py --model llama                     # Llama only
python generate_configs.py --model qwq --dataset squad       # one model, one dataset
python generate_configs.py --model qwq --dataset hotpotqa --variation original
```

Generated layout:

```
configs_contriever/
├── Llama_3.1_8B_Instruct_dense/
│   └── adaptive_rag_squad/
│       ├── SR-RAG_original.json
│       ├── DRAGIN_original.json
│       └── ...
└── qwq32b_dense/
    └── ...
```

### Step 1 — run

```bash
# one config
cd <QR_ROOT>/baselines/AdaRAGUE/dragin/src
python main_newstore.py -c ../configs_contriever/Llama_3.1_8B_Instruct_dense/adaptive_rag_squad/SR-RAG_original.json

# a whole dataset / method / all variations
cd ../configs_contriever
./run_batch.sh squad DRAGIN
./run_batch.sh all all
```

### Step 2 — evaluate

```bash
cd <QR_ROOT>/baselines/AdaRAGUE/dragin/src
python evaluate.py --dir ../result_contriever_dense/Llama_3.1_8B_Instruct_adaptive_rag_squad/sr-rag_original
```

`evaluate.py` writes `details.txt` (per-question EM / F1 / Accuracy) and
`result.tsv` (the run averages, including the retrieval/generation counters) into
the run directory.

## Key config fields

**Model**

- Llama: `"model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"`
- QwQ: `"model_name_or_path": "Qwen/QwQ-32B"`

**Retriever**

| Field | Value |
| --- | --- |
| `retriever` | `"Contriever"` |
| `contriever_model_name_or_path` | `"facebook/contriever"` |
| `contriever_faiss_index_path` | FAISS index file |
| `contriever_meta_file_path` | passage metadata file |
| `contriever_max_query_length` | 256 |
| `retrieve_topk` | 3 |

**DRAGIN only**

| Field | Value | Meaning |
| --- | --- | --- |
| `query_formulation` | `"real_words"` | build the follow-up query from real words |
| `retrieve_keep_top_k` | 25 | number of tokens kept |
| `hallucination_threshold` | 0.6 | attention threshold that triggers retrieval |
| `check_real_words` | true | restrict the query to real words |

**Data**

- `data_path`: `<QR_ROOT>/baselines/AdaRAGUE/data/adaptive_rag_{dataset}/test_{variation}.csv`
- `dataset`: `squad`, `nq`, `triviaqa`, `hotpotqa`, `2wikimultihopqa`, `musique`
- `fewshot`: 6 — number of few-shot examples
- `sample`: number of questions to evaluate

## Output layout

```
result_contriever_dense/{model}_adaptive_rag_{dataset}/
├── sr-rag_original/
│   ├── config.json
│   ├── output.txt      # raw generations
│   ├── details.txt     # per-question EM / F1 / Accuracy
│   └── result.tsv      # aggregated metrics
├── dragin_formality/
└── ...
```

## Variation → data file

| Variation | File |
| --- | --- |
| `original` | `test.csv` |
| `formality` | `test_formality.csv` |
| `readability` | `test_readability.csv` |
| `declarative` | `test_declarative.csv` |
| `imperative` | `test_imperative.csv` |
| `grammar` | `test_grammar.csv` (or `test_grammar2.csv`) |
| `spelling_error` | `test_spelling_error.csv` |
| `keyboard_error` | `test_keyboard_error.csv` (some datasets only) |
| `human` | `test_human_100.csv` |
