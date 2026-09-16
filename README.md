# AdaptiveRAG-Robustness
Official repository for "How You Ask Matters! Adaptive RAG Robustness to Query Variations".

## Overview

Adaptive Retrieval-Augmented Generation (RAG) dynamically determines when retrieval is needed, improving both accuracy and efficiency. However, users can express the same information need in many different ways.

In this work, we systematically study the robustness of Adaptive RAG to **meaning-preserving query variations**.

We introduce a large-scale benchmark containing both **model-generated query variations** and **human-written rewrites**, and evaluate Adaptive RAG robustness along three dimensions:

- **Answer Robustness** — Does the system produce consistent and correct answers?
- **Computation Robustness** — Does computational cost remain stable across equivalent queries?
- **Retrieval Decision Robustness** — Does the system consistently decide when retrieval is necessary?

Our results reveal a substantial robustness gap: even semantically equivalent queries can lead to different retrieval decisions, retrieval trajectories, computational costs, and final answers.

## Benchmark

We evaluate Adaptive RAG across six widely used QA benchmarks.

### Single-hop QA

- Natural Questions (NQ)
- TriviaQA
- SQuAD v1.1

### Multi-hop QA

- 2WikiMultiHopQA
- HotpotQA
- MuSiQue

For each query, we consider controlled model-generated variations spanning three categories:

| Category | Variation |
| --- | --- |
| Style | Formality |
| Style | Readability |
| Sentence Type | Declarative |
| Sentence Type | Imperative |
| Error | Spelling |
| Error | Grammar |

## Adaptive RAG Methods

We evaluate two representative Adaptive RAG paradigms:

- **Logit-based retrieval control** — DRAGIN
- **Generation-based retrieval control** — Search-o1

## Models

Our main experiments use:

- Llama-3.1-8B-Instruct
- QwQ-32B

## Retrievers

We evaluate both sparse and dense retrieval:

- BM25
- Contriever

## Key Findings

Our experiments reveal several consistent robustness issues:

- Better performance does **not necessarily imply greater robustness**.
- Adaptive RAG systems are vulnerable to both **human rewrites and surface-level query errors**.
- Semantically equivalent queries can trigger different retrieval decisions and computational costs.
- Human-written queries can cause intermediate subqueries to drift from the original retrieval trajectory.
- This drift can lead to failures in retrieving relevant evidence and ultimately degrade answer accuracy.

## Repository Structure

```text
AdaptiveRAG-Robustness/
├── data_construction/        query-variation generation and dataset building
│   ├── rewrite_queries.py            style / sentence-type / error variations
│   ├── rewrite_queries_grammarerror.py   grammar errors, verified with LanguageTool
│   ├── prompts.py, models.py         rewrite prompts and the OpenAI wrapper
│   ├── jsonl_to_adaptiverag_csv.py   variation JSONL -> the CSV the runners read
│   ├── human_annotation_to_csv.py    annotator TSV export -> csv / jsonl
│   └── build_final_dataset.py        the public release files
├── baselines_overlay/        the files we changed in / added to the two baseline repos
│   ├── apply_overlay.sh              copies them over freshly cloned repos
│   ├── AdaRAGUE/                     NR / SR / DRAGIN: runners, retrievers, configs
│   └── Search-o1/                    Search-o1 driven by our local retriever
├── requirements.txt
└── README.md
```

The baseline frameworks themselves are not vendored here: they are cloned from
their original repositories and our files are copied over the clone. See
[`baselines_overlay/README.md`](baselines_overlay/README.md).

## Installation

```bash
git clone https://github.com/YunahJang/AdaptiveRAG-Robustness.git QueryRobustness
cd QueryRobustness
export QR_ROOT="$PWD"          # every script falls back to this layout anyway
pip install -r requirements.txt

export OPENAI_API_KEY=...      # LLM query rewriting
export HF_TOKEN=...            # gated Llama checkpoints
```

Then clone the two baseline frameworks and apply our overlay:

```bash
mkdir -p $QR_ROOT/baselines && cd $QR_ROOT/baselines
git clone https://github.com/s-nlp/AdaRAGUE.git
git clone https://github.com/RUC-NLPIR/Search-o1.git
$QR_ROOT/baselines_overlay/apply_overlay.sh
```

Follow each upstream README for its environment (AdaRAGUE needs spaCy +
Elasticsearch, Search-o1 needs vLLM).

All paths are derived from `$QR_ROOT`. Expected layout:

```text
$QR_ROOT/processed_data/{dataset}/test_subsampled*.jsonl   # queries + variations
$QR_ROOT/datasets/data/dpr/psgs_w100.tsv                   # retrieval corpus
$QR_ROOT/indexes/contriever_psgs_w100_fp16_stream/         # dense index
$QR_ROOT/baselines/{AdaRAGUE,Search-o1}/                   # cloned baselines
```

## Usage

### 1. Query variations

Starting point: `processed_data/{dataset}/test_subsampled.jsonl`, the 500-question
subsample of each benchmark in the Adaptive-RAG JSONL format.

```bash
cd data_construction

# LLM rewrites: style and sentence type
python rewrite_queries.py --dataset squad --variations formality readability
python rewrite_queries.py --dataset squad --variations declarative imperative

# surface errors (nlpaug, no API calls)
python rewrite_queries.py --dataset squad --variations spelling_error keyboard_error ocr_error

# grammar errors: LLM rewrite, verified with LanguageTool
python rewrite_queries_grammarerror.py

# human rewrites collected by annotators
python human_annotation_to_csv.py

# convert every variation into the CSV layout the runners read
python jsonl_to_adaptiverag_csv.py
```

Each variation file adds one field, `{variation}_variation`, beside the original
`question_text`; `jsonl_to_adaptiverag_csv.py` promotes it to `question_text` and
writes `baselines/AdaRAGUE/data/adaptive_rag_{dataset}/test_{variation}.csv`.

### 2. Retrieval setup

BM25 (used by the main configs) runs on a local Elasticsearch:

```bash
bash $QR_ROOT/baselines/AdaRAGUE/standard_retriever/run_elastic.sh
python $QR_ROOT/baselines/AdaRAGUE/dragin/prep_elastic.py \
    --data_path $QR_ROOT/datasets/data/dpr/psgs_w100.tsv --index_name wiki
```

Contriever instead builds a FAISS index:

```bash
python $QR_ROOT/baselines/AdaRAGUE/dragin/src/build_contriever_index.py
```

### 3. Inference

| Tag | Method | Runner |
| --- | --- | --- |
| NR | no retrieval | `dragin/src/main_newstore.py` with `wo-RAG_*.json` |
| SR | single retrieval | `dragin/src/main_newstore.py` with `SR-RAG_*.json` |
| LB | DRAGIN (logit-based) | `dragin/src/main_newstore.py` with `DRAGIN_*.json` |
| GB | Search-o1 (generation-based) | `Search-o1/scripts/run_search_o1_local.py` |

One config per (model, dataset, variation, method); all of them ship under
`baselines_overlay/AdaRAGUE/dragin/{Llama_3.1_8B_Instruct,qwq32b}/`.

```bash
# NR / SR / DRAGIN
cd $QR_ROOT/baselines/AdaRAGUE/dragin
python ./src/main_newstore.py -c ./Llama_3.1_8B_Instruct/adaptive_rag_squad/DRAGIN_original.json

# Search-o1
cd $QR_ROOT/baselines/Search-o1/scripts
python run_search_o1_local.py --dataset_name squad --split test \
    --model_path "meta-llama/Llama-3.1-8B-Instruct" --retriever BM25 \
    --data_path $QR_ROOT/baselines/AdaRAGUE/data/adaptive_rag_squad/test.csv \
    --style original
```

Every run gets its own directory with `config.json` and `output.txt`. Re-running
the same config creates `<name>-1`, `<name>-2`, ... so repeated runs can be
averaged.

Config files ship with a `<QR_ROOT>` placeholder in `data_path`; the runners
expand it from `$QR_ROOT`, or infer it from the config location.

### 4. Evaluation

`evaluate.py` scores one run directory: it reads `output.txt`, re-extracts the
final answer from each generation and compares it with the gold answers.

```bash
cd $QR_ROOT/baselines/AdaRAGUE/dragin

# NR / SR / DRAGIN runs
python ./src/evaluate.py --dir ./result_final_debug/Llama_3.1_8B_Instruct_adaptive_rag_squad/dragin_original

# Search-o1 runs are scored with the same evaluator
python ./src/evaluate.py --dir $QR_ROOT/baselines/Search-o1/scripts/outputs/runs.baselines/squad.llama-3.1-8b.search_o1_local.original
```

It writes two files into the run directory:

- `details.txt` — one JSON line per question: `qid`, `question`,
  `ground_truth`, `final_pred`, `EM`, `F1`, `Accuracy`
- `result.tsv` — the run averages: EM, F1, Precision, Recall, Accuracy, plus the
  retrieval/generation counters (`retrieve_count`, `generate_count`,
  `hallucinated_count`, `token_count`, `sentence_count`)

Robustness is read off these per-run scores: the same command is repeated for
every variation of a dataset, and each variation is compared against the
`original` run of the same method and model.

### Notes

- Most scripts keep their configuration in a block at the top of the file (model,
  dataset, method); edit it before running.
- The `.sh` files are SLURM job scripts from our cluster. Adapt the `#SBATCH`
  header and the conda activation line to your environment.
- The LLM rewrites call the OpenAI API (default `gpt-5-mini`) and therefore cost
  money; everything else runs locally.

## Data

The benchmark contains model-generated query variations and human-written
rewrites constructed from six QA datasets. Questions are the 500-question
subsamples (`dev_500_subsampled`) distributed by
[Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG); the gold answers are
identical across every form of a question, so any change in system output is
attributable to the query surface form alone.

`data_construction/build_final_dataset.py` assembles the release files
(one JSONL per benchmark, plus a combined file). Each record holds the original
question, the gold answers, the six model-generated variations and, for a
100-question subset per benchmark, 1–4 human rewrites. Retrieval contexts are not
included.

### Licensing and attribution

The six source benchmarks keep their own licenses, three of which carry a
ShareAlike condition:

| Dataset | License |
| --- | --- |
| [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [Natural Questions](https://ai.google.com/research/NaturalQuestions) | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [HotpotQA](https://hotpotqa.github.io/) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) |
| [MuSiQue](https://github.com/StonyBrookNLP/musique) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |

Because of the ShareAlike terms, the released query set is distributed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); attribute the
source benchmark of the config you use and indicate any modifications.

`baselines_overlay/` contains files we modified in, or added to, two MIT-licensed
projects — [AdaRAGUE](https://github.com/s-nlp/AdaRAGUE) and
[Search-o1](https://github.com/RUC-NLPIR/Search-o1). Each such file carries a
header naming its origin and license, and the upstream license texts are included
at `baselines_overlay/AdaRAGUE/LICENSE` and `baselines_overlay/Search-o1/LICENSE`.

## Citation

The official citation will be updated upon publication.

## Contact

For questions or issues, please open an issue in this repository.
