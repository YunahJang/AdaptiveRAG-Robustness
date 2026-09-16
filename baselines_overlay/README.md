# Baselines

The two RAG frameworks we build on are **not** vendored here. Clone them from
their original repositories and copy the files in this directory over the clone:
everything under `AdaRAGUE/` and `Search-o1/` is either a file we modified or a
file we added.

```bash
mkdir -p $QR_ROOT/baselines && cd $QR_ROOT/baselines
git clone https://github.com/s-nlp/AdaRAGUE.git
git clone https://github.com/RUC-NLPIR/Search-o1.git
$QR_ROOT/baselines_overlay/apply_overlay.sh      # copies our files over both clones
```

`apply_overlay.sh` refuses to run if a clone is missing and prints every file it
copies, so you can diff the result against the upstream checkout.

Follow each upstream README for its environment (AdaRAGUE needs spaCy +
Elasticsearch, Search-o1 needs vLLM).

## Licensing

Both upstream projects are MIT-licensed, which permits redistributing modified
copies as long as the copyright notice and the license text travel with them:

- AdaRAGUE - https://github.com/s-nlp/AdaRAGUE - MIT, (c) 2025 Natural Language
  Processing - license text in `AdaRAGUE/LICENSE`
- Search-o1 - https://github.com/RUC-NLPIR/Search-o1 - MIT, (c) 2025 Xiaoxi Li -
  license text in `Search-o1/LICENSE`

Every file in this directory that is a modified upstream file, or a new file
derived from one, carries a three-line header naming the project it comes from,
its license, and what we changed. The tables below list the same information per
file. Nothing else of the upstream repositories is redistributed here - the code
is cloned from its original location.

## What we changed

### AdaRAGUE (`dragin/`) — NR, SR and DRAGIN

| File | Status | Why |
| --- | --- | --- |
| `src/main_newstore.py` | added | Runner used for every NR/SR/DRAGIN run. Creates one output directory per config (`-1`, `-2`, ... for repeats) and expands `<QR_ROOT>` in config paths. |
| `src/main.py` | modified | Same `<QR_ROOT>` expansion for the upstream entry point. |
| `src/data.py` | modified | Reads our CSV format (`qid`, `question_text`, `answer_objects`, `reference`) and the answer-extraction rule. |
| `src/evaluate.py` | modified | Per-question EM / F1 / Accuracy into `details.txt`, aggregate into `result.tsv`. |
| `src/retriever.py` | modified | BM25 retriever adjustments. |
| `src/dense_retriever.py` | added | Contriever retriever with a GPU/CPU fallback. |
| `src/build_contriever_index.py` | added | Streams `psgs_w100.tsv` into a FAISS index (resumable). |
| `src/algorithm/{base_rag,generator,rag_single,rag_dragin}.py` | modified | Attention-based DRAGIN bookkeeping and the counters we report. |
| `prep_elastic.py` | as upstream, kept here | Builds the Elasticsearch BM25 index. |
| `Llama_3.1_8B_Instruct/`, `qwq32b/` | added | The configs behind the reported numbers: 3 methods x 8 variations x 6 datasets per model (BM25). |
| `configs_contriever/` | added | Dense-retrieval variant plus its config generator; see its README. |
| `bash_contriever_dragin_full.sh` | added | Example SLURM sweep. |
| `standard_retriever/run_elastic.sh` | modified | Starts the local Elasticsearch server. |

Config files ship with a `<QR_ROOT>` placeholder in `data_path`; the runners
expand it from `$QR_ROOT` (or infer it from the config location), so they work
unchanged after cloning.

### Search-o1 (`scripts/`) — GB

| File | Status | Why |
| --- | --- | --- |
| `run_search_o1_local.py` | added | Search-o1 against our local retriever (BM25/SGPT) instead of Bing. Produces the `runs.baselines` / `runs.qa` output directories the aggregation reads. |
| `run_searcho1_denseretriever.py` | added | Same, with the Contriever index. |
| `rag_retriever_adapter.py` | added | Adapts the DRAGIN retriever to the interface Search-o1 expects (`search`, `fetch_pages`). |
| `evaluate.py` | modified | Also writes a JSON-serializable per-question record so the runs can be scored like the AdaRAGUE ones. |
| `bash_searcho1_contriever_full.sh` | added | Example SLURM sweep. |

`run_search_o1_local.py` imports the retriever from
`baselines/AdaRAGUE/dragin/src`, so both clones must sit next to each other under
`$QR_ROOT/baselines/`.
