#!/usr/bin/env python3
# Part of the QueryRobustness overlay for AdaRAGUE (https://github.com/s-nlp/AdaRAGUE).
# MIT License, Copyright (c) 2025 Natural Language Processing.
# Added by this project.
"""
Generate Contriever-based SR-RAG / DRAGIN config files.
Supports the Llama-3.1-8B and QwQ-32B models.
"""

import os as _os
# Project root. Override with:  export QR_ROOT=/path/to/QueryRobustness
QR_ROOT = _os.environ.get(
    "QR_ROOT",
    _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "..")),
)


import json
import os
import argparse
from pathlib import Path

# ========================================
# Settings
# ========================================
BASE_DIR = Path(__file__).parent
DATA_BASE = f"{QR_ROOT}/baselines/AdaRAGUE/data"
INDEX_DIR = f"{QR_ROOT}/indexes/contriever_psgs_w100_fp16_stream"

# Models
MODELS = {
    "llama": {
        "name": "meta-llama/Llama-3.1-8B-Instruct",
        "output_prefix": "Llama_3.1_8B_Instruct",
        "config_dir": "Llama_3.1_8B_Instruct_dense"
    },
    "qwq": {
        "name": "Qwen/QwQ-32B",
        "output_prefix": "qwq32b",
        "config_dir": "qwq32b_dense"
    }
}

# Dataset mapping (config dataset key -> data directory name)
DATASETS = {
    "squad": "squad",
    "nq": "natural_questions",
    "triviaqa": "triviaqa",
    "hotpotqa": "hotpotqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "musique": "musique"
}

# Variation mapping (variation name -> adaptive_rag file name)
VARIATION_MAPPING = {
    "original": "",  # test.csv
    "formality": "formality",
    "readability": "readability",
    "declarative": "declarative",
    "imperative": "imperative",
    "grammar": "grammar",  # grammar2 or grammar
    "spelling_error": "spelling_error",
    "human": "human"  # or human_100
}

# ========================================
# Templates
# ========================================
def get_sr_rag_template(model_name):
    return {
        "model_name_or_path": model_name,
        "method": "single-retrieval",
        "dataset": "",
        "data_path": "",
        "fewshot": 6,
        "sample": 1000,
        "shuffle": False,
        "generate_max_length": 64,
        "query_formulation": "direct",
        "output_dir": "",
        "retriever": "Contriever",
        "contriever_model_name_or_path": "facebook/contriever",
        "contriever_faiss_index_path": f"{INDEX_DIR}/index.faiss",
        "contriever_meta_file_path": f"{INDEX_DIR}/meta.jsonl",
        "contriever_max_query_length": 256,
        "retrieve_topk": 3,
        "use_counter": True
    }

def get_dragin_template(model_name):
    return {
        "model_name_or_path": model_name,
        "method": "dragin",
        "dataset": "",
        "data_path": "",
        "fewshot": 6,
        "sample": 1000,
        "shuffle": False,
        "generate_max_length": 64,
        "query_formulation": "real_words",
        "retrieve_keep_top_k": 25,
        "output_dir": "",
        "retriever": "Contriever",
        "contriever_model_name_or_path": "facebook/contriever",
        "contriever_faiss_index_path": f"{INDEX_DIR}/index.faiss",
        "contriever_meta_file_path": f"{INDEX_DIR}/meta.jsonl",
        "contriever_max_query_length": 256,
        "retrieve_topk": 3,
        "hallucination_threshold": 0.6,
        "check_real_words": True,
        "use_counter": True
    }

# ========================================
# Config generation
# ========================================
def find_data_file(dataset_name, variation):
    """Locate the adaptive_rag data file."""
    data_dir = f"{DATA_BASE}/adaptive_rag_{dataset_name}"

    # file name per variation
    if variation == "original":
        candidates = ["test.csv"]
    elif variation == "human":
        candidates = ["test_human_100.csv"]
    else:
        variation_suffix = VARIATION_MAPPING.get(variation, variation)
        candidates = [f"test_{variation_suffix}.csv"]

    # check that the file exists
    for candidate in candidates:
        filepath = f"{data_dir}/{candidate}"
        if os.path.exists(filepath):
            return filepath

    return None

def generate_config(model_key, method, dataset_key, variation, output_subdir=None):
    """Generate a single config file."""

    model_info = MODELS[model_key]
    dataset_name = DATASETS[dataset_key]

    # locate the data file
    data_file = find_data_file(dataset_name, variation)
    if not data_file:
        print(f"[SKIP] Data file not found for {dataset_name}/{variation}")
        return None

    # pick the template
    if method == "SR-RAG":
        config = get_sr_rag_template(model_info["name"])
    elif method == "DRAGIN":
        config = get_dragin_template(model_info["name"])
    else:
        raise ValueError(f"Unknown method: {method}")

    # update the settings
    config["dataset"] = dataset_key
    config["data_path"] = data_file
    config["output_dir"] = f"./result_contriever_dense/{model_info['output_prefix']}_adaptive_rag_{dataset_name}"

    # output directory (one sub-directory per model)
    if output_subdir:
        output_dir = BASE_DIR / output_subdir
    else:
        output_dir = BASE_DIR / model_info["config_dir"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # create one sub-directory per dataset
    dataset_dir = output_dir / f"adaptive_rag_{dataset_name}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # config file name
    config_filename = f"{method}_{variation}.json"
    config_path = dataset_dir / config_filename

    # save
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"[CREATED] {model_info['config_dir']}/adaptive_rag_{dataset_name}/{config_filename}")
    return config_path

# ========================================
# Bash script generation
# ========================================
def generate_bash_script(model_key, datasets_to_generate, variations_to_generate):
    """Generate a run + evaluate bash script for each model."""

    model_info = MODELS[model_key]
    config_dir = BASE_DIR / model_info["config_dir"]

    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name  {model_info['output_prefix']}_contriever",
        "#SBATCH --time      4-00:00:00",
        "#SBATCH -c          4",
        "#SBATCH --gpus      2",
        "#SBATCH --mem       150G",
        "",
        "# ========================================",
        f"# {model_info['name']} - Contriever SR-RAG & DRAGIN",
        "# Auto-generated by generate_configs.py",
        "# ========================================",
        "",
        "# Activate your conda installation here, e.g.:",
        "#   source \"$(conda info --base)/etc/profile.d/conda.sh\"",
        "",
        f"cd {QR_ROOT}/baselines/AdaRAGUE/dragin",
        "conda activate dragin",
        "",
    ]

    # emit one command per dataset / variation / method
    for dataset_key in datasets_to_generate.keys():
        dataset_name = DATASETS[dataset_key]
        script_lines.append(f"# ========================================")
        script_lines.append(f"# {dataset_key.upper()}")
        script_lines.append(f"# ========================================")
        script_lines.append("")

        for method in ["SR-RAG", "DRAGIN"]:
            method_lower = method.lower()  # sr-rag, dragin
            script_lines.append(f"# {dataset_name} - {method}")

            for variation in variations_to_generate:
                config_file = BASE_DIR / model_info["config_dir"] / f"adaptive_rag_{dataset_name}" / f"{method}_{variation}.json"

                if not config_file.exists():
                    continue

                config_path = str(config_file)
                output_base = f"./result_contriever_dense/{model_info['output_prefix']}_adaptive_rag_{dataset_name}"
                eval_pattern = f"{output_base}/{method_lower}_{variation}"

                script_lines.append(f"python ./src/main_newstore.py -c '{config_path}'")
                script_lines.append(f"python ./src/evaluate.py --dir \"$(ls -dt '{eval_pattern}'* 2>/dev/null | head -1)\"")
                script_lines.append("")

            script_lines.append("")

    script_lines.extend([
        "echo \"ALL EXPERIMENTS COMPLETED!\"",
        ""
    ])

    # save the bash script
    script_path = config_dir / f"run_all_{model_info['output_prefix']}.sh"
    with open(script_path, "w") as f:
        f.write("\n".join(script_lines))

    # make it executable
    os.chmod(script_path, 0o755)

    print(f"\n[CREATED BASH SCRIPT] {script_path}")
    return script_path

# ========================================
# Main
# ========================================
def main():
    parser = argparse.ArgumentParser(description="Generate Contriever config files")
    parser.add_argument("--model", choices=["llama", "qwq", "all"], default="all",
                       help="Model to generate configs for (default: all)")
    parser.add_argument("--dataset", type=str, nargs='+', default=None,
                       help="Specific dataset(s) (default: all). Can specify multiple: --dataset squad 2wikimultihopqa")
    parser.add_argument("--variation", type=str, default=None,
                       help="Specific variation (default: all)")
    args = parser.parse_args()

    print("=" * 60)
    print("Generating Contriever Config Files (Dense Retrieval)")
    print("=" * 60)

    # model selection
    if args.model == "all":
        models_to_generate = list(MODELS.keys())
    else:
        models_to_generate = [args.model]

    # dataset selection
    if args.dataset:
        datasets_to_generate = {ds: DATASETS[ds] for ds in args.dataset if ds in DATASETS}
    else:
        datasets_to_generate = DATASETS

    # variation selection
    if args.variation:
        variations_to_generate = [args.variation]
    else:
        variations_to_generate = list(VARIATION_MAPPING.keys())

    total_created = 0

    for model_key in models_to_generate:
        print(f"\n{'=' * 60}")
        print(f"Model: {MODELS[model_key]['name']}")
        print(f"{'=' * 60}")

        for dataset_key in datasets_to_generate.keys():
            print(f"\n[Dataset: {dataset_key}]")

            for variation in variations_to_generate:
                for method in ["SR-RAG", "DRAGIN"]:
                    result = generate_config(model_key, method, dataset_key, variation)
                    if result:
                        total_created += 1

    print("\n" + "=" * 60)
    print(f"Total configs created: {total_created}")
    print("=" * 60)
    print("\nConfig directories:")
    for model_key in models_to_generate:
        config_dir = BASE_DIR / MODELS[model_key]["config_dir"]
        print(f"  - {config_dir}")

    # Bash script generation
    print("\n" + "=" * 60)
    print("Generating Bash Scripts (with evaluate.py)")
    print("=" * 60)

    for model_key in models_to_generate:
        generate_bash_script(model_key, datasets_to_generate, variations_to_generate)

if __name__ == "__main__":
    main()
