# Part of the QueryRobustness overlay for AdaRAGUE (https://github.com/s-nlp/AdaRAGUE).
# MIT License, Copyright (c) 2025 Natural Language Processing.
# Added by this project; derived from the upstream src/main.py.
import os
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging
from pathlib import Path

from data import *
from algorithm import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def _qr_root(config_path: str) -> str:
    """Project root used to expand <QR_ROOT> placeholders inside config files.

    Order: $QR_ROOT env var, else inferred from the config path
    (<QR_ROOT>/baselines/AdaRAGUE/dragin/...).
    """
    root = os.environ.get("QR_ROOT")
    if root:
        return root
    for parent in Path(config_path).resolve().parents:
        if parent.name == "dragin":
            return str(parent.parents[2])
    return ""


def _expand_paths(cfg: dict, config_path: str) -> dict:
    root = _qr_root(config_path)
    if not root:
        return cfg
    return {k: (v.replace("<QR_ROOT>", root) if isinstance(v, str) else v)
            for k, v in cfg.items()}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no_log", action="store_true", help="Disable logging mode")
    cli_args = parser.parse_args()

    if cli_args.debug:
        import debugpy
        debugpy.listen(5679)
        print("wait for debugger")
        debugpy.wait_for_client()
        print("attached")

    enable_logging = not cli_args.no_log
    config_path = cli_args.config_path

    with open(config_path, "r") as f:
        cfg = json.load(f)
    cfg = _expand_paths(cfg, config_path)

    args = argparse.Namespace(**cfg)
    args.config_path = config_path
    args.enable_logging = enable_logging

    if "shuffle" not in args:
        args.shuffle = False
    if "use_counter" not in args:
        args.use_counter = True

    return args


def _safe_name(s: str) -> str:
    # strip characters that cannot be used in a folder name
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in s)


def find_dragin_root(config_path: str) -> Path:
    p = Path(config_path).resolve()
    for parent in [p.parent] + list(p.parents):
        if parent.name == "dragin":
            return parent
    raise RuntimeError(
        f"Cannot find 'dragin' directory in parents of config_path: {config_path}"
    )


def resolve_output_root(output_dir: str, config_path: str) -> str:
    outp = Path(output_dir)
    if outp.is_absolute():
        return str(outp)

    dragin_root = find_dragin_root(config_path)  # .../AdaRAGUE/dragin
    return str((dragin_root / outp).resolve())


def make_unique_run_dir(output_root: str, config_path: str) -> str:
    """
    Create a folder under output_root named after the config file stem (lowercased).
    When the name already exists, a -1, -2 ... suffix is appended.
    e.g. SR-RAG_readability.json -> sr-rag_readability, sr-rag_readability-1, ...
    """
    cfg_stem = Path(config_path).stem  # SR-RAG_readability
    base_name = _safe_name(cfg_stem.lower())  # sr-rag_readability

    root = Path(output_root)
    base_dir = root / base_name
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=False)
        return str(base_dir)

    for k in range(1, 10000):
        cand = root / f"{base_name}-{k}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=False)
            return str(cand)

    raise RuntimeError("Too many output directories with the same base name.")


def main():
    args = get_args()
    logger.info(f"{args}")

    output_file = None

    if args.enable_logging:
        # 1) resolve the config output_dir (may be relative) against the dragin root
        output_root = resolve_output_root(args.output_dir, args.config_path)
        os.makedirs(output_root, exist_ok=True)

        # 2) folder named after the config stem (lowercased), with -1, -2 on collision
        args.output_dir = make_unique_run_dir(output_root, args.config_path)
        logger.info(f"output dir: {args.output_dir}")

        # 3) save the config
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(args.__dict__, f, indent=4)

        # 4) create the output file
        output_file = open(os.path.join(args.output_dir, "output.txt"), "w")

    # load data
    if args.dataset == "strategyqa":
        data = StrategyQA(args.data_path)
    elif args.dataset == "2wikimultihopqa":
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset == "hotpotqa" or args.dataset == "musique":
        data = HotpotQA(args.data_path)
    elif args.dataset == "iirc":
        data = IIRC(args.data_path)
    elif args.dataset == "nq" or args.dataset == "squad" or args.dataset == "triviaqa":
        data = NaturalQuestions(args.data_path)
    else:
        raise NotImplementedError

    data.format(fewshot=args.fewshot)
    data = data.dataset

    if args.shuffle:
        data = data.shuffle()

    if args.sample != -1:
        samples = min(len(data), args.sample)
        data = data.select(range(samples))

    # select the method
    if args.method == "non-retrieval":
        model = BasicRAG(args)
    elif args.method == "single-retrieval":
        model = SingleRAG(args)
    elif args.method == "fix-length-retrieval" or args.method == "fix-sentence-retrieval":
        model = FixLengthRAG(args)
    elif args.method == "flare":
        model = FlareRAG(args)
    elif args.method == "attn_prob" or args.method == "dragin":
        model = AttnWeightRAG(args)
    elif args.method == "entity-flare":
        model = EntityFlareRAG(args)
    else:
        model = OurRAG(args)

    logger.info("start inference")

    for i in tqdm(range(len(data))):
        last_counter = copy(model.counter)
        batch = data[i]
        pred, tokens = model.inference(batch["question"], batch["demo"], batch["case"])

        pred = pred.strip()
        ret = {
            "qid": batch["qid"],
            "prediction": pred,
            "token_scores": tokens,
        }
        if args.use_counter:
            ret.update(model.counter.calc(last_counter))
        if args.enable_logging:
            output_file.write(json.dumps(ret) + "\n")

    if output_file is not None:
        output_file.close()


if __name__ == "__main__":
    main()
