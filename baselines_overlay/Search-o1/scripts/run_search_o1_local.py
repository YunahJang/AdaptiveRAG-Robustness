# Part of the QueryRobustness overlay for Search-o1 (https://github.com/RUC-NLPIR/Search-o1).
# MIT License, Copyright (c) 2025 Xiaoxi Li.
# Added by this project; derived from the upstream scripts/run_search_o1.py.

import os as _os
# Project root. Override with:  export QR_ROOT=/path/to/QueryRobustness
QR_ROOT = _os.environ.get(
    "QR_ROOT",
    _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..")),
)

import os
import json
import time
import re
from typing import Optional, List, Dict
import argparse

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

#  from evaluate import (
#     run_evaluation,
#     extract_answer
# )
from prompts import (
    get_gpqa_search_o1_instruction,
    get_math_search_o1_instruction,
    get_code_search_o1_instruction,
    get_singleqa_search_o1_instruction,
    get_multiqa_search_o1_instruction,
    get_webpage_to_reasonchain_instruction,
    get_task_instruction_openqa,
    get_task_instruction_math,
    get_task_instruction_multi_choice,
    get_task_instruction_code,
)
import sys
RAG_SRC = f"{QR_ROOT}/baselines/AdaRAGUE/dragin/src"
EVAL_SRC = f"{QR_ROOT}/baselines/Search-o1/scripts"
if RAG_SRC not in sys.path:
    sys.path.insert(0, RAG_SRC)
from retriever import BM25, SGPT
from data import *
from algorithm import *


BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

def parse_args():
    p = argparse.ArgumentParser("Run Search O1 with LOCAL retriever (BM25/SGPT)")

    # dataset & split
    p.add_argument('--dataset_name', required=True,
                   choices=['gpqa', 'math500', 'aime', 'amc', 'livecode', 'squad',
                            'nq', 'triviaqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle'])
    p.add_argument('--split', required=True,
                   choices=['test', 'diamond', 'main', 'extended'])
    p.add_argument('--data_path', required=True)
    p.add_argument('--fewshot', type=int, default=6)
    p.add_argument('--sample', type=int, default=1000)
    p.add_argument('--shuffle', type=bool, default=False)


    # search & generation config
    p.add_argument('--max_search_limit', type=int, default=10)
    p.add_argument('--max_turn', type=int, default=5)
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--max_doc_len', type=int, default=3000)

    # model
    p.add_argument('--model_path', required=True)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.8)
    p.add_argument('--top_k_sampling', type=int, default=20)
    p.add_argument('--repetition_penalty', type=float, default=None)
    p.add_argument('--max_tokens', type=int, default=32768)

    # retriever selection & settings
    p.add_argument('--retriever', type=str, required=True, choices=['BM25', 'SGPT'],
                   help="BM25 or SGPT, as in BasicRAG")
    # BM25 (Elastic)
    p.add_argument('--es_index_name', type=str, default='wiki',
                   help="BEIR ElasticSearch index name (same as in BM25Search)")
    # SGPT
    p.add_argument('--sgpt_model_name_or_path', type=str, default=None)
    p.add_argument('--sgpt_encode_file_path', type=str, default=None)
    p.add_argument('--passage_file', type=str, default=None)
    p.add_argument('--tp_size', type=int, default=None, help="Tensor parallel size. If not set, inferred from visible GPUs. If set bigger than visible GPUs, it will be clamped.")
    p.add_argument('--style', type=str, default='original')
    return p.parse_args()

def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None
def _load_eval_funcs():
    """
    Load RAG_SRC/evaluate.py directly under the name 'eval_local'.
    Falls back gracefully so the script keeps running when the import fails.
    """
    # default fallback
    def _run_eval_stub(*args, **kwargs):
        print("[WARN] run_evaluation not found; skipping evaluation.")
    def _extract_stub(text, mode=None):
        return text  # return the string unchanged

    try:
        import importlib.util
        eval_path = os.path.join(EVAL_SRC, "evaluate.py")
        spec = importlib.util.spec_from_file_location("eval_local", eval_path)
        if spec is None or spec.loader is None:
            print(f"[WARN] cannot build spec for evaluate.py at {eval_path}")
            return _run_eval_stub, _extract_stub
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        run_eval = getattr(mod, "run_evaluation", _run_eval_stub)
        extract = getattr(mod, "extract_answer", _extract_stub)
        return run_eval, extract
    except Exception as e:
        print(f"[WARN] failed to import evaluate.py: {e}")
        return _run_eval_stub, _extract_stub
def replace_recent_steps(origin_str, replace_str):
    step_pattern = re.compile(r"Step\s+(\d+):\s*")
    def parse_steps(text):
        steps = {}
        current_step_num = None
        current_content = []
        for line in text.splitlines():
            m = step_pattern.match(line)
            if m:
                if current_step_num is not None:
                    steps[current_step_num] = "\n".join(current_content).strip()
                current_step_num = int(m.group(1))
                content = line[step_pattern.end():].strip()
                current_content = [content] if content else []
            else:
                if current_step_num is not None:
                    current_content.append(line)
        if current_step_num is not None:
            steps[current_step_num] = "\n".join(current_content).strip()
        return steps

    origin_steps = parse_steps(origin_str)
    replace_steps = parse_steps(replace_str)

    for n, content in replace_steps.items():
        if "DELETE THIS STEP" in content:
            if n in origin_steps:
                del origin_steps[n]
        else:
            origin_steps[n] = content

    sorted_steps = sorted(origin_steps.items())
    return "\n\n".join([f"{c}" for _, c in sorted_steps])

def make_unique_dir(path: str) -> str:
    path = path.rstrip("/")

    # use the path as-is when it does not exist yet
    if not os.path.exists(path):
        return path

    # otherwise append -1, -2 ... until an unused path is found
    i = 1
    while True:
        cand = f"{path}-{i}"
        if not os.path.exists(cand):
            return cand
        i += 1


def infer_visible_gpu_count(default: int = 1) -> int:
    """How many GPUs are visible to this process (respecting CUDA_VISIBLE_DEVICES)."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() != "":
        # e.g., "0,1,2,3" or "3" or "0,2"
        ids = [x.strip() for x in cvd.split(",") if x.strip() != ""]
        return max(0, len(ids))

    # fallback: torch detection
    try:
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass

    return default

def choose_tp_size(tp_arg: Optional[int], default: int = 1) -> int:
    """Choose TP size based on visible GPUs; clamp tp_arg if too large."""
    visible = infer_visible_gpu_count(default=default)
    if visible <= 0:
        return 1  # CPU fallback-like safety

    if tp_arg is None:
        return visible  # auto: use all visible GPUs
    return max(1, min(tp_arg, visible))  # clamp
def _format_docs_for_llm(docs: List[str], max_len: int) -> str:
    blocks = []
    for i, d in enumerate(docs):
        d = (d or "")[: max_len]  # truncate safely
        payload = {"rank": i + 1, "document": d}
        blocks.append(f"**Doc {i+1}:**\n" + json.dumps(payload, ensure_ascii=False, indent=2))
    return "\n".join(blocks)

def main():
    args = parse_args()

    # === per-dataset parameters (same as the original) ===
    MAX_SEARCH_LIMIT = args.max_search_limit
    MAX_TURN = args.max_turn
    top_k = args.top_k
    max_doc_len = args.max_doc_len

    if args.dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wikimultihopqa', 'medmcqa', 'pubhealth', 'squad']:
        MAX_SEARCH_LIMIT = 5
        if args.dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wikimultihopqa']:
            MAX_SEARCH_LIMIT = 10
            MAX_TURN = 15
        top_k = 10
        max_doc_len = 3000

    repetition_penalty = 1.05 if (args.repetition_penalty is None and 'qwq' in args.model_path.lower()) \
                            else (args.repetition_penalty if args.repetition_penalty is not None else 1.0)

    # load data
    if args.dataset_name == "strategyqa":
        data = StrategyQA(args.data_path)
    elif args.dataset_name == "2wikimultihopqa":
        data = WikiMultiHopQA(args.data_path)
    elif args.dataset_name == "hotpotqa" or args.dataset_name=="musique":
        data = HotpotQA(args.data_path)
    elif args.dataset_name == "iirc":
        data = IIRC(args.data_path)
    elif args.dataset_name == "nq" or args.dataset_name =="squad" or args.dataset_name=="triviaqa":
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
    # === tokenizer / llm ===
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    if 'qwq' in args.model_path.lower():
        if args.dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{args.dataset_name}.qwq.search_o1_local'
            if args.dataset_name == 'gpqa' and (MAX_SEARCH_LIMIT != 5 or top_k != 10):
                output_dir = f'./outputs/runs.analysis/{args.dataset_name}.qwq.search_o1_local.{MAX_SEARCH_LIMIT}.{top_k}'
        else:
            output_dir = f'./outputs/runs.qa/{args.dataset_name}.qwq.search_o1_local.{args.style}'
    else:
        model_short = args.model_path.split('/')[-1].lower().replace('-instruct','').replace('meta-','')
        output_dir = f'./outputs/runs.baselines/{args.dataset_name}.{model_short}.search_o1_local.{args.style}'
    
    output_dir = make_unique_dir(output_dir)
    os.makedirs(output_dir, exist_ok=False)
    print(f"[INFO] Saving outputs to: {output_dir}")
        
    run_evaluation, extract_answer = _load_eval_funcs()
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=choose_tp_size(args.tp_size, default=1),
        gpu_memory_utilization=0.95,
    )

    # === retriever initialization ===
    if args.retriever == 'BM25':
        retriever = BM25(
            tokenizer=tokenizer,
            index_name=args.es_index_name,  # same parameter name as in the retriever code
            engine='elasticsearch',
        )
        retriever_type = 'BM25'
    else:
        retriever = SGPT(
            model_name_or_path=args.sgpt_model_name_or_path,
            sgpt_encode_file_path=args.sgpt_encode_file_path,
            passage_file=args.passage_file
        )
        retriever_type = 'SGPT'

    # === build the user prompt (same as the original) ===
    input_list = []
    for item in data:
        q = item['question']
        if args.dataset_name in ['nq','triviaqa','hotpotqa','musique','bamboogle','2wikimultihopqa', 'squad']:
            if args.dataset_name in ['nq','triviaqa', 'squad']:
                instruction = get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT)
            else:
                instruction = get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT)
            user_prompt = get_task_instruction_openqa(q, model_name='qwq') if 'qwq' in args.model_path.lower() \
                          else get_task_instruction_openqa(q)
        elif args.dataset_name in ['math500','aime','amc']:
            instruction = get_math_search_o1_instruction(MAX_SEARCH_LIMIT)
            user_prompt = get_task_instruction_math(q, model_name='qwq') if 'qwq' in args.model_path.lower() \
                          else get_task_instruction_math(q)
        elif args.dataset_name == 'gpqa':
            instruction = get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT)
            if 'qwq' in args.model_path.lower():
                user_prompt = get_task_instruction_multi_choice(q, model_name='qwq')
            elif 'llama' in args.model_path.lower():
                user_prompt = get_task_instruction_multi_choice(q, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(q)
        elif args.dataset_name == 'livecode':
            instruction = get_code_search_o1_instruction(MAX_SEARCH_LIMIT)
            user_prompt = get_task_instruction_code(
                q, question_title=item.get('question_title', ''),
                model_name='qwq'
            ) if 'qwq' in args.model_path.lower() else get_task_instruction_code(q)
        else:
            instruction = ""
            user_prompt = ""

        prompt = [{"role": "user", "content": instruction + user_prompt}]
        prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        input_list.append(prompt)


    # === active sequences ===
    active_sequences = [{
        'item': item,
        'prompt': prompt,
        'output': '',
        'finished': False,
        'history': [],
        'search_count': 0,
        'executed_search_queries': set(),
        'generate_count': 0,
        'token_count': 0,
        'sentence_count': 0,
        'hallucinated_count': 0,
    } for item, prompt in zip(data, input_list)]

    # === max tokens (original rule) ===
    if 'qwq' in args.model_path.lower():
        max_tokens = 32768 if args.dataset_name in ['aime','amc','livecode'] else 20480
    else:
        max_tokens = 8192

    def run_generation(seqs: List[Dict], max_tokens_: int):
        prompts = [s['prompt'] for s in seqs]
        sp = SamplingParams(
            max_tokens=max_tokens_,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k_sampling,
            repetition_penalty=repetition_penalty,
            stop=[END_SEARCH_QUERY, tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
        return llm.generate(prompts, sampling_params=sp)

    start_time = time.time()
    turn = 0

    while True:
        need = [s for s in active_sequences if not s['finished']]
        if not need:
            break

        turn += 1
        print(f"\n-------------- Turn {turn} --------------")
        outputs = run_generation(need, max_tokens)

        batch_sequences = []
        batch_documents = []
        batch_original_questions = []
        batch_prev_reasonings = []
        batch_search_queries = []

        for seq, out in zip(need, outputs):
            text = out.outputs[0].text
            seq['generate_count'] += 1
            gen_text = out.outputs[0].text
            seq['token_count'] += len(tokenizer.tokenize(gen_text))
            seq['sentence_count'] += len([s for s in re.split(r'[.!?]+\s+', gen_text) if s.strip()])
            seq['history'].append(text)
            seq['prompt'] += text
            seq['output'] += text

            search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

            if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
                if seq['search_count'] < MAX_SEARCH_LIMIT and search_query not in seq['executed_search_queries']:
                    try:
                        if retriever_type == 'BM25':
                            # BM25: returns (docids, docs) ndarrays of shape (bs, topk)
                            _, docs_nd = retriever.retrieve(
                                queries=[search_query],
                                topk=top_k,
                                max_query_length=64
                            )
                            docs = list(docs_nd[0]) if docs_nd.size else []
                        else:
                            # SGPT: returns List[List[str]]
                            docs_list = retriever.retrieve(
                                queries=[search_query],
                                topk=top_k
                            )
                            docs = docs_list[0] if docs_list else []
                    except Exception as e:
                        print(f"[Retriever ERROR] {e}")
                        docs = []

                    formatted_documents = _format_docs_for_llm(docs, args.max_doc_len)

                    all_steps = seq['output'].replace('\n\n', '\n').split("\n")
                    truncated_prev = ""
                    for i, step in enumerate(all_steps):
                        truncated_prev += f"Step {i+1}: {step}\n\n"
                    parts = truncated_prev.split('\n\n')
                    if len(parts) > 5:
                        truncated_prev = ''
                        for i, s_ in enumerate(parts):
                            if i == 0 or i >= len(parts) - 4 or BEGIN_SEARCH_QUERY in s_ or BEGIN_SEARCH_RESULT in s_:
                                truncated_prev += s_ + '\n\n'
                            else:
                                if not truncated_prev.endswith('\n\n...\n\n'):
                                    truncated_prev += '...\n\n'
                        truncated_prev = truncated_prev.strip('\n')

                    batch_sequences.append(seq)
                    batch_documents.append(formatted_documents)
                    batch_original_questions.append(seq['item']['question'])
                    batch_prev_reasonings.append(truncated_prev)
                    batch_search_queries.append(search_query)

                    seq['search_count'] += 1
                    seq['executed_search_queries'].add(search_query)

                elif seq['search_count'] >= MAX_SEARCH_LIMIT:
                    msg = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                    seq['prompt'] += msg
                    seq['output'] += msg
                    seq['history'].append(msg)
                else:
                    msg = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                    seq['prompt'] += msg
                    seq['output'] += msg
                    seq['history'].append(msg)
            else:
                seq['finished'] = True

        if batch_sequences:
            user_prompts = [
                get_webpage_to_reasonchain_instruction(prev_r, sq, docs_blob)
                for prev_r, sq, docs_blob in zip(batch_prev_reasonings, batch_search_queries, batch_documents)
            ]
            prompts = [{"role": "user", "content": up} for up in user_prompts]
            prompts = [tokenizer.apply_chat_template([p], tokenize=False, add_generation_prompt=True) for p in prompts]

            out2 = llm.generate(
                prompts,
                sampling_params=SamplingParams(
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    repetition_penalty=1.05,
                )
            )
            for seq, o in zip(batch_sequences, out2):
                seq['generate_count'] += 1
                gen_text2 = o.outputs[0].text
                seq['token_count'] += len(tokenizer.encode(gen_text2))
                seq['sentence_count'] += len([s for s in re.split(r'[.!?]+\s+', gen_text2) if s.strip()])
            raw = [o.outputs[0].text for o in out2]
            analyses = [extract_answer(r_, mode='infogen') for r_ in raw]

            for seq, analysis in zip(batch_sequences, analyses):
                if isinstance(analysis, str):
                    append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                else:
                    append_text = replace_recent_steps(seq['output'], analysis)
                seq['prompt'] += append_text
                seq['output'] += append_text
                seq['history'].append(append_text)
                seq['token_count'] += len(tokenizer.encode(append_text))
                seq['sentence_count'] += len([s for s in re.split(r'[.!?]+\s+', append_text) if s.strip()])

        if turn >= MAX_TURN:
            print(f"Maximum number of turns ({MAX_TURN}) reached, stopping.")
            break

    total_time = time.time() - start_time
    output_path = os.path.join(output_dir, "output.txt")
    with open(output_path, "w", encoding="utf-8") as fout:
        for seq in active_sequences:
            item = seq["item"]
            qid = item.get("qid", item.get("id", item.get("uid", "unknown")))
            record = {
                "qid": qid,
                "prediction": seq["output"],
                "retrieve_count": int(seq.get("search_count", 0)),
                "generate_count": int(seq.get("generate_count", 0)),
                "hallucinated_count": int(seq.get("hallucinated_count", 0)),
                "token_count": int(seq.get("token_count", 0)),
                "sentence_count": int(seq.get("sentence_count", 0)),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2) config.json (read by the evaluator via --dir)
    cfg = {
        "dataset": args.dataset_name,
        "data_path": args.data_path,
        "fewshot": args.fewshot,
        "model_name_or_path": args.model_path,
        "use_counter": True,
        "output_dir": output_dir
    }
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    t = time.localtime()
    out_file = os.path.join(output_dir, f'{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.info_extract.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({"note": "local retriever version (BM25/SGPT)"}, f, ensure_ascii=False, indent=2)

    output_list = [seq['output'] for seq in active_sequences]
    run_evaluation(data, input_list, output_list, args.dataset_name, output_dir, total_time, args.split)

    print("Process completed (Local Retriever).")

if __name__ == "__main__":
    import multiprocessing as mp
    import os

    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    mp.set_start_method("spawn", force=True)

    main()
