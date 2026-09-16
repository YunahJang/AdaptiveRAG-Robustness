# Part of the QueryRobustness overlay for AdaRAGUE (https://github.com/s-nlp/AdaRAGUE).
# MIT License, Copyright (c) 2025 Natural Language Processing.
# Added by this project.
"""
Dense Retriever implementations using FAISS
Supports Contriever and other dense retrieval models
"""

from typing import List
import json
import os
import pickle
import logging
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_random_doc_id():
    import uuid
    return f'_{uuid.uuid4()}'


class Contriever:
    """
    Contriever dense retriever using FAISS index.
    Uses mean pooling for embeddings and cosine similarity for retrieval.
    """
    def __init__(
        self,
        model_name_or_path: str = "facebook/contriever",
        faiss_index_path: str = None,
        meta_file_path: str = None,
        max_query_length: int = 256,
        use_faiss_gpu: bool = False,
        faiss_gpu_id: int = 0,
    ):
        logger.info(f"[Contriever] Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(
                model_name_or_path,
                use_safetensors=True
            )
        self.model.eval()
        self.device = torch.device(f"cuda:{faiss_gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_query_length = max_query_length

        # --- FAISS index ---
        logger.info(f"[Contriever] Loading FAISS index from {faiss_index_path}")
        self.index = faiss.read_index(faiss_index_path)
        logger.info(f"[Contriever] FAISS index loaded: {self.index.ntotal} vectors")

        self.on_gpu = False  # optional: state flag

        if use_faiss_gpu and faiss.get_num_gpus() > 0:
            try:
                logger.info(f"[Contriever] Moving FAISS index to GPU {faiss_gpu_id}")
                res = faiss.StandardGpuResources()

                # (optional) avoid OOM from an oversized temporary buffer
                # res.setTempMemory(256 * 1024 * 1024)  # 256MB

                co = faiss.GpuClonerOptions()
                co.useFloat16 = True  # keep it compact (good for memory)

                self.index = faiss.index_cpu_to_gpu(res, faiss_gpu_id, self.index, co)
                self.on_gpu = True
                logger.info("[Contriever] FAISS index on GPU")

            except RuntimeError as e:
                # fall back to CPU here
                logger.warning(f"[Contriever] FAISS GPU move failed -> fallback to CPU. Reason: {e}")
                self.on_gpu = False
        else:
            if use_faiss_gpu:
                logger.warning("[Contriever] faiss-gpu not available, falling back to CPU")
            self.on_gpu = False

        # --- Passage metadata (pickle > jsonl) ---
        pkl_path = os.path.splitext(meta_file_path)[0] + ".pkl"
        if os.path.exists(pkl_path):
            logger.info(f"[Contriever] Loading passage metadata from pickle: {pkl_path}")
            with open(pkl_path, "rb") as f:
                meta = pickle.load(f)
            self.doc_ids = meta["doc_ids"]
            self.docs = meta["docs"]
            self.titles = meta["titles"]
        else:
            logger.info(f"[Contriever] Loading passage metadata from {meta_file_path}")
            self.doc_ids = []
            self.docs = []
            self.titles = []
            with open(meta_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    self.doc_ids.append(obj['pid'])
                    self.docs.append(obj['text'])
                    self.titles.append(obj.get('title', ''))
        logger.info(f"[Contriever] Loaded {len(self.docs)} passages")

    def mean_pooling(self, last_hidden_state, attention_mask):
        """Mean pooling for Contriever embeddings"""
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """Encode queries into dense embeddings"""
        inputs = self.tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean pooling (Contriever style)
        embeddings = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = embeddings.cpu().numpy().astype(np.float32)

        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        return embeddings

    def retrieve(
        self,
        queries: List[str],
        topk: int = 1,
        max_query_length: int = None,
    ):
        """
        Retrieve top-k documents for given queries.

        Args:
            queries: List of query strings
            topk: Number of documents to retrieve per query
            max_query_length: Optional override for max query length (not used, kept for compatibility)

        Returns:
            docids: numpy array of shape (batch_size, topk) with document IDs
            docs: numpy array of shape (batch_size, topk) with document texts
        """
        bs = len(queries)
        query_embeddings = self.encode_queries(queries)

        # FAISS search (cosine similarity with normalized vectors = inner product)
        scores, indices = self.index.search(query_embeddings, topk)

        docids = []
        docs = []
        for i in range(bs):
            _docids = []
            _docs = []
            for j in range(topk):
                idx = indices[i][j]
                if idx == -1 or idx >= len(self.docs):
                    # Handle invalid indices
                    _docids.append(get_random_doc_id())
                    _docs.append('')
                else:
                    _docids.append(str(self.doc_ids[idx]))
                    _docs.append(self.docs[idx])
            docids.extend(_docids)
            docs.extend(_docs)

        docids = np.array(docids).reshape(bs, topk)
        docs = np.array(docs).reshape(bs, topk)
        return docids, docs
