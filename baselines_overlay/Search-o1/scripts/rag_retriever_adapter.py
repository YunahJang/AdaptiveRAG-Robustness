# Part of the QueryRobustness overlay for Search-o1 (https://github.com/RUC-NLPIR/Search-o1).
# MIT License, Copyright (c) 2025 Xiaoxi Li.
# Added by this project.
# rag_retriever_adapter.py
import hashlib
from typing import List, Dict, Any, Tuple

class RagRetrieverAdapter:
    """
    Search adapter exposing the RAG retriever through the interface run_search_o1.py expects.
    - search(query, top_k) -> List[Dict]: [{"url","name","snippet","score"}...]
    - fetch_pages(urls) -> Dict[url, full_text]
    """
    def __init__(self, rag, make_name_from_text: bool = True, snippet_chars: int = 300):
        self.rag = rag
        self.make_name_from_text = make_name_from_text
        self.snippet_chars = snippet_chars
        self._store: Dict[str, str] = {}  # url -> full_text

    def _mk_url(self, text: str) -> str:
        h = hashlib.md5(text.encode("utf-8")).hexdigest()[:16]
        return f"rag://{h}"

    def _mk_name(self, text: str) -> str:
        if not self.make_name_from_text:
            return "RAG Document"
        words = text.strip().split()
        return " ".join(words[:10]) if words else "RAG Document"

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Calls AttnWeightRAG.retrieve(query, topk), which returns a list of texts
        (or (text, score) pairs). The schema matches the replacement for
        extract_relevant_info in run_search_o1:
        [{"url","name","snippet","score"} ...]
        """
        # branch on what AttnWeightRAG.retrieve returned
        results = self.rag.retrieve(query, topk=top_k)

        normalized: List[Tuple[str, float]] = []
        # case A: ["full text", ...]
        if results and isinstance(results[0], str):
            normalized = [(r, 0.0) for r in results]
        # case B: [(text, score), ...] or dict with 'text'
        elif results and isinstance(results[0], (tuple, list)) and len(results[0]) >= 1:
            for item in results[:top_k]:
                text = item[0]
                score = float(item[1]) if len(item) > 1 else 0.0
                normalized.append((text, score))
        elif results and isinstance(results[0], dict):
            for item in results[:top_k]:
                text = item.get("text", "")
                score = float(item.get("score", 0.0))
                normalized.append((text, score))

        infos: List[Dict[str, Any]] = []
        for text, score in normalized[:top_k]:
            url = self._mk_url(text)
            name = self._mk_name(text)
            snippet = text[: self.snippet_chars].replace("\n", " ")
            self._store[url] = text  # keep the full text for fetch_pages
            infos.append({
                "url": url,
                "name": name,
                "snippet": snippet,
                "score": float(score),
            })
        return infos

    def fetch_pages(self, urls: List[str]) -> Dict[str, str]:
        """
        Replacement for Bing's fetch_page_content: return the stored text.
        Unknown urls yield an empty string.
        """
        out: Dict[str, str] = {}
        for u in urls:
            out[u] = self._store.get(u, "")
        return out
