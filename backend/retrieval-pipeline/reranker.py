from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from config import PipelineConfig, RerankerConfig, SearchMode
from logger import logger
import aiohttp
import asyncio
import torch
import time
import hashlib


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: Optional[Dict] = None
    rerank_score: float = 0.0

    def to_dict(self, rank: Optional[int] = None) -> Dict[str, Any]:
        meta = self.metadata or {}
        result = {
            "rank": rank,
            "id": str(meta.get("doc_id", "")),
            "score": float(self.rerank_score or self.score),
            "fact": self.text,
            "accusation": meta.get("accusation", []),
            "relevant_articles": meta.get("relevant_articles", []),
            "imprisonment": meta.get("term_of_imprisonment", {
                "death_penalty": False,
                "life_imprisonment": False,
                "imprisonment": 0
            }),
            "punish_of_money": float(meta.get("punish_of_money", 0.0)),
            "criminals": meta.get("criminals", []),
            "matched_keywords": meta.get("matched_keywords", {}),
            "query_schema": meta.get("query_schema"),
            "document_schema": meta.get("document_schema"),
            "document_keywords": meta.get("document_keywords", {})
        }
        if rank is not None:
            result["rank"] = rank
        return result

class RerankerWorker:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = RerankerConfig.device
        self.tokenizer = None
        self.model = None
        self.semaphore = asyncio.Semaphore(1)  
        self.batch_size = RerankerConfig.batch_size
        self.max_length = RerankerConfig.max_length
        self._init_model()

    def _init_model(self):
        """初始化模型"""
        try:
            model_path = RerankerConfig.model_path 

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            ).to(self.device)

            if self.config.reranker.use_fp16:
                self.model = self.model.half()

            self.model.eval()
            logger.debug("Reranker模型加载完成")

        except Exception as e:
            logger.error(f"Reranker初始化失败: {str(e)}", exc_info=True)
            raise

    def preprocess_text(self, text: str) -> str:
        """
        基础预处理，并将文本截断至最多 self.max_length // 2 - 2 个token。
        """
        if not isinstance(text, str) or not text.strip():
            return "[EMPTY]"
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        truncated_tokens = tokens[:self.max_length//2 - 2]
        truncated_text = self.tokenizer.decode(truncated_tokens, clean_up_tokenization_spaces=True)
        
        return truncated_text


    @torch.no_grad()
    async def process_batch(self, query: str, texts: List[str]) -> List[float]:
        """处理单个批次"""
        try:
            processed_texts = [self.preprocess_text(text) for text in texts]
            
            inputs = self.tokenizer(
                [query] * len(processed_texts),
                processed_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            batch_inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**batch_inputs).logits
            scores = outputs[:, 1] if outputs.size(1) > 1 else outputs[:, 0]
            
            scores = scores.cpu()
            del batch_inputs, outputs
            torch.cuda.empty_cache()

            return scores.tolist()

        except Exception as e:
            logger.error(f"批处理失败: {str(e)}", exc_info=True)
            return [0.0] * len(texts)


class RetrievalPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.worker = RerankerWorker(config)
        self.score_cache = {}
        
    def get_cache_key(self, query: str, text: str) -> str:
        """生成缓存key"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{query_hash}:{text_hash}"

    def _build_metadata(self, item: Dict) -> Dict:
        """统一构建 metadata, 兼容 fact_id / id"""
        doc_id = item.get("fact_id") or item.get("id") or ""
        return {
            "doc_id": doc_id,
            "doc_index": item.get("metadata", {}).get("doc_index"),
            "accusation": item.get("accusation", []),
            "relevant_articles": item.get("relevant_articles", []),
            "term_of_imprisonment": item.get("imprisonment", {}),
            "punish_of_money": item.get("punish_of_money", 0),
            "criminals": item.get("criminals", []),
            "matched_keywords": item.get("matched_keywords", {}),
            "query_schema": item.get("query_schema"),
            "document_schema": item.get("document_schema"),
            "document_keywords": item.get("document_keywords", {})
        }

    async def _get_dense_results(self, query: str, top_k: int) -> List[SearchResult]:
        url = f"{self.config.services.dense_service_url.rstrip('/')}/search"
        payload = {"fact": query, "top_k": top_k}
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return [
                    SearchResult(
                        text=item.get("fact", ""),
                        score=item.get("score", 0.0),
                        metadata=self._build_metadata(item)
                    )
                    for item in data
                ]

    async def _get_sparse_results(self, query: str, top_k: int) -> List[SearchResult]:
        url = f"{self.config.services.sparse_service_url.rstrip('/')}/search"
        payload = {"fact": query, "top_k": top_k}
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return [
                    SearchResult(
                        text=item.get("fact", ""),
                        score=item.get("score", 0.0),
                        metadata=self._build_metadata(item)
                    )
                    for item in data
                ]

    async def _batch_get_dense_results(self, queries: List[str], top_k: int) -> List[List[SearchResult]]:
        url = f"{self.config.services.dense_service_url.rstrip('/')}/batch_search"
        payload = {"facts": queries, "top_k": top_k}
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    return [[] for _ in queries]
                data = await resp.json()
                return [
                    [
                        SearchResult(
                            text=item.get("fact", ""),
                            score=item.get("score", 0.0),
                            metadata=self._build_metadata(item)
                        )
                        for item in batch
                    ]
                    for batch in data
                ]

    async def _batch_get_sparse_results(self, queries: List[str], top_k: int) -> List[List[SearchResult]]:
        url = f"{self.config.services.sparse_service_url.rstrip('/')}/batch_search"
        payload = {"facts": queries, "top_k": top_k}
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                if resp.status != 200:
                    return [[] for _ in queries]
                data = await resp.json()
                return [
                    [
                        SearchResult(
                            text=item.get("fact", ""),
                            score=item.get("score", 0.0),
                            metadata=self._build_metadata(item)
                        )
                        for item in batch
                    ]
                    for batch in data
                ]

    async def _async_batch_process(self, tasks):
        return await asyncio.gather(*tasks)

    async def rerank(self, query: str, candidates: List[SearchResult]) -> List[SearchResult]:
        if not candidates:
            return []

        texts = [c.text for c in candidates]
        cache_keys = [self.get_cache_key(query, text) for text in texts]
        
        cached_scores = {}
        uncached_texts = []
        uncached_indices = []

        for i, (key, text) in enumerate(zip(cache_keys, texts)):
            if key in self.score_cache:
                cached_scores[i] = self.score_cache[key]
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            async with self.worker.semaphore:
                for i in range(0, len(uncached_texts), self.worker.batch_size):
                    batch_texts = uncached_texts[i:i + self.worker.batch_size]
                    batch_scores = await self.worker.process_batch(query, batch_texts)
                    
                    for text, score, idx in zip(batch_texts, batch_scores, uncached_indices[i:i + self.worker.batch_size]):
                        key = self.get_cache_key(query, text)
                        self.score_cache[key] = score
                        cached_scores[idx] = score

        for i, candidate in enumerate(candidates):
            score = cached_scores.get(i, 0.0)
            candidate.rerank_score = float(score)
            candidate.score = float(score)

        return sorted(candidates, key=lambda x: x.rerank_score, reverse=True)

    async def batch_rerank(
        self,
        queries: List[str],
        batch_candidates: List[List[SearchResult]]
    ) -> List[List[SearchResult]]:
        results = []
        for query, candidates in zip(queries, batch_candidates):
            reranked = await self.rerank(query, candidates)
            results.append(reranked)
        return results

    async def search(
        self,
        query: str,
        mode: SearchMode = SearchMode.HYBRID,
        top_k: Optional[int] = None,
        rerank: bool = True
    ) -> List[Dict]:
        top_k = top_k or self.config.default_top_k
        fetch_k = min(self.config.rerank_top_k, top_k * 2)

        tasks = []
        if mode in (SearchMode.DENSE, SearchMode.HYBRID):
            tasks.append(self._get_dense_results(query, fetch_k))
        if mode in (SearchMode.SPARSE, SearchMode.HYBRID):
            tasks.append(self._get_sparse_results(query, fetch_k))

        results = await asyncio.gather(*tasks)
        
        all_results = []
        if mode in (SearchMode.DENSE, SearchMode.HYBRID):
            all_results.extend(results[0])
        if mode in (SearchMode.SPARSE, SearchMode.HYBRID):
            all_results.extend(results[-1])

        seen = set()
        unique = []
        for r in all_results:
            key = r.text.strip()
            if key and key not in seen:
                seen.add(key)
                unique.append(r)

        if rerank and unique:
            unique = await self.rerank(query, unique)

        results = []
        for rank, item in enumerate(unique[:top_k], start=1):
            results.append(item.to_dict(rank=rank))
        return results

    async def batch_search(
        self,
        queries: List[str],
        mode: SearchMode = SearchMode.HYBRID,
        top_k: Optional[int] = None,
        rerank: bool = True
    ) -> List[List[Dict]]:
        start_time = time.time()
        logger.info(f"开始批量检索: {len(queries)} 条查询")
        
        top_k = top_k or self.config.default_top_k
        fetch_k = min(self.config.rerank_top_k, top_k * 2)
        
        tasks = []
        if mode in (SearchMode.DENSE, SearchMode.HYBRID):
            tasks.append(self._batch_get_dense_results(queries, fetch_k))
        if mode in (SearchMode.SPARSE, SearchMode.HYBRID):
            tasks.append(self._batch_get_sparse_results(queries, fetch_k))
        
        results = await asyncio.gather(*tasks)
        
        merged_results = []
        for i, query in enumerate(queries):
            candidates = []
            if mode in (SearchMode.DENSE, SearchMode.HYBRID):
                candidates.extend(results[0][i])
            if mode in (SearchMode.SPARSE, SearchMode.HYBRID):
                candidates.extend(results[-1][i])
                
            seen = set()
            unique = []
            for r in candidates:
                key = r.text.strip()
                if key and key not in seen:
                    seen.add(key)
                    unique.append(r)
                    
            if rerank:
                unique = await self.rerank(query, unique)
                
            reranked_list = []
            for rank, item in enumerate(unique[:top_k], start=1):
                reranked_list.append(item.to_dict(rank=rank))
            merged_results.append(reranked_list)

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"批量检索完成:\n"
            f"  - 总耗时: {total_time:.2f}ms (平均: {total_time/len(queries):.2f}ms/查询)\n"
        )
        return merged_results