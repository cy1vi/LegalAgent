import os
import json
import random
import numpy as np
import torch
import httpx
import asyncio
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple

from prepare_datasets_v2 import GoldDatasetBuilder
from config import EvalConfig, CleanConfig


# =========================
# Evaluation 安全参数区
# =========================
SEARCH_BATCH_SIZE = 4
SEARCH_TOP_K = 100
SEARCH_CONCURRENCY = 1
MIN_FACT_LEN = 100


# ============================================================
# Lazy Loaders
# ============================================================

class LazySparseLoader:
    """稀疏特征 JSONL 懒加载器"""
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, 'rb')
        self.offsets = []

        print(f"正在构建稀疏特征索引: {path}")
        offset = 0
        while True:
            line = self.file.readline()
            if not line:
                break
            self.offsets.append(offset)
            offset += len(line)

        print(f"稀疏索引构建完成，共 {len(self.offsets)} 条")

    def __getitem__(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self.offsets):
            return {}
        self.file.seek(self.offsets[idx])
        try:
            return json.loads(self.file.readline())
        except Exception:
            return {}

    def __len__(self):
        return len(self.offsets)

    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()


class LazyJSONLLoader:
    """主数据 JSONL 懒加载器"""
    def __init__(self, path: str):
        self.path = path
        self.file = open(path, 'rb')
        self.offsets = []

        print(f"正在构建 JSONL 偏移索引: {path}")
        offset = 0
        while True:
            line = self.file.readline()
            if not line:
                break
            self.offsets.append(offset)
            offset += len(line)

        print(f"JSONL 索引完成，共 {len(self.offsets)} 条")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict:
        if idx < 0 or idx >= len(self.offsets):
            raise IndexError
        self.file.seek(self.offsets[idx])
        return json.loads(self.file.readline())

    def __del__(self):
        if hasattr(self, 'file'):
            self.file.close()


# ============================================================
# Sampler
# ============================================================

class DiverseSampler(GoldDatasetBuilder):

    def __init__(self, data_path, device='cuda', service_url="http://localhost:4241"):
        super().__init__(data_path)

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.service_url = service_url
        self.search_semaphore = asyncio.Semaphore(SEARCH_CONCURRENCY)

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        print(f"🚀 使用设备: {self.device}")
        print(f"📡 搜索服务地址: {self.service_url}")
        print(f"🔒 搜索并发限制: {SEARCH_CONCURRENCY}")

    # ------------------------------------------------
    # Embeddings / Sparse
    # ------------------------------------------------
    def _load_embeddings(self):
        embedding_path = r"F:\LegalAgent\backend\data\fact_embeddings\fact_embeddings_bge-m3.npy"
        if os.path.exists(embedding_path):
            print(f"加载 Dense Embedding (mmap): {embedding_path}")
            self.fact_embeddings = np.load(embedding_path, mmap_mode='r')

        sparse_path = r"F:\LegalAgent\backend\sparse-embedding\data\train_sparse_features.jsonl"
        if os.path.exists(sparse_path):
            self.sparse_features = LazySparseLoader(sparse_path)
        else:
            self.sparse_features = {}

    # ------------------------------------------------
    # Query 选择（返回 index + item）
    # ------------------------------------------------
    def _select_diverse_queries(self, target_count: int) -> List[Tuple[int, Dict]]:
        acc_to_cases: Dict[str, List[int]] = defaultdict(list)

        print("正在筛选 Query 候选集...")
        for idx in range(len(self.all_data)):
            item = self.all_data[idx]
            accs = item.get('meta', {}).get('accusation', [])
            if len(accs) == 1 and len(item.get('fact', '')) >= MIN_FACT_LEN:
                acc_to_cases[accs[0]].append(idx)

        selected_indices = []

        for cases in acc_to_cases.values():
            if cases:
                selected_indices.append(random.choice(cases))

        if len(selected_indices) < target_count:
            rest = []
            for cases in acc_to_cases.values():
                rest.extend([i for i in cases if i not in selected_indices])
            need = target_count - len(selected_indices)
            if rest:
                selected_indices.extend(random.sample(rest, min(need, len(rest))))

        if len(selected_indices) < target_count:
            rest = [
                i for i in range(len(self.all_data))
                if i not in selected_indices
                and len(self.all_data[i].get('meta', {}).get('accusation', [])) > 1
                and len(self.all_data[i].get('fact', '')) >= MIN_FACT_LEN
            ]
            need = target_count - len(selected_indices)
            if rest:
                selected_indices.extend(random.sample(rest, min(need, len(rest))))

        print(f"最终选择了 {len(selected_indices)} 个 Query")
        return [(i, self.all_data[i]) for i in selected_indices]

    # ------------------------------------------------
    # Search
    # ------------------------------------------------
    async def _batch_search(self, facts: List[str]) -> List[List[Dict]]:
        async with self.search_semaphore:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self.service_url}/batch_search",
                    json={"facts": facts, "top_k": SEARCH_TOP_K},
                    timeout=60.0
                )
                resp.raise_for_status()
                return resp.json()

    # ------------------------------------------------
    # Main
    # ------------------------------------------------
    async def build_diverse_dataset_async(
        self,
        output_path: str,
        num_queries: int = 300,
        initial_positives: int = 8
    ):
        if not self.all_data:
            print(f"使用 LazyJSONLLoader 加载数据: {self.data_path}")
            self.all_data = LazyJSONLLoader(self.data_path)
            print(f"数据总量: {len(self.all_data)}")

        queries = self._select_diverse_queries(num_queries)
        dataset = []

        for i in tqdm(range(0, len(queries), SEARCH_BATCH_SIZE), desc="Evaluation 构建中"):
            batch = queries[i:i + SEARCH_BATCH_SIZE]
            batch_facts = [item['fact'] for _, item in batch]

            try:
                search_results = await self._batch_search(batch_facts)
            except Exception as e:
                print(f"❌ 搜索失败，跳过 batch: {e}")
                continue

            for (query_idx, query_item), candidates in zip(batch, search_results):
                scored = []

                for c in candidates:
                    c_idx = int(c["fact_id"])
                    if c_idx >= len(self.all_data):
                        continue

                    cand = self.all_data[c_idx]
                    if len(cand.get("fact", "")) < MIN_FACT_LEN:
                        continue

                    score_info = self._calculate_comprehensive_score(
                        query_idx, query_item, c_idx, cand
                    )

                    scored.append({
                        "candidate_item": cand,
                        "candidate_idx": c_idx,
                        "score_info": score_info
                    })

                scored.sort(key=lambda x: x["score_info"]["total_score"], reverse=True)
                topk = scored[:initial_positives]

                if len(topk) >= initial_positives:
                    dataset.append({
                        "query": query_item,
                        "query_idx": query_idx,
                        "positives": [x["candidate_item"] for x in topk],
                        "score_details": [x["score_info"] for x in topk],
                        "positives_count": len(topk)
                    })

        tmp_final = output_path.replace(".jsonl", ".tmp_final.jsonl")
        with open(tmp_final, "w", encoding="utf-8") as f:
            for d in dataset:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        print("🧹 开始清洗数据集")
        CleanConfig.INPUT_FILE = tmp_final
        CleanConfig.OUTPUT_CLEAN_FILE = output_path
        CleanConfig.OUTPUT_DIRTY_FILE = output_path.replace(".jsonl", ".dirty.jsonl")

        import clean_prepared_datasets
        import importlib
        importlib.reload(clean_prepared_datasets)
        clean_prepared_datasets.clean_data()

        os.remove(tmp_final)
        print("✅ Evaluation 数据集构建完成")

    def build_diverse_dataset(self, *args, **kwargs):
        asyncio.run(self.build_diverse_dataset_async(*args, **kwargs))


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    sampler = DiverseSampler(
        EvalConfig.RAW_DATA_PATH,
        service_url="http://localhost:4241"
    )

    sampler.build_diverse_dataset(
        EvalConfig.EVAL_DATASET_PATH.replace(".jsonl", ".diverse.jsonl"),
        num_queries=300,
        initial_positives=8
    )
