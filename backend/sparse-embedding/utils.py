import json
import numpy as np
from scipy import sparse
from typing import List, Dict, Any
import logging
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

class SparseRetriever:
    def __init__(self, schema_fields: List[str], one_hot_maps: Dict[str, Dict[str, int]], crime_keywords_path: str):
        self.schema_fields = schema_fields
        self.one_hot_maps = one_hot_maps
        self.matrix = None
        self.ids = [] 
        
        # 1. 加载关键词映射 
        kw_data = {}
        with open(crime_keywords_path, 'r', encoding='utf-8') as f:
            try:
                # 首先尝试作为完整JSON文件读取
                kw_data = json.load(f)
            except json.JSONDecodeError:
                # 如果失败，回退到JSONL格式读取
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if line:
                        kw_data.update(json.loads(line))

        all_keywords = sorted(list(set([kw for kws in kw_data.values() for kw in kws])))
        self.kw_map = {kw: i for i, kw in enumerate(all_keywords)}
        
        # 2. 计算维度偏移量
        self.schema_offsets = {}
        current_offset = 0
        for field in self.schema_fields:
            self.schema_offsets[field] = current_offset
            current_offset += len(self.one_hot_maps.get(field, {}))
            
        self.schema_dim = current_offset
        self.keyword_dim = len(all_keywords)
        self.total_dim = self.schema_dim + self.keyword_dim
        
        logger.debug(f"Retriever initialized. Schema Dim: {self.schema_dim}, Keyword Dim: {self.keyword_dim}, Total: {self.total_dim}")

    def load_precomputed_data(self, matrix_path: str):
        logger.debug(f"Loading sparse matrix from {matrix_path}...")
        self.matrix = sparse.load_npz(matrix_path)
        self.norm_matrix = normalize(self.matrix, norm='l2')  # 👈 预计算
        logger.debug(f"Matrix loaded and normalized. Shape: {self.norm_matrix.shape}")


    def search(self, query_schema: Dict[str, Any], query_keyword_counts: Dict[str, int], top_k: int = 5) -> List[Dict]:
        """
        执行混合检索（使用余弦相似度）
        :param query_schema: 扁平化的 Schema 字典 (e.g., {"act.has_violence": true})
        :param query_keyword_counts: 关键词频次 (e.g., {"抢劫": 1})
        """
        if self.matrix is None:
            raise ValueError("Matrix not loaded!")

        # 1. 构建查询向量 (维度 = Schema + Keywords)
        query_vec = sparse.dok_matrix((1, self.total_dim), dtype=np.float32)
        
        # --- A. 填充 Schema 部分 ---
        for field, value in query_schema.items():
            if field in self.one_hot_maps and str(value) in self.one_hot_maps[field]:
                val_idx = self.one_hot_maps[field][str(value)]
                col_idx = self.schema_offsets[field] + val_idx
                query_vec[0, col_idx] = 1.0
        
        # --- B. 填充 Keyword 部分 ---
        for kw, count in query_keyword_counts.items():
            if kw in self.kw_map:
                col_idx = self.schema_dim + self.kw_map[kw]
                query_vec[0, col_idx] = float(count)

        # 转换为 CSR 格式
        query_csr = query_vec.tocsr()
        
        # 2. 归一化
        norm_query = normalize(query_csr, norm='l2')
        scores = self.norm_matrix.dot(norm_query.T).toarray().flatten()
        
        # 3. 获取 Top-K
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:  
                results.append({
                    "id": str(idx),
                    "index": int(idx),
                    "score": score
                })
                    
        return results