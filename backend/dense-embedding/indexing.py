import gc
import os
import pickle
import numpy as np
from typing import List, Tuple
from logger import logger

# ---------------------------------------------------------
# 依赖检查
# ---------------------------------------------------------
try:
    import hnswlib
except ImportError:
    hnswlib = None

try:
    import faiss
except ImportError:
    faiss = None

# ---------------------------------------------------------
# 基类定义
# ---------------------------------------------------------
class BaseIndex:
    def __init__(self, dimension: int, index_file: str):
        self.dimension = dimension
        self.index_file = index_file
        self.ids = [] # 用于存储 ID 列表 (int_index -> str_id)

    def build(self, ids: List[str], embeddings: np.ndarray):
        raise NotImplementedError

    def search(self, query: np.ndarray, top_k: int) -> Tuple[List[str], List[float]]:
        raise NotImplementedError

    def save(self):
        # 保存 ID 映射
        meta_file = self.index_file + ".meta.pkl"
        with open(meta_file, 'wb') as f:
            pickle.dump(self.ids, f)

    def load(self, index_path: str):
        # 加载 ID 映射
        meta_file = index_path + ".meta.pkl"
        if os.path.exists(meta_file):
            with open(meta_file, 'rb') as f:
                self.ids = pickle.load(f)
        else:
            print(f"Warning: Meta file {meta_file} not found. IDs might be missing.")

# ---------------------------------------------------------
# HNSW 实现 (保留作为备选)
# ---------------------------------------------------------
class HNSWIndex(BaseIndex):
    def __init__(self, dimension: int, index_file: str, max_elements=2000000):
        super().__init__(dimension, index_file)
        if hnswlib is None:
            raise ImportError("hnswlib not installed. Please run `pip install hnswlib`")
        
        self.index = hnswlib.Index(space='cosine', dim=dimension)
        self.max_elements = max_elements
        self.initialized = False

    def build(self, ids, embeddings):
        """构建 HNSW 索引，支持分批添加以节省内存"""
        num_elements = len(embeddings)
        logger.info(f"HNSW: 开始构建索引，总数据量: {num_elements}, 维度: {self.dimension}")
        
        # 设置 HNSW 参数
        self.index.set_ef(1000)  # 搜索时考虑的邻居数
        self.index.set_num_threads(4)  # 并行线程数

        # 分批添加向量，避免内存溢出
        batch_size = 1000  # 每批处理 1000 条
        for i in range(0, num_elements, batch_size):
            end_i = min(i + batch_size, num_elements)
            batch_embeddings = embeddings[i:end_i].astype(np.float32)
            batch_ids = np.arange(i, end_i)
            
            logger.info(f"HNSW: 添加批次 {i//batch_size + 1}, 数据范围: [{i}, {end_i})")
            self.index.add_items(batch_embeddings, batch_ids)
            
            del batch_embeddings
            if i + batch_size < num_elements:
                import gc
                gc.collect()

        logger.info(f"HNSW: 索引构建完成，总数据量: {self.index.get_current_count()}")


    def search(self, query: np.ndarray, top_k: int) -> Tuple[List[str], List[float]]:
        if not self.initialized:
            raise RuntimeError("Index not initialized")
        
        self.index.set_ef(max(50, top_k * 2))
        
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
            
        labels, distances = self.index.knn_query(query, k=top_k)
        
        result_ids = [self.ids[i] for i in labels[0]]
        result_scores = [1 - d for d in distances[0]]
        
        return result_ids, result_scores

    def save(self):
        super().save()
        self.index.save_index(self.index_file)

    def load(self, index_path: str):
        super().load(index_path)
        self.index.load_index(index_path, max_elements=self.max_elements if hasattr(self, 'max_elements') else 0)
        self.initialized = True

# ---------------------------------------------------------
# FAISS 实现 (推荐: 内存友好且高精度)
# ---------------------------------------------------------
class FAISSIndex(BaseIndex):
    def __init__(self, dimension: int, index_file: str):
        super().__init__(dimension, index_file)
        if faiss is None:
            raise ImportError("faiss not installed. Please run `pip install faiss-cpu`")
        
        self.index = None
        self.built = False
        self.nlist = 4096  

    def build(self, ids: List[str], embeddings: np.ndarray):
        """
        使用 IVF4096 + SQ8，保持策略不变，优化内存使用
        """
        self.ids = ids
        
        # 原地归一化，避免复制
        print("FAISS: Normalizing vectors in-place...")
        faiss.normalize_L2(embeddings)  
        
        # 保持现有量化策略
        quantizer_str = f"IVF{self.nlist},SQ8"
        print(f"FAISS: Building index '{quantizer_str}'...")
        
        # 创建索引
        self.index = faiss.index_factory(self.dimension, quantizer_str, faiss.METRIC_INNER_PRODUCT)
        
        # 分批训练，避免内存峰值
        print("FAISS: Training index...")
        batch_size = 100000  # 每批10万个向量
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            if i == 0:
                self.index.train(batch)  # 只用第一批训练
            
            # 清理内存
            del batch
            import gc
            gc.collect()
        
        # 分批添加向量
        print("FAISS: Adding vectors...")
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i + batch_size]
            self.index.add(batch)
            
            # 清理内存
            del batch
            gc.collect()
        
        self.built = True
        print(f"FAISS: Index built. Total: {self.index.ntotal}")

    def search(self, query: np.ndarray, top_k: int) -> Tuple[List[str], List[float]]:
        if not self.built or self.index is None:
            raise RuntimeError("Index not initialized")
        
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        # 原地归一化查询向量
        query = query.astype(np.float32)
        faiss.normalize_L2(query)
        
        # 保持现有搜索参数
        self.index.nprobe = 128
        
        # 执行搜索
        scores, indices = self.index.search(query, top_k)
        
        valid_results = []
        valid_scores = []
        
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.ids):
                valid_results.append(self.ids[idx])
                valid_scores.append(float(scores[0][i]))
        
        # 清理内存
        del scores, indices
        gc.collect()
        
        return valid_results, valid_scores

    def save(self):
        super().save()
        if self.index:
            faiss.write_index(self.index, self.index_file)
            # 清理内存
            gc.collect()

    def load(self, index_path: str):
        super().load(index_path)
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.built = True
            # 清理内存
            gc.collect()
        else:
            raise FileNotFoundError(f"Index file not found: {index_path}")

class FAISSIndex(BaseIndex):
    def __init__(self, dimension: int, index_file: str):
        super().__init__(dimension, index_file)
        if faiss is None:
            raise ImportError("faiss not installed. Please run `pip install faiss-cpu`")
        
        self.index = None
        self.built = False
        self.nlist = 4096  

    def build(self, ids: List[str], embeddings: np.ndarray):
        """
        使用 IVF4096 + SQ8
        """
        self.ids = ids
        
        print("FAISS: Normalizing vectors for Cosine Similarity...")
        faiss.normalize_L2(embeddings)
        
        # 使用 SQ8 (Scalar Quantization 8-bit)
        # 精度损失极小，内存占用仅为原始的 25%
        quantizer_str = f"IVF{self.nlist},SQ8" 
        
        print(f"FAISS: Building index '{quantizer_str}'...")
        
        # METRIC_INNER_PRODUCT 在归一化后等同于 Cosine
        self.index = faiss.index_factory(self.dimension, quantizer_str, faiss.METRIC_INNER_PRODUCT)
        
        print("FAISS: Training index (this might take a minute)...")
        self.index.train(embeddings)
        
        print("FAISS: Adding vectors...")
        self.index.add(embeddings)
        
        self.built = True
        print(f"FAISS: Index built. Total: {self.index.ntotal}")

    def search(self, query: np.ndarray, top_k: int) -> Tuple[List[str], List[float]]:
        if not self.built or self.index is None:
            raise RuntimeError("Index not initialized")
        
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        query = query.astype(np.float32)
        faiss.normalize_L2(query)
        
        # --- 关键优化 ---
        # nprobe 控制搜索时查看多少个聚类中心。
        # 默认是 1。设为 128 (nlist的3%左右) 可以获得极高的召回率。
        # 虽然速度会变慢，但对于实验室环境完全没问题。
        self.index.nprobe = 128 
        
        scores, indices = self.index.search(query, top_k)
        
        valid_results = []
        valid_scores = []
        
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.ids):
                valid_results.append(self.ids[idx])
                valid_scores.append(float(scores[0][i]))
        
        return valid_results, valid_scores

    def save(self):
        super().save()
        if self.index:
            faiss.write_index(self.index, self.index_file)

    def load(self, index_path: str):
        super().load(index_path)
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.built = True
        else:
            raise FileNotFoundError(f"Index file not found: {index_path}")