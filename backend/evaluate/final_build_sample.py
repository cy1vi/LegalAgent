import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict

# 继承基类，但我们会覆盖大部分耗资源的方法
from prepare_datasets_v2 import GoldDatasetBuilder
from config import EvalConfig, CleanConfig

class FastSampler(GoldDatasetBuilder):
    def __init__(self, data_path, device='cuda'):
        # 不调用 super().__init__，因为我们要控制加载过程，避免内存爆炸
        self.data_path = data_path
        self.all_data = []
        self.accusation_index = {}
        self.fact_embeddings = None
        
        # 稀疏特征相关 (使用文件指针而非全量加载)
        self.sparse_file_offsets = []
        self.sparse_file_handle = None
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🚀 使用设备: {self.device}")
        
        # 1. 加载基础数据 (必须)
        self.load_and_index()
        
        # 2. 优化加载 Embedding 和 Sparse
        self._load_resources_optimized()

    def _load_resources_optimized(self):
        """优化加载：Embedding 尝试 mmap，Sparse 使用文件索引"""
        # A. 加载 Embedding
        embedding_path = r"F:\LegalAgent\backend\data\fact_embeddings\fact_embeddings_bge-m3.npy"
        if os.path.exists(embedding_path):
            try:
                # 尝试全量加载，如果内存不够则报错被捕获
                self.fact_embeddings = np.load(embedding_path)
                print(f"已加载嵌入向量: {embedding_path}, shape: {self.fact_embeddings.shape}")
            except Exception:
                print("⚠️ 内存紧张，使用 mmap 模式加载嵌入向量 (速度稍慢但不会崩溃)")
                self.fact_embeddings = np.load(embedding_path, mmap_mode='r')
        
        # B. 构建 Sparse 特征索引 (不加载内容)
        sparse_path = r"F:\LegalAgent\backend\sparse-embedding\data\train_sparse_features.jsonl"
        if os.path.exists(sparse_path):
            print(f"正在构建 Sparse 特征索引 (避免内存占用): {sparse_path}")
            self.sparse_file_handle = open(sparse_path, 'r', encoding='utf-8')
            self.sparse_file_offsets = []
            
            # 快速扫描文件位置
            while True:
                offset = self.sparse_file_handle.tell()
                line = self.sparse_file_handle.readline()
                if not line:
                    break
                self.sparse_file_offsets.append(offset)
            print(f"Sparse 索引构建完成: {len(self.sparse_file_offsets)} 条")

    def get_sparse_item(self, idx):
        """按需从磁盘读取稀疏特征"""
        if self.sparse_file_handle is None or idx >= len(self.sparse_file_offsets):
            return {}
        self.sparse_file_handle.seek(self.sparse_file_offsets[idx])
        line = self.sparse_file_handle.readline()
        try:
            return json.loads(line)
        except:
            return {}

    def _calculate_schema_keywords_score_fast(self, q_idx, c_idx):
        """
        重写评分逻辑：直接使用索引，避免 O(N) 查找
        """
        score = 0
        
        # 按需读取
        query_sparse = self.get_sparse_item(q_idx)
        candidate_sparse = self.get_sparse_item(c_idx)
        
        # 获取sparse_extraction部分
        query_extraction = query_sparse.get('sparse_extraction', {})
        candidate_extraction = candidate_sparse.get('sparse_extraction', {})
        
        # 1. 关键词重合度
        q_kws = set(query_extraction.get('keyword_counts', {}).keys())
        c_kws = set(candidate_extraction.get('keyword_counts', {}).keys())
        
        if q_kws and c_kws:
            intersection = len(q_kws & c_kws)
            union = len(q_kws | c_kws)
            if union > 0:
                score += (intersection / union) * 15
        
        # 2. 犯罪类型重合度
        q_crimes = set(query_extraction.get('crime_counts', {}).keys())
        c_crimes = set(candidate_extraction.get('crime_counts', {}).keys())
        
        if q_crimes and c_crimes:
            intersection = len(q_crimes & c_crimes)
            union = len(q_crimes | c_crimes)
            if union > 0:
                score += (intersection / union) * 15
                
        return min(30, score)

    def _batch_embedding_score_torch(self, query_emb_tensor, candidate_embs_tensor):
        """GPU 批量计算 Embedding 分数"""
        q_norm = F.normalize(query_emb_tensor, p=2, dim=1)
        c_norm = F.normalize(candidate_embs_tensor, p=2, dim=1)
        similarity = torch.mm(q_norm, c_norm.t()).squeeze(0)
        
        scores = torch.zeros_like(similarity)
        mask_1 = similarity >= 0.95
        scores[mask_1] = 30.0
        mask_2 = (similarity >= 0.90) & (similarity < 0.95)
        scores[mask_2] = 25.0 + (similarity[mask_2] - 0.90) * 100.0
        mask_3 = (similarity >= 0.80) & (similarity < 0.90)
        scores[mask_3] = 15.0 + (similarity[mask_3] - 0.80) * 100.0
        mask_4 = (similarity >= 0.70) & (similarity < 0.80)
        scores[mask_4] = 5.0 + (similarity[mask_4] - 0.70) * 100.0
        mask_5 = similarity < 0.70
        scores[mask_5] = torch.clamp((similarity[mask_5] - 0.60) * 100.0, min=0.0)
        return scores

    def run_fast_sampling(self, output_path, num_queries=100, positives_per_query=10, run_clean=False,
                          initial_pool_factor=3, top_k_for_fine=500, quality_metric_topk=5):
        
        if not self.all_data:
            self.load_and_index()

        if self.fact_embeddings is None:
            raise ValueError("必须加载 Embedding 才能使用快速采样模式")

        # 1. 初选 Query (优先选择单标签且覆盖不同罪名)
        acc_to_indices = defaultdict(list)
        all_single_indices = []
        
        for idx, item in enumerate(self.all_data):
            accs = item.get('meta', {}).get('accusation', [])
            # 仅保留单标签数据
            if len(accs) == 1:
                acc_name = accs[0]
                acc_to_indices[acc_name].append(idx)
                all_single_indices.append(idx)

        available_accs = list(acc_to_indices.keys())
        print(f"发现 {len(available_accs)} 种单标签罪名，共 {len(all_single_indices)} 条数据")

        target_count = num_queries * initial_pool_factor
        initial_queries = []
        
        # 策略A: 尽量覆盖每种罪名至少取1个
        for acc in available_accs:
            initial_queries.append(random.choice(acc_to_indices[acc]))
            
        # 策略B: 如果数量不够，从剩余单标签数据中随机补充
        current_count = len(initial_queries)
        if current_count < target_count:
            needed = target_count - current_count
            # 排除已选的
            chosen_set = set(initial_queries)
            remaining_candidates = [i for i in all_single_indices if i not in chosen_set]
            
            if len(remaining_candidates) >= needed:
                initial_queries.extend(random.sample(remaining_candidates, needed))
            else:
                initial_queries.extend(remaining_candidates)
        
        # 如果数量超了（比如罪名种类特别多），随机截断
        if len(initial_queries) > target_count:
            initial_queries = random.sample(initial_queries, target_count)
        
        print(f"初选 {len(initial_queries)} 条单标签 query，开始处理...")

        query_results = []
        
        # 2. 主循环
        for q_idx in tqdm(initial_queries, desc="Processing"):
            query_item = self.all_data[q_idx]
            q_accs = query_item.get('meta', {}).get('accusation', [])

            # A. 智能构建候选集 (防止内存爆炸)
            candidate_indices = set()
            for acc in q_accs:
                cands = self.accusation_index.get(acc, [])
                # 关键优化：如果某罪名候选太多，先采样再合并
                if len(cands) > 10000: 
                    cands = random.sample(cands, 10000)
                candidate_indices.update(cands)
            
            if q_idx in candidate_indices:
                candidate_indices.remove(q_idx)
            
            candidate_list = list(candidate_indices)
            if not candidate_list:
                continue
            
            # 二次保险：总候选数限制
            if len(candidate_list) > 30000:
                candidate_list = random.sample(candidate_list, 30000)

            # B. GPU 粗筛 (Embedding)
            q_emb = self.fact_embeddings[q_idx]
            c_embs = self.fact_embeddings[candidate_list]
            
            with torch.no_grad():
                q_tensor = torch.from_numpy(q_emb).unsqueeze(0).to(self.device)
                c_tensor = torch.from_numpy(c_embs).to(self.device)
                emb_scores = self._batch_embedding_score_torch(q_tensor, c_tensor).cpu().numpy()

            # C. 选取 Top-K 进入精排
            c_scores_indices = list(zip(emb_scores, candidate_list))
            c_scores_indices.sort(key=lambda x: x[0], reverse=True)
            top_k_candidates = c_scores_indices[:top_k_for_fine]

            # D. CPU 精排 (Schema + Legal)
            final_candidates = []
            for emb_score, c_idx in top_k_candidates:
                candidate_item = self.all_data[c_idx]
                
                # 使用优化后的快速评分方法 (传入索引)
                schema_score = self._calculate_schema_keywords_score_fast(q_idx, c_idx)
                
                legal_score = self._calculate_accusation_articles_imprisonment_score(
                    query_item.get('meta', {}), candidate_item.get('meta', {})
                )
                
                total_score = emb_score + schema_score + legal_score
                
                final_candidates.append({
                    'candidate_item': candidate_item,
                    'total_score': float(total_score),
                    'score_info': {
                        'total_score': float(total_score),
                        'embedding_score': float(emb_score),
                        'schema_keywords_score': float(schema_score),
                        'acc_article_imp_score': float(legal_score)
                    }
                })

            final_candidates.sort(key=lambda x: x['total_score'], reverse=True)
            positives = final_candidates[:positives_per_query]
            
            if not positives:
                continue

            # 计算质量分
            top_for_quality = final_candidates[:max(quality_metric_topk, 1)]
            quality = float(sum(x['total_score'] for x in top_for_quality) / len(top_for_quality))

            query_results.append({
                'query_idx': q_idx,
                'query_item': query_item,
                'positives': [x['candidate_item'] for x in positives],
                'score_details': [x['score_info'] for x in positives],
                'quality': quality
            })

        # 3. 最终筛选与保存
        query_results.sort(key=lambda x: x['quality'], reverse=True)
        final_results = query_results[:num_queries]

        dataset = []
        for rec in final_results:
            dataset.append({
                "query": rec['query_item'],
                "query_idx": rec['query_idx'],
                "positives": rec['positives'],
                "positives_count": len(rec['positives']),
                "score_details": rec['score_details'],
                "quality": rec['quality']
            })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"完成：初选 {len(initial_queries)} -> 最终 {len(dataset)} ，保存到 {output_path}")

        if run_clean:
            self._run_cleaner(output_path)

    def _run_cleaner(self, input_path):
        print("\n=== 触发自动清洗流程 ===")
        old_input = getattr(CleanConfig, "INPUT_FILE", None)
        CleanConfig.INPUT_FILE = input_path
        CleanConfig.OUTPUT_CLEAN_FILE = input_path.replace(".jsonl", ".cleaned.jsonl")
        CleanConfig.OUTPUT_DIRTY_FILE = input_path.replace(".jsonl", ".dirty.jsonl")

        try:
            import importlib
            import clean_prepared_datasets
            importlib.reload(clean_prepared_datasets)
            clean_prepared_datasets.clean_data()
        except Exception as e:
            print(f"清洗过程出错: {e}")
        finally:
            if old_input: CleanConfig.INPUT_FILE = old_input

if __name__ == "__main__":
    TRAIN_PATH = r"F:\LegalAgent\dataset\final_all_data\first_stage\train.json"
    OUTPUT_PATH = r"F:\LegalAgent\backend\evaluate\sample_100x10_fast.jsonl"
    
    sampler = FastSampler(TRAIN_PATH)
    sampler.run_fast_sampling(
        OUTPUT_PATH, 
        num_queries=100, 
        positives_per_query=10,
        run_clean=True
    )