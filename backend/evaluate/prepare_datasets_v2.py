import json
import random
import os
import numpy as np
from tqdm import tqdm
from config import EvalConfig


class GoldDatasetBuilder:
    def __init__(self, data_path):
        self.data_path = data_path
        self.all_data = []
        self.accusation_index = {}
        # 加载嵌入向量
        self.fact_embeddings = None
        self.sparse_features = {}
        self._load_embeddings()

    def _load_embeddings(self):
        """加载预计算的嵌入向量"""
        # 加载BGE-M3嵌入向量
        embedding_path = r"F:\LegalAgent\backend\data\fact_embeddings\fact_embeddings_bge-m3.npy"
        if os.path.exists(embedding_path):
            self.fact_embeddings = np.load(embedding_path)
            print(f"已加载嵌入向量: {embedding_path}, shape: {self.fact_embeddings.shape}")
        
        # 加载sparse特征 - 使用train_sparse_features.jsonl的实际格式
        sparse_path = r"F:\LegalAgent\backend\sparse-embedding\data\train_sparse_features.jsonl"
        if os.path.exists(sparse_path):
            with open(sparse_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        item = json.loads(line)
                        # 使用行索引作为doc_id，因为数据中可能没有明确的doc_id
                        self.sparse_features[idx] = item
            print(f"已加载sparse特征: {sparse_path}, count: {len(self.sparse_features)}")

    def load_and_index(self):
        """加载数据并建立基于罪名的倒排索引"""
        print(f"正在加载全量数据: {self.data_path} ...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(tqdm(f)):
                if not line.strip():
                    continue
                item = json.loads(line)
                self.all_data.append(item)
                
                meta = item.get('meta', {})
                accusations = meta.get('accusation', [])
                for acc in accusations:
                    if acc not in self.accusation_index:
                        self.accusation_index[acc] = []
                    self.accusation_index[acc].append(idx)
        
        print(f"数据加载完成，共 {len(self.all_data)} 条。索引构建完成。")

    def _parse_imprisonment(self, meta):
        """解析刑期"""
        term = meta.get('term_of_imprisonment', {})
        return {
            'death': term.get('death_penalty', False),
            'life': term.get('life_imprisonment', False),
            'months': term.get('imprisonment', 0)
        }

    def _calculate_embedding_score(self, query_idx, candidate_idx):
        """
        Embedding模块评分 (30分)
        Top-1: 30分, Top-3: 25-30分, Top-10: 快速衰减, Top-30后接近0
        """
        if self.fact_embeddings is None or query_idx >= len(self.fact_embeddings) or candidate_idx >= len(self.fact_embeddings):
            return 0
        
        # 计算余弦相似度
        query_emb = self.fact_embeddings[query_idx].reshape(1, -1)
        candidate_emb = self.fact_embeddings[candidate_idx].reshape(1, -1)
        
        # 余弦相似度计算
        dot_product = np.dot(query_emb, candidate_emb.T)[0][0]
        norm_query = np.linalg.norm(query_emb)
        norm_candidate = np.linalg.norm(candidate_emb)
        
        if norm_query == 0 or norm_candidate == 0:
            return 0
        
        similarity = dot_product / (norm_query * norm_candidate)
        
        # 根据相似度映射到30分制
        if similarity >= 0.95:  # Top-1水平
            return 30
        elif similarity >= 0.90:  # Top-3水平
            return 25 + (similarity - 0.90) * 100  # 25-30分
        elif similarity >= 0.80:  # Top-10水平
            return 15 + (similarity - 0.80) * 100  # 15-25分
        elif similarity >= 0.70:  # Top-30水平
            return 5 + (similarity - 0.70) * 100  # 5-15分
        else:
            return max(0, (similarity - 0.60) * 100)  # 0-5分
        
        return 0

    def _calculate_schema_keywords_score(self, query_item, candidate_item):
        """
        Schema/Keywords模块评分 (30分)
        基于结构重合度，关键项优先、非线性加分
        """
        score = 0
        
        # 从sparse_features中获取对应索引的数据
        query_idx = self.all_data.index(query_item) if query_item in self.all_data else -1
        candidate_idx = self.all_data.index(candidate_item) if candidate_item in self.all_data else -1
        
        if query_idx == -1 or candidate_idx == -1:
            return 0
            
        query_sparse = self.sparse_features.get(query_idx, {})
        candidate_sparse = self.sparse_features.get(candidate_idx, {})
        
        # 获取sparse_extraction部分
        query_extraction = query_sparse.get('sparse_extraction', {})
        candidate_extraction = candidate_sparse.get('sparse_extraction', {})
        
        # 计算关键词重合度
        query_keywords_dict = query_extraction.get('keyword_counts', {})
        candidate_keywords_dict = candidate_extraction.get('keyword_counts', {})
        
        query_keywords = set(query_keywords_dict.keys())
        candidate_keywords = set(candidate_keywords_dict.keys())
        
        if query_keywords and candidate_keywords:
            intersection = len(query_keywords & candidate_keywords)
            union = len(query_keywords | candidate_keywords)
            
            if union > 0:
                keyword_similarity = intersection / union
                score += keyword_similarity * 15  # 关键词部分15分
        
        # 计算犯罪类型重合度
        query_crimes = set(query_extraction.get('crime_counts', {}).keys())
        candidate_crimes = set(candidate_extraction.get('crime_counts', {}).keys())
        
        if query_crimes and candidate_crimes:
            intersection = len(query_crimes & candidate_crimes)
            union = len(query_crimes | candidate_crimes)
            
            if union > 0:
                crime_similarity = intersection / union
                score += crime_similarity * 15  # 犯罪类型部分15分
        
        return min(30, score)

    def _calculate_accusation_articles_imprisonment_score(self, query_meta, candidate_meta):
        """
        罪名/法条/判刑模块评分 (40分)
        分层设计：罪名一致性(15分) + 法条接近度(15分) + 判刑接近度(10分)
        """
        score = 0
        
        # 1. 罪名一致性 (15分)
        q_acc = set(query_meta.get('accusation', []))
        c_acc = set(candidate_meta.get('accusation', []))
        
        if q_acc and c_acc:
            if q_acc == c_acc:
                score += 15  # 完全一致
            elif q_acc & c_acc:  # 有交集
                acc_similarity = len(q_acc & c_acc) / len(q_acc | c_acc)
                score += acc_similarity * 15
        elif not q_acc and not c_acc:
            score += 15  # 都没有罪名也算一致
        
        # 2. 法条接近度 (15分)
        q_articles = set(query_meta.get('relevant_articles', []))
        c_articles = set(candidate_meta.get('relevant_articles', []))
        
        if q_articles and c_articles:
            if q_articles == c_articles:
                score += 15  # 完全一致
            elif q_articles & c_articles:  # 有交集
                article_similarity = len(q_articles & c_articles) / len(q_articles | c_articles)
                score += article_similarity * 15
        elif not q_articles and not c_articles:
            score += 15  # 都没有法条也算一致
        
        # 3. 判刑接近度 (10分)
        q_imp = self._parse_imprisonment(query_meta)
        c_imp = self._parse_imprisonment(candidate_meta)
        
        # 检查死刑和无期徒刑
        if q_imp['death'] and c_imp['death']:
            score += 10
        elif q_imp['life'] and c_imp['life']:
            score += 10
        elif q_imp['death'] != c_imp['death'] or q_imp['life'] != c_imp['life']:
            # 一个有死刑/无期，一个没有，直接扣分
            score += 0
        else:
            # 都是有期徒刑，计算月数差异
            q_months = q_imp['months']
            c_months = c_imp['months']
            
            if q_months == c_months:
                score += 10
            elif q_months == 0 or c_months == 0:  # 其中一个是0
                score += 0 if abs(q_months - c_months) > 12 else 5
            else:
                # 计算比例差异
                ratio_diff = abs(q_months - c_months) / max(q_months, c_months)
                if ratio_diff <= 0.1:  # 10%以内
                    score += 10
                elif ratio_diff <= 0.3:  # 30%以内
                    score += 7
                elif ratio_diff <= 0.5:  # 50%以内
                    score += 4
                else:
                    score += 1
        
        return min(40, score)

    def _calculate_comprehensive_score(self, query_idx, query_item, candidate_idx, candidate_item):
        """
        综合评分：模块内取极值，模块间加权融合
        Embedding: 30分 + Schema/Keywords: 30分 + 罪名/法条/判刑: 40分 = 100分
        """
        embedding_score = self._calculate_embedding_score(query_idx, candidate_idx)
        schema_keywords_score = self._calculate_schema_keywords_score(query_item, candidate_item)
        acc_article_imp_score = self._calculate_accusation_articles_imprisonment_score(
            query_item.get('meta', {}), candidate_item.get('meta', {})
        )
        
        total_score = embedding_score + schema_keywords_score + acc_article_imp_score
        return {
            'total_score': total_score,
            'embedding_score': embedding_score,
            'schema_keywords_score': schema_keywords_score,
            'acc_article_imp_score': acc_article_imp_score
        }

    def build(self, output_file, num_queries=500, num_positives_per_query=55):
        """构建评测集"""
        if not self.all_data:
            self.load_and_index()

        print(f"开始构建评测集 (Target: {num_queries} queries, {num_positives_per_query} positives each)...")
        print("过滤条件: fact 长度必须 >= 100")
        
        dataset = []
        all_indices = list(range(len(self.all_data)))
        random.shuffle(all_indices)

        count = 0
        pbar = tqdm(total=num_queries)
        
        for idx in all_indices:
            if count >= num_queries:
                break
            
            query_item = self.all_data[idx]
            
            if len(query_item.get('fact', '')) < 100:
                continue

            q_meta = query_item.get('meta', {})
            q_accs = q_meta.get('accusation', [])
            
            if not q_accs: 
                continue

            candidate_indices = set()
            for acc in q_accs:
                if acc in self.accusation_index:
                    candidates = self.accusation_index[acc]
                    if len(candidates) > 2500:
                        candidate_indices.update(random.sample(candidates, 2500))
                    else:
                        candidate_indices.update(candidates)
            
            if idx in candidate_indices:
                candidate_indices.remove(idx)

            # 计算所有候选的综合评分
            scored_candidates = []
            for c_idx in candidate_indices:
                candidate_item = self.all_data[c_idx]
                
                if len(candidate_item.get('fact', '')) < 100:
                    continue

                score_info = self._calculate_comprehensive_score(
                    idx, query_item, c_idx, candidate_item
                )
                
                scored_candidates.append({
                    'candidate_item': candidate_item,
                    'score_info': score_info,
                    'candidate_idx': c_idx
                })
            
            # 按总分排序，选择top-k
            scored_candidates.sort(key=lambda x: x['score_info']['total_score'], reverse=True)
            top_candidates = scored_candidates[:num_positives_per_query]
            
            if len(top_candidates) >= num_positives_per_query: 
                dataset.append({
                    "query": query_item,
                    "positives": [item['candidate_item'] for item in top_candidates],
                    "positives_count": len(top_candidates),
                    "score_details": [item['score_info'] for item in top_candidates],
                    "query_idx": idx
                })
                count += 1
                pbar.update(1)

        pbar.close()
        
        print(f"写入结果到 {output_file} ...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("Finished")


if __name__ == "__main__":
    INPUT_FILE = EvalConfig.RAW_DATA_PATH
    OUTPUT_FILE = EvalConfig.EVAL_DATASET_PATH
    if os.path.exists(INPUT_FILE):
        builder = GoldDatasetBuilder(INPUT_FILE)
        builder.build(OUTPUT_FILE, num_queries=EvalConfig.NUM_queries, num_positives_per_query=EvalConfig.NUM_positives_per_query)
    else:
        print(f"error: can't find {INPUT_FILE}")