import json
import random
import os
from tqdm import tqdm
from config import EvalConfig


class GoldDatasetBuilder:
    def __init__(self, data_path):
        self.data_path = data_path
        self.all_data = []
        self.accusation_index = {} 

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

    def _get_similarity_level(self, query_meta, candidate_meta):
        """
        判断相似度等级 (1-5)，数字越小越相似。如果不满足任何条件返回 0。
        """
        # 1. 准备数据
        q_acc = set(query_meta.get('accusation', []))
        c_acc = set(candidate_meta.get('accusation', []))
        q_art = set(query_meta.get('relevant_articles', []))
        c_art = set(candidate_meta.get('relevant_articles', []))
        
        q_imp = self._parse_imprisonment(query_meta)
        c_imp = self._parse_imprisonment(candidate_meta)

        # 2. 基础刑期计算
        if q_imp['death'] != c_imp['death'] or q_imp['life'] != c_imp['life']:
            return 0
        
        imp_diff_ratio = 0.0
        is_imp_exact = False
        
        if q_imp['death'] or q_imp['life']:
            is_imp_exact = True
            imp_diff_ratio = 0.0
        else:
            q_m = q_imp['months']
            c_m = c_imp['months']
            if q_m == c_m:
                is_imp_exact = True
                imp_diff_ratio = 0.0
            else:
                base = q_m if q_m > 0 else 1 
                imp_diff_ratio = abs(q_m - c_m) / base

        # 3. 逐级判断 (Waterfall)
        acc_exact = (q_acc == c_acc)
        acc_overlap = bool(q_acc & c_acc)
        art_exact = (q_art == c_art)
        art_overlap = bool(q_art & c_art)

        if acc_exact and art_exact and is_imp_exact: return 1
        if acc_exact and art_exact and imp_diff_ratio <= 0.1: return 2
        if acc_exact and art_exact and imp_diff_ratio <= 0.3: return 3
        if acc_exact and art_overlap and imp_diff_ratio <= 0.3: return 4
        if acc_overlap and art_overlap and imp_diff_ratio <= 0.3: return 5

        return 0

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
            
            if not q_accs: continue 

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

            buckets = {1: [], 2: [], 3: [], 4: [], 5: []}
            
            for c_idx in candidate_indices:
                candidate_item = self.all_data[c_idx]
                
                if len(candidate_item.get('fact', '')) < 100:
                    continue

                level = self._get_similarity_level(q_meta, candidate_item.get('meta', {}))
                if level > 0:
                    buckets[level].append(candidate_item)
            
            final_positives = []
            for level in range(1, 6):
                needed = num_positives_per_query - len(final_positives)
                if needed <= 0:
                    break
                
                current_bucket = buckets[level]
                if len(current_bucket) > needed:
                    final_positives.extend(random.sample(current_bucket, needed))
                else:
                    final_positives.extend(current_bucket)

            if len(final_positives) >= num_positives_per_query: 
                dataset.append({
                    "query": query_item,
                    "positives": final_positives,
                    "positives_count": len(final_positives),
                    "level_distribution": {k: len(v) for k, v in buckets.items()} 
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