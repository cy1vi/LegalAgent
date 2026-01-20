import json
import random
import os
import requests
import time
from tqdm import tqdm
from collections import defaultdict
from config import EvalConfig

class DatasetBuilderV4:
    def __init__(self, raw_data_path, output_path, search_url="http://127.0.0.1:4241/batch_search"):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.search_url = search_url
        self.all_data = []
        
    def load_data(self):
        """加载原始数据"""
        print(f"正在加载全量数据: {self.raw_data_path} ...")
        with open(self.raw_data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                if line.strip():
                    self.all_data.append(json.loads(line))
        print(f"数据加载完成，共 {len(self.all_data)} 条。")

    def _parse_imprisonment(self, meta):
        """解析刑期结构"""
        term = meta.get('term_of_imprisonment', {})
        return {
            'death': term.get('death_penalty', False),
            'life': term.get('life_imprisonment', False),
            'months': term.get('imprisonment', 0)
        }

    # --- 阶段 0: Query Pool 构建 ---
    def build_query_pool(self, target_size=500):
        """
        构建多样化的 Query 池
        规则: fact >= 100, 分罪名抽样
        """
        print("阶段 0: 构建 Query Pool (目标 500 条, 罪名多样化)...")
        
        # 1. 预筛选和分组
        valid_indices = []
        acc_groups = defaultdict(list)
        
        for idx, item in enumerate(self.all_data):
            fact = item.get('fact', '')
            meta = item.get('meta', {})
            accs = meta.get('accusation', [])
            
            if len(fact) < 100 or not accs:
                continue
                
            # 使用第一个罪名作为分组依据
            primary_acc = accs[0]
            acc_groups[primary_acc].append(idx)
            
        # 2. 轮询抽样 (Round-Robin) 以保证多样性
        selected_indices = []
        groups = list(acc_groups.values())
        # 打乱组的顺序，避免每次都从同一个罪名开始
        random.shuffle(groups) 
        
        while len(selected_indices) < target_size and groups:
            # 每一轮从每个非空组里抽一个
            empty_groups = []
            for group in groups:
                if not group:
                    empty_groups.append(group)
                    continue
                
                # 随机选一个并移除
                chosen = random.choice(group)
                group.remove(chosen)
                selected_indices.append(chosen)
                
                if len(selected_indices) >= target_size:
                    break
            
            # 移除空组
            for g in empty_groups:
                groups.remove(g)
                
        print(f"Query Pool 构建完成，共选中 {len(selected_indices)} 条 Query。")
        return selected_indices

    # --- 阶段 1: 向量召回 ---
    def recall_candidates(self, query_indices, top_k=500):
        """
        调用检索服务进行召回
        """
        print(f"阶段 1: 向量召回 (Top-K={top_k})...")
        
        query_facts = [self.all_data[idx]['fact'] for idx in query_indices]
        
        # 分批请求，避免请求体过大
        batch_size = 2
        all_candidates = [] # List[List[SearchResult]]
        
        for i in tqdm(range(0, len(query_facts), batch_size)):
            batch_facts = query_facts[i : i + batch_size]
            payload = {
                "facts": batch_facts,
                "top_k": top_k
            }
            
            try:
                resp = requests.post(self.search_url, json=payload)
                resp.raise_for_status()
                results = resp.json() # List[List[Dict]]
                all_candidates.extend(results)
            except Exception as e:
                print(f"Error calling search API: {e}")
                # 如果失败，填充空列表避免索引错位
                all_candidates.extend([[] for _ in batch_facts])
                
        return all_candidates

    # --- 阶段 2: 法律结构强过滤 ---
    def filter_candidates(self, query_meta, candidates):
        """
        对召回结果进行硬过滤
        candidates: API返回的字典列表
        """
        q_imp = self._parse_imprisonment(query_meta)
        q_acc = set(query_meta.get('accusation', []))
        q_art = set(query_meta.get('relevant_articles', []))
        
        legal_candidates = []
        
        for cand in candidates:
            # API 返回的结构中，meta 信息可能直接在根目录或 meta 字段下，根据 main.py 的 SearchResult
            # SearchResult: {fact, accusation, relevant_articles, imprisonment, ...}
            c_acc = set(cand.get('accusation', []))
            c_art = set(cand.get('relevant_articles', []))
            c_imp = cand.get('imprisonment', {}) # 已经是解析好的结构或原始结构
            
            # 兼容处理：如果 API 返回的是原始 meta 结构，需解析
            if 'death' not in c_imp: 
                # 尝试解析
                c_imp = self._parse_imprisonment({'term_of_imprisonment': c_imp})

            # 2.1 刑罚类型过滤 (第一刀)
            if q_imp['death'] != c_imp['death']: continue
            if q_imp['life'] != c_imp['life']: continue
            
            # 2.2 罪名 + 法条约束 (第二刀)
            acc_exact = (q_acc == c_acc)
            acc_overlap = bool(q_acc & c_acc)
            art_overlap = bool(q_art & c_art)
            
            # 规则: accusation 完全一致 或 (accusation 有交集 且 relevant_articles 有交集)
            if not (acc_exact or (acc_overlap and art_overlap)):
                continue
                
            # 2.3 刑期差距 (第三刀)
            # 只对有期徒刑生效 (死刑/无期在 2.1 已处理)
            if not q_imp['death'] and not q_imp['life']:
                q_m = q_imp['months']
                c_m = c_imp['months']
                
                base = max(q_m, 1)
                diff_ratio = abs(q_m - c_m) / base
                
                if diff_ratio > 0.3:
                    continue
            
            legal_candidates.append(cand)
            
        return legal_candidates

    # --- 阶段 3: 轻量精排 + 选 Top 10 ---
    def rank_and_select(self, query_meta, candidates):
        """
        规则打分并截断
        """
        q_acc = set(query_meta.get('accusation', []))
        q_art = set(query_meta.get('relevant_articles', []))
        
        scored_candidates = []
        
        for cand in candidates:
            c_acc = set(cand.get('accusation', []))
            c_art = set(cand.get('relevant_articles', []))
            
            acc_exact = (q_acc == c_acc)
            art_exact = (q_art == c_art)
            art_overlap = bool(q_art & c_art)
            
            # 打分公式
            score = 0
            if acc_exact: score += 3
            if art_exact: score += 2
            if art_overlap: score += 1
            
            scored_candidates.append((score, cand))
            
        # 降序排列
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 取 Top 10 的原始对象
        top_10 = [x[1] for x in scored_candidates[:10]]
        
        return top_10

    def run(self):
        if not self.all_data:
            self.load_data()
            
        # Stage 0
        query_indices = self.build_query_pool(target_size=500)
        
        # Stage 1
        # 注意：这里我们传入 query_indices，内部获取 fact 文本去请求 API
        raw_candidates_list = self.recall_candidates(query_indices, top_k=500)
        
        final_dataset = []
        
        print("开始执行 Stage 2 (过滤) & Stage 3 (精排)...")
        pbar = tqdm(total=len(query_indices))
        
        for q_idx, candidates in zip(query_indices, raw_candidates_list):
            # 显式注入 ID 到 Query (使用 copy 避免修改原始数据)
            query_item = self.all_data[q_idx].copy()
            query_item['id'] = q_idx
                        
            candidates = [c for c in candidates if str(c.get('fact_id')) != str(q_idx)]

            q_meta = query_item.get('meta', {})
            
            # Stage 2: Hard Filter
            legal_cands = self.filter_candidates(q_meta, candidates)
            
            # Stage 3: Rank & Select
            gold_cands = self.rank_and_select(q_meta, legal_cands)
            
            # 只有凑够 10 条才要，保证评测质量
            if len(gold_cands) >= 10:
                # 截断到 10 条
                gold_cands = gold_cands[:10]
                
                # 确保 positive 也有 id (API 返回的 SearchResult 有 fact_id)
                for cand in gold_cands:
                    if 'fact_id' in cand:
                        cand['id'] = int(cand['fact_id'])
                
                final_dataset.append({
                    "query_id": q_idx,
                    "query": query_item,
                    "positives": gold_cands,
                    "positives_count": len(gold_cands)
                })
            
            pbar.update(1)
        
        pbar.close()
        
        print(f"构建完成。原始 Query {len(query_indices)} 条，有效 Query {len(final_dataset)} 条。")
        print(f"写入结果到 {self.output_path} ...")
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for item in final_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print("Finished.")

if __name__ == "__main__":
    # 配置路径
    RAW_DATA = EvalConfig.RAW_DATA_PATH
    OUTPUT_FILE = os.path.join(os.path.dirname(EvalConfig.EVAL_DATASET_PATH), "eval_dataset_v4.jsonl")
    
    # 确保服务地址正确 (根据 main.py 的默认配置)
    SEARCH_API = f"http://localhost:4241/batch_search"
    
    if os.path.exists(RAW_DATA):
        # 检查服务是否在线
        try:
            requests.get(f"http://localhost:4241/docs", timeout=3)
            print("检索服务在线，开始构建数据集...")
            
            builder = DatasetBuilderV4(RAW_DATA, OUTPUT_FILE, SEARCH_API)
            builder.run()
            
        except requests.exceptions.ConnectionError:
            print(f"错误: 无法连接到检索服务 {SEARCH_API}。请先运行 main.py 启动服务。")
    else:
        print(f"错误: 找不到原始数据文件 {RAW_DATA}")