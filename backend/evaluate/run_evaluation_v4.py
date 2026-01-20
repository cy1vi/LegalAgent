import json
import requests
import numpy as np
import os
import hashlib
from tqdm import tqdm
from typing import List, Dict, Any
from config import EvalConfig
import argparse

class LegalEvaluator:
    def __init__(self, eval_file_path, service_url=EvalConfig.SERVICE_URL, eval_type="dense"):
        self.eval_file_path = eval_file_path
        self.service_url = service_url
        self.eval_type = eval_type  # "dense", "sparse", "rerank"
        
        # 根据评估类型设置不同的服务URL
        if eval_type == "sparse":
            self.service_url = "http://localhost:4240/search"  # 稀疏检索服务端口
        elif eval_type == "dense":
            self.service_url = "http://localhost:4241/search"  # 稠密检索服务端口
        elif eval_type == "rerank":
            self.service_url = "http://localhost:8000/search"  # 重排序服务端口 
        
        self.dataset = []
        self.output_report_path = os.path.join(
            os.path.dirname(eval_file_path), 
            f"evaluate_summary_{eval_type}.txt"
        )
        self.k_list = EvalConfig.K_LIST
        self.tasks = ['gold', 'accusation', 'article', 'imprisonment']
        
        # Macro 统计容器: 存每个 Query 的指标
        self.metrics = {
            task: {k: {'p': [], 'r': [], 'f1': [], 'hit': []} for k in self.k_list}
            for task in self.tasks
        }
        
        self.max_positives = EvalConfig.MAX_POSITIVES 

    def load_data(self):
        print(f"正在加载评测集: {self.eval_file_path} ...")
        if not os.path.exists(self.eval_file_path):
            raise FileNotFoundError(f"找不到文件: {self.eval_file_path}")
            
        with open(self.eval_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.dataset.append(json.loads(line))
        print(f"加载完成，共 {len(self.dataset)} 条测试 Query。")

    def _get_text_hash(self, text):
        if not text: return ""
        return hashlib.md5(text.strip().encode('utf-8')).hexdigest()

    def _parse_imprisonment(self, term_data):
        if not term_data:
            return {'death': False, 'life': False, 'months': 0}
        return {
            'death': term_data.get('death_penalty', False),
            'life': term_data.get('life_imprisonment', False),
            'months': term_data.get('imprisonment', 0)
        }

    def check_attributes(self, query_meta, doc_res):
        # 确保从 positives 中提取的 doc_res 有正确的字段结构
        q_acc = set(query_meta.get('accusation', []))
        d_acc = set(doc_res.get('accusation', []))
        acc_match = (q_acc == d_acc)

        q_art = set(query_meta.get('relevant_articles', []))
        d_art = set(doc_res.get('relevant_articles', []))
        art_match = (q_art == d_art)

        q_imp = self._parse_imprisonment(query_meta.get('term_of_imprisonment', {}))
        d_imp = self._parse_imprisonment(doc_res.get('imprisonment', {}))
        imp_match = False

        if q_imp['death'] == d_imp['death'] and q_imp['life'] == d_imp['life']:
            if q_imp['death'] or q_imp['life']:
                imp_match = True
            else:
                q_m = q_imp['months']
                d_m = d_imp['months']
                tolerance = max(6, q_m * 0.2)
                if abs(q_m - d_m) <= tolerance:
                    imp_match = True

        return acc_match, art_match, imp_match
    def calculate_metrics(self, hits_count, top_k, total_positives):
        precision = hits_count / top_k if top_k else 0
        recall = hits_count / total_positives if total_positives > 0 else 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        return precision, recall, f1
    
    def load_data(self):
        print(f"正在加载评测集: {self.eval_file_path} ...")
        if not os.path.exists(self.eval_file_path):
            raise FileNotFoundError(f"找不到文件: {self.eval_file_path}")
        with open(self.eval_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # 确保 positives 中的每个条目都有正确的字段
                    for pos in item.get('positives', []):
                        if 'accusation' not in pos:
                            pos['accusation'] = []
                        if 'relevant_articles' not in pos:
                            pos['relevant_articles'] = []
                        if 'imprisonment' not in pos:
                            pos['imprisonment'] = {}
                    self.dataset.append(item)
        print(f"加载完成，共 {len(self.dataset)} 条测试 Query。")

    def run(self):
        if not self.dataset:
            self.load_data()
        print(f"开始评测 {self.eval_type} 检索 (Top-K: 10)...")

        BATCH_SIZE = EvalConfig.BATCH_SIZE
        
        # 根据评估类型设置不同的URL和请求格式
        if self.eval_type == "rerank":
            batch_url = self.service_url.replace("/search", "/batch_search")
            # 测试连接时使用正确的参数格式
            test_payload = {
                "queries": ["test"],
                "top_k": 1,
                "mode": "hybrid",
                "rerank": True
            }
        else:
            batch_url = self.service_url.replace("/search", "/batch_search")
            test_payload = {"facts": ["test"], "top_k": 1}

        try:
            # 测试连接
            test_response = requests.post(batch_url, json=test_payload)
            if test_response.status_code != 200:
                print(f"错误: 无法连接到 {self.eval_type} 检索服务。")
                print(f"尝试连接的URL: {batch_url}")
                return
        except Exception as e:
            print(f"错误: 无法连接到 {self.eval_type} 检索服务。错误信息: {e}")
            return

        for i in tqdm(range(0, len(self.dataset), BATCH_SIZE), desc=f"{self.eval_type} evaluation"):
            batch_items = self.dataset[i : i + BATCH_SIZE]
            batch_facts = [item['query'].get('fact', '') for item in batch_items]

            # 根据评估类型构造不同的请求
            if self.eval_type == "rerank":
                payload = {
                    "queries": batch_facts,
                    "top_k": 12,
                    "mode": "hybrid",
                    "rerank": True
                }
            else:
                payload = {"facts": batch_facts, "top_k": 12}

            try:
                resp = requests.post(batch_url, json=payload)
                if resp.status_code != 200:
                    print(f"API请求失败，状态码: {resp.status_code}, 响应: {resp.text}")
                    continue
                batch_results = resp.json()
                
                # 处理rerank服务返回结果的特殊结构
                if self.eval_type == "rerank":
                    # 检查返回结果结构
                    if "results" in batch_results:
                        batch_results = batch_results["results"]
            except Exception as e:
                print(f"API Error: {e}")
                continue

            # 确保batch_results是列表格式
            if not isinstance(batch_results, list):
                print(f"返回结果格式错误: {type(batch_results)}")
                continue

            for idx, raw_results in enumerate(batch_results):
                # 确保raw_results是列表格式
                if not isinstance(raw_results, list):
                    print(f"单个查询结果格式错误: {type(raw_results)}")
                    continue
                    
                item = batch_items[idx]
                query_fact = item['query'].get('fact', '')
                query_meta = item['query'].get('meta', {})
                query_hash = self._get_text_hash(query_fact)
                
                filtered_results = []
                for res in raw_results:
                    # 处理不同格式的结果
                    if isinstance(res, str):
                        # 如果是字符串，跳过处理
                        continue
                    elif isinstance(res, dict):
                        # 检查字段名，适配不同服务返回格式
                        fact_text = (res.get('fact') or 
                                   res.get('text') or 
                                   res.get('content') or 
                                   res.get('sentence') or
                                   '')
                        if self._get_text_hash(fact_text) == query_hash:
                            continue
                        # 标准化字段名
                        res['fact'] = fact_text
                        filtered_results.append(res)
                    else:
                        continue

                all_positives = item.get('positives', [])
                actual_limit = min(self.max_positives, len(all_positives))
                target_positives = all_positives[:actual_limit]
                total_positives = actual_limit

                gold_hashes = set()
                for pos in target_positives:
                    h = self._get_text_hash(pos.get('fact', ''))
                    if h:
                        gold_hashes.add(h)

                for k in self.k_list:
                    if k > len(filtered_results):
                        continue
                    current_results = filtered_results[:k]
                    hits = {'gold': 0, 'accusation': 0, 'article': 0, 'imprisonment': 0}

                    for res in current_results:
                        res_hash = self._get_text_hash(res.get('fact', ''))
                        if res_hash in gold_hashes:
                            hits['gold'] += 1
                        is_acc, is_art, is_imp = self.check_attributes(query_meta, res)
                        if is_acc:
                            hits['accusation'] += 1
                        if is_art:
                            hits['article'] += 1
                        if is_imp:
                            hits['imprisonment'] += 1

                    for task in self.tasks:
                        h_count = hits[task]
                        p, r, f1 = self.calculate_metrics(h_count, k, total_positives)
                        self.metrics[task][k]['p'].append(p)
                        self.metrics[task][k]['r'].append(r)
                        self.metrics[task][k]['f1'].append(f1)
                        self.metrics[task][k]['hit'].append(1 if h_count > 0 else 0)

        self.generate_report()

    def generate_report(self):
        lines = []
        lines.append("=" * 80)
        lines.append(f"EVALUATION REPORT - {self.eval_type.upper()} RETRIEVAL")
        lines.append(f"Dataset: {self.eval_file_path}")
        lines.append(f"Total Queries: {len(self.dataset)}")
        lines.append(f"Positives per Query: {self.max_positives}")
        lines.append(f"Service URL: {self.service_url}")
        lines.append("=" * 80)
        lines.append("")

        header = f"{'Task':<15} | {'Metric':<10} | {'@3':<10} | {'@5':<10} | {'@10':<10}"
        lines.append(header)
        lines.append("-" * 80)

        for task in self.tasks:
            # Macro Average
            p_3 = np.mean(self.metrics[task][3]['p']) * 100 if self.metrics[task][3]['p'] else 0
            r_3 = np.mean(self.metrics[task][3]['r']) * 100 if self.metrics[task][3]['r'] else 0
            macro_f1_3 = np.mean(self.metrics[task][3]['f1']) * 100 if self.metrics[task][3]['f1'] else 0
            hit_3 = np.mean(self.metrics[task][3]['hit']) * 100 if self.metrics[task][3]['hit'] else 0
            
            p_5 = np.mean(self.metrics[task][5]['p']) * 100 if self.metrics[task][5]['p'] else 0
            r_5 = np.mean(self.metrics[task][5]['r']) * 100 if self.metrics[task][5]['r'] else 0
            macro_f1_5 = np.mean(self.metrics[task][5]['f1']) * 100 if self.metrics[task][5]['f1'] else 0
            hit_5 = np.mean(self.metrics[task][5]['hit']) * 100 if self.metrics[task][5]['hit'] else 0
            
            p_10 = np.mean(self.metrics[task][10]['p']) * 100 if self.metrics[task][10]['p'] else 0
            r_10 = np.mean(self.metrics[task][10]['r']) * 100 if self.metrics[task][10]['r'] else 0
            macro_f1_10 = np.mean(self.metrics[task][10]['f1']) * 100 if self.metrics[task][10]['f1'] else 0
            hit_10 = np.mean(self.metrics[task][10]['hit']) * 100 if self.metrics[task][10]['hit'] else 0


            task_name = task.capitalize()
            lines.append(f"{task_name:<15} | Precision  | {p_3:.2f}%     | {p_5:.2f}%     | {p_10:.2f}%")
            lines.append(f"{'':<15} | Recall     | {r_3:.2f}%     | {r_5:.2f}%     | {r_10:.2f}%")
            lines.append(f"{'':<15} | Hit Rate   | {hit_3:.2f}%     | {hit_5:.2f}%     | {hit_10:.2f}%")
            lines.append(f"{'':<15} | Macro-F1   | {macro_f1_3:.2f}%     | {macro_f1_5:.2f}%     | {macro_f1_10:.2f}%")
            lines.append("-" * 80)

        with open(self.output_report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print('\n'.join(lines))
        print(f"\n详细报告已保存至: {self.output_report_path}")


def main():
    parser = argparse.ArgumentParser(description='Legal Retrieval Evaluator')
    parser.add_argument(
        '--eval_type', 
        type=str, 
        choices=['dense', 'sparse', 'rerank'], 
        default='dense',
        help='选择评估类型: dense (稠密检索), sparse (稀疏检索), rerank (重排序 - 预留接口)'
    )
    parser.add_argument(
        '--eval_file', 
        type=str, 
        default=EvalConfig.EVAL_DATASET_PATH,
        help='评估数据集路径'
    )
    
    args = parser.parse_args()
    
    print(f"开始进行 {args.eval_type} 检索评估...")
    evaluator = LegalEvaluator(
        eval_file_path=args.eval_file,
        eval_type=args.eval_type
    )
    evaluator.run()


if __name__ == "__main__":
    main()