import json
import requests
import time
from tqdm import tqdm
from typing import List

# 配置
API_URL = "http://localhost:4240/batch_search"
DATASET_PATH = r"F:\LegalAgent\dataset\final_all_data\first_stage\train.json"
OUTPUT_PATH = r"F:\LegalAgent\backend\sparse-embedding\data\eval_results.jsonl"
BATCH_SIZE = 50  # 每批发送多少条数据 (建议 50-100)
TEST_LIMIT = 1000 # 测试多少条数据 (None 表示跑全量)

def load_dataset(path, limit=None):
    data = []
    print(f"📖 Loading dataset from {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                item = json.loads(line)
                # 假设每行都有 fact 字段
                if "fact" in item:
                    data.append(item)
            except:
                pass
    return data

def run_evaluation():
    # 1. 加载数据
    dataset = load_dataset(DATASET_PATH, limit=TEST_LIMIT)
    total_samples = len(dataset)
    print(f"✅ Loaded {total_samples} samples.")

    # 2. 准备结果文件
    f_out = open(OUTPUT_PATH, 'w', encoding='utf-8')

    # 3. 分批处理
    total_time = 0
    success_count = 0
    
    # 进度条
    pbar = tqdm(total=total_samples, unit="doc")
    
    for i in range(0, total_samples, BATCH_SIZE):
        batch_items = dataset[i : i + BATCH_SIZE]
        batch_facts = [item['fact'] for item in batch_items]
        
        payload = {
            "facts": batch_facts,
            "top_k": 5
        }
        
        try:
            t0 = time.time()
            resp = requests.post(API_URL, json=payload)
            
            if resp.status_code == 200:
                results_list = resp.json()
                batch_time = time.time() - t0
                total_time += batch_time
                
                # 写入结果
                for original_item, search_res in zip(batch_items, results_list):
                    output_line = {
                        "query_fact": original_item['fact'],
                        "ground_truth_meta": original_item.get('meta', {}),
                        "retrieved_docs": search_res
                    }
                    f_out.write(json.dumps(output_line, ensure_ascii=False) + "\n")
                    success_count += 1
            else:
                print(f"❌ Batch failed: {resp.status_code} - {resp.text}")
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            
        pbar.update(len(batch_items))

    pbar.close()
    f_out.close()

    # 4. 统计
    avg_time = (total_time / success_count * 1000) if success_count > 0 else 0
    print("\n" + "="*40)
    print(f"📊 Evaluation Complete")
    print(f"   - Total Processed: {success_count}/{total_samples}")
    print(f"   - Total Time: {total_time:.2f}s")
    print(f"   - Avg Latency: {avg_time:.2f}ms / doc")
    print(f"   - Throughput: {success_count / total_time:.2f} docs/s")
    print(f"💾 Results saved to: {OUTPUT_PATH}")
    print("="*40)

if __name__ == "__main__":
    run_evaluation()