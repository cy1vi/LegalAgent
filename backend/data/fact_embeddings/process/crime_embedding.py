import json
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm

def compute_crime_centroids(embedding_path: str, data_path: str, output_path: str):
    """
    计算每个罪名的平均嵌入向量及统计信息。
    
    Args:
        embedding_path: str, path to embeddings.npy (shape: N x D)
        data_path: str, path to data.json (list of dicts) or .jsonl
        output_path: str, output .pkl file path
    """
    print("Loading embeddings...")
    embeddings = np.load(embedding_path)  # (N, D)
    N, D = embeddings.shape
    print(f"Embeddings loaded: {N} samples, {D} dimensions")

    print("Loading metadata...")
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Warning: Skip invalid JSON at line {line_num}: {e}")
                continue
    assert len(data) == N, f"Data length ({len(data)}) != embeddings ({N})"

    # Step 1: 收集每个罪名对应的 embedding 列表
    crime_embeddings = defaultdict(list)  # crime -> List[np.ndarray]

    print("Aggregating embeddings by accusation...")
    for i, item in enumerate(tqdm(data, total=N)):
        meta = item.get("meta", {})
        accusations = meta.get("accusation", [])
        if not accusations:
            continue
        emb = embeddings[i]  # (D,)
        for crime in accusations:
            crime_embeddings[crime].append(emb)

    # Step 2: 计算每个罪名的统计量
    crime_stats = {}
    print("Computing centroids and stats...")
    for crime, emb_list in tqdm(crime_embeddings.items()):
        if not emb_list:
            continue
        arr = np.stack(emb_list)  # (M, D)
        centroid = np.mean(arr, axis=0)  # (D,)
        std_per_dim = np.std(arr, axis=0)  # (D,)
        avg_std = float(np.mean(std_per_dim))  # scalar

        # 计算每个点到 centroid 的 L2 距离
        distances = np.linalg.norm(arr - centroid, axis=1)  # (M,)
        max_dist = float(np.max(distances))
        min_dist = float(np.min(distances))

        crime_stats[crime] = {
            "centroid": centroid,      # np.ndarray, shape (D,)
            "count": len(emb_list),
            "avg_std": avg_std,        # 平均维度标准差
            "std_per_dim": std_per_dim, # 可选：保留 per-dim（但会很大）
            "max_dist_to_centroid": max_dist,
            "min_dist_to_centroid": min_dist,
            "mean_distance": float(np.mean(distances))
        }

    # Step 3: 保存结果（pickle 可保留 np.ndarray）
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(crime_stats, f)

    # 可选：也保存一个 JSON 版本（但 centroid 会转成 list，体积大）
    json_output = output_path.replace('.pkl', '_summary.json')
    summary = {
        crime: {
            k: v for k, v in stats.items()
            if k not in ['centroid', 'std_per_dim']  # 只保留标量
        }
        for crime, stats in crime_stats.items()
    }
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Done! Processed {len(crime_stats)} crimes.")
    print(f"Top 5 crimes by count:")
    for crime, stats in sorted(crime_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:5]:
        print(f"  {crime}: {stats['count']} cases")


if __name__ == "__main__":
    # 配置路径
    EMBEDDING_PATH = r"F:\LegalAgent\backend\data\fact_embeddings\fact_embeddings_bge-m3.npy"      
    DATA_PATH = r"F:\LegalAgent\dataset\final_all_data\first_stage\train.json"                
    OUTPUT_PATH = r"F:\LegalAgent\backend\data\fact_embeddings\crime_centroids.pkl"

    compute_crime_centroids(EMBEDDING_PATH, DATA_PATH, OUTPUT_PATH)