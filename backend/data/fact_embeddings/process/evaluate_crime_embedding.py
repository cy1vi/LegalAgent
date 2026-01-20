import json
import numpy as np
import pickle
from tqdm import tqdm
import random
import os

def evaluate_crime_centroids(
    embedding_path: str,
    data_path: str,
    centroid_path: str,
    sample_size: int = 20000,
    top_k_list=(1, 3, 5, 10),
    seed: int = 42
):
    """
    使用 crime_centroids.pkl 对随机样本进行罪名预测，并计算 Recall@K。
    
    Args:
        embedding_path: fact_embeddings.npy 路径
        data_path: train.json (JSONL)
        centroid_path: crime_centroids.pkl 路径
        sample_size: 随机抽取样本数
        top_k_list: 要评估的 K 值
        seed: 随机种子
    """
    print("🔍 Loading embeddings...")
    embeddings = np.load(embedding_path)  # (N, D)
    N, D = embeddings.shape
    print(f"✅ Embeddings: {N:,} samples, {D} dims")

    print("🔍 Loading crime centroids...")
    with open(centroid_path, 'rb') as f:
        crime_stats = pickle.load(f)
    
    # 构建罪名列表和 centroid 矩阵
    crimes = sorted(crime_stats.keys())  # 固定顺序
    num_crimes = len(crimes)
    centroids = np.stack([crime_stats[crime]["centroid"] for crime in crimes])  # (C, D)
    print(f"✅ Loaded {num_crimes} crime centroids.")

    # 验证：是否归一化？（BGE-M3 默认是）
    norm_diff = np.abs(np.linalg.norm(centroids, axis=1) - 1.0)
    if norm_diff.max() > 1e-3:
        print("⚠️ Warning: Centroids not normalized! Normalizing now.")
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # 获取所有真实罪名（用于抽样）
    print("🔍 Reading all accusations for sampling...")
    all_accusations = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                acc = item.get("meta", {}).get("accusation", [])
                all_accusations.append(acc)
            except:
                all_accusations.append([])

    if len(all_accusations) != N:
        print(f"❗ Data length mismatch! Using min({len(all_accusations)}, {N})")
        N = min(len(all_accusations), len(embeddings))
        embeddings = embeddings[:N]
        all_accusations = all_accusations[:N]

    # 随机抽样
    random.seed(seed)
    np.random.seed(seed)
    indices = np.random.choice(N, size=min(sample_size, N), replace=False)
    print(f"🎲 Selected {len(indices)} random samples for evaluation.")

    # 初始化计数器
    hit_counts = {k: 0 for k in top_k_list}
    total_valid = 0  # 排除无罪名样本

    print("🔄 Running evaluation...")
    for idx in tqdm(indices, desc="Evaluating"):
        true_crimes = all_accusations[idx]
        if not true_crimes:
            continue
        total_valid += 1

        query_emb = embeddings[idx].reshape(1, -1)  # (1, D)

        # 计算与所有 centroid 的相似度（cosine = dot product if normalized）
        similarities = np.dot(query_emb, centroids.T).flatten()  # (C,)

        # 获取 top-K 预测罪名
        top_k_indices = np.argsort(-similarities)  # descending
        predicted_crimes = [crimes[i] for i in top_k_indices]

        # 检查 recall@k
        true_set = set(true_crimes)
        for k in top_k_list:
            top_k_pred = set(predicted_crimes[:k])
            if true_set & top_k_pred:  # 有交集即命中
                hit_counts[k] += 1

    # 计算 recall
    print("\n📊 Evaluation Results:")
    print(f"Total valid samples: {total_valid:,}")
    recalls = {}
    for k in top_k_list:
        recall = hit_counts[k] / total_valid if total_valid > 0 else 0.0
        recalls[k] = recall
        print(f"  Recall@{k:2d}: {recall:.4f} ({hit_counts[k]}/{total_valid})")

    # 保存结果
    result = {
        "sample_size": len(indices),
        "valid_samples": total_valid,
        "recall": {f"R@{k}": round(recalls[k], 4) for k in top_k_list}
    }
    result_path = os.path.splitext(centroid_path)[0] + "_eval.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to {result_path}")

    return recalls


if __name__ == "__main__":
    EMBEDDING_PATH = r"F:\LegalAgent\backend\data\fact_embeddings\fact_embeddings_bge-m3.npy"
    DATA_PATH = r"F:\LegalAgent\dataset\final_all_data\first_stage\train.json"
    CENTROID_PATH = r"F:\LegalAgent\backend\data\fact_embeddings\crime_centroids.pkl"

    evaluate_crime_centroids(
        embedding_path=EMBEDDING_PATH,
        data_path=DATA_PATH,
        centroid_path=CENTROID_PATH,
        sample_size=200000,
        top_k_list=(1, 3, 5, 10),
        seed=42
    )