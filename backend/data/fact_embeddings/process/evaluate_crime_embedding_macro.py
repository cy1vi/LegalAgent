import json
import numpy as np
import pickle
from tqdm import tqdm
import random
import os
from sklearn.metrics import f1_score

def evaluate_crime_centroids_single_accusation_only(
    embedding_path: str,
    data_path: str,
    centroid_path: str,
    sample_size: int = 50000,
    top_k_list=(1, 3, 5, 10),
    seed: int = 42,
    top_bottom_k: int = 30,
    min_samples_for_macro_f1: int = 50
):
    print("🔍 Loading embeddings...")
    embeddings = np.load(embedding_path)
    N_emb, D = embeddings.shape
    print(f"✅ Embeddings: {N_emb:,} samples, {D} dims")

    # ✅ 归一化 query embeddings
    print("🔄 Normalizing query embeddings (L2)...")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    print("🔍 Loading crime centroids...")
    with open(centroid_path, 'rb') as f:
        crime_stats = pickle.load(f)
    
    crimes = sorted(crime_stats.keys())
    crime_to_idx = {crime: i for i, crime in enumerate(crimes)}
    num_crimes = len(crimes)
    centroids = np.stack([crime_stats[crime]["centroid"] for crime in crimes])
    print(f"✅ Loaded {num_crimes} crime centroids.")

    # ✅ 归一化 centroids
    if np.abs(np.linalg.norm(centroids, axis=1) - 1.0).max() > 1e-3:
        print("🔄 Normalizing centroids (L2)...")
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    print("🔍 Reading ONLY single-accusation samples (|accusation| == 1)...")
    single_accusations = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                acc = item.get("meta", {}).get("accusation", [])
                if isinstance(acc, list) and len(acc) == 1:
                    crime_name = acc[0]
                    if crime_name in crime_to_idx:
                        single_accusations.append(crime_name)
            except Exception:
                continue

    N_data = len(single_accusations)
    print(f"✅ Total single-accusation samples with valid crime: {N_data:,}")

    # 对齐数量
    N = min(N_emb, N_data)
    embeddings = embeddings[:N]
    single_accusations = single_accusations[:N]
    print(f"📌 Using first {N:,} aligned samples.")

    # === 随机采样 5 万样本 ===
    random.seed(seed)
    np.random.seed(seed)
    eval_size = min(sample_size, N)
    indices = np.random.choice(N, size=eval_size, replace=False)
    print(f"🎲 Selected {eval_size} samples for evaluation.")

    # 提取采样子集
    sampled_accusations = [single_accusations[i] for i in indices]
    sampled_embeddings = embeddings[indices]

    # === 在采样样本中统计频率（关键修改）===
    freq_in_sample = {}
    for crime in sampled_accusations:
        freq_in_sample[crime] = freq_in_sample.get(crime, 0) + 1

    unique_crimes_in_sample = list(freq_in_sample.keys())
    if len(unique_crimes_in_sample) < top_bottom_k * 2:
        old_k = top_bottom_k
        top_bottom_k = max(1, len(unique_crimes_in_sample) // 2)
        print(f"⚠️ Only {len(unique_crimes_in_sample)} unique crimes in sampled data. Adjusting top_bottom_k to {top_bottom_k}.")

    # 按采样内频率排序
    sorted_by_freq = sorted(freq_in_sample.items(), key=lambda x: x[1], reverse=True)
    top_crimes = set([crime for crime, _ in sorted_by_freq[:top_bottom_k]])
    bottom_crimes = set([crime for crime, _ in sorted_by_freq[-top_bottom_k:]])

    print(f"\n📊 Frequency stats IN SAMPLED DATA ({eval_size} cases):")
    print(f"  - Unique crimes: {len(unique_crimes_in_sample)}")
    print(f"  - Top-{top_bottom_k} examples: {list(top_crimes)[:3]}")
    print(f"  - Bottom-{top_bottom_k} examples: {list(bottom_crimes)[:3]}")
    print(f"  - Min freq in bottom group: {sorted_by_freq[-top_bottom_k][1]}")

    def init_group():
        return {
            "hits": {k: 0 for k in top_k_list},
            "total": 0,
            "y_true": [],
            "y_pred": []
        }

    overall = init_group()
    top_group = init_group()
    bottom_group = init_group()

    print("🔄 Evaluating on sampled data...")
    for i in tqdm(range(eval_size), desc="Processing"):
        true_crime = sampled_accusations[i]
        emb = sampled_embeddings[i].reshape(1, -1)
        sims = np.dot(emb, centroids.T).flatten()
        pred_indices = np.argsort(-sims)
        pred_crimes = [crimes[i] for i in pred_indices]

        # Update overall
        overall["total"] += 1
        overall["y_true"].append(true_crime)
        overall["y_pred"].append(pred_crimes[0])
        for k in top_k_list:
            if true_crime in pred_crimes[:k]:
                overall["hits"][k] += 1

        # Update groups based on sampled frequency
        if true_crime in top_crimes:
            top_group["total"] += 1
            top_group["y_true"].append(true_crime)
            top_group["y_pred"].append(pred_crimes[0])
            for k in top_k_list:
                if true_crime in pred_crimes[:k]:
                    top_group["hits"][k] += 1

        if true_crime in bottom_crimes:
            bottom_group["total"] += 1
            bottom_group["y_true"].append(true_crime)
            bottom_group["y_pred"].append(pred_crimes[0])
            for k in top_k_list:
                if true_crime in pred_crimes[:k]:
                    bottom_group["hits"][k] += 1

    def report(group, name, min_n=min_samples_for_macro_f1):
        total = group["total"]
        print(f"\n📊 {name} Results:")
        if total == 0:
            print("  ⚠️ No samples.")
            return {}, None

        recalls = {}
        for k in top_k_list:
            r = group["hits"][k] / total
            recalls[k] = r
            print(f"  Recall@{k:2d}: {r:.4f} ({group['hits'][k]}/{total})")

        macro_f1 = None
        if total >= min_n:
            y_true_idx = [crime_to_idx[c] for c in group["y_true"]]
            y_pred_idx = [crime_to_idx[c] for c in group["y_pred"]]
            labels = sorted(set(y_true_idx))
            macro_f1 = f1_score(y_true_idx, y_pred_idx, labels=labels, average='macro')
            print(f"  📈 Macro-F1: {macro_f1:.4f} (samples: {total}, unique crimes: {len(labels)})")
        else:
            print(f"  ⚠️ Skipped Macro-F1 (samples={total} < {min_n})")

        return recalls, macro_f1

    # Report
    _, _ = report(overall, "Overall (Single-Accusation)")
    _, _ = report(top_group, f"Top-{top_bottom_k} Crimes (in sample)")
    _, _ = report(bottom_group, f"Bottom-{top_bottom_k} Crimes (in sample)")

    # Save result
    result = {
        "config": {
            "sample_size_requested": sample_size,
            "sample_size_actual": eval_size,
            "top_bottom_k": top_bottom_k,
            "grouping_based_on": "sampled_data_frequency",
            "only_single_accusation": True,
            "embeddings_normalized": True,
            "centroids_normalized": True
        },
        "groups": {
            "overall": {
                "samples": overall["total"],
                "recall": {f"R@{k}": round(overall["hits"][k]/overall["total"], 4) for k in top_k_list}
            },
            f"top_{top_bottom_k}": {
                "samples": top_group["total"],
                "recall": {f"R@{k}": round(top_group["hits"][k]/top_group["total"], 4) for k in top_k_list},
                "crime_names": sorted(top_crimes)
            },
            f"bottom_{top_bottom_k}": {
                "samples": bottom_group["total"],
                "recall": {f"R@{k}": round(bottom_group["hits"][k]/bottom_group["total"], 4) for k in top_k_list},
                "crime_names": sorted(bottom_crimes)
            }
        }
    }

    save_path = os.path.splitext(centroid_path)[0] + "_eval_single_accusation_sample_freq.json"
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Results saved to: {save_path}")

    return result


if __name__ == "__main__":
    EMBEDDING_PATH = r"F:\LegalAgent\backend\data\fact_embeddings\fact_embeddings_bge-m3.npy"
    DATA_PATH = r"F:\LegalAgent\dataset\final_all_data\first_stage\train.json"
    CENTROID_PATH = r"F:\LegalAgent\backend\data\fact_embeddings\crime_centroids.pkl"

    evaluate_crime_centroids_single_accusation_only(
        embedding_path=EMBEDDING_PATH,
        data_path=DATA_PATH,
        centroid_path=CENTROID_PATH,
        sample_size=50000,
        top_k_list=(1, 3, 5, 10),
        seed=42,
        top_bottom_k=30,
        min_samples_for_macro_f1=50
    )