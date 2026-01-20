import os
import json
import random
from tqdm import tqdm

from prepare_datasets_v2 import GoldDatasetBuilder
from config import EvalConfig, CleanConfig

def build_sample_from_train(
    train_path=r"F:\LegalAgent\dataset\final_all_data\first_stage\train.json",
    output_path=r"F:\LegalAgent\backend\evaluate\sample_100x10.jsonl",
    num_queries=100,
    positives_per_query=10,
    fact_min=300,
    fact_max=700,
    candidate_sample_limit=2500,
    run_clean=False
):
    builder = GoldDatasetBuilder(train_path)
    builder.load_and_index()

    n_data = len(builder.all_data)
    if builder.fact_embeddings is None:
        print("Warning: embeddings not loaded; embedding module will return 0 scores.")
    else:
        if builder.fact_embeddings.shape[0] < n_data:
            print(f"Warning: embeddings length ({builder.fact_embeddings.shape[0]}) < data lines ({n_data}). 请确认向量与数据顺序对齐。")

    all_indices = list(range(n_data))
    random.shuffle(all_indices)

    selected_count = 0
    seen_acc_sets = set()
    out_dataset = []

    for q_idx in tqdm(all_indices, desc="Selecting queries"):
        if selected_count >= num_queries:
            break

        q_item = builder.all_data[q_idx]
        fact_len = len(q_item.get('fact', ''))
        if not (fact_min <= fact_len <= fact_max):
            continue

        q_accs = tuple(sorted(q_item.get('meta', {}).get('accusation', [])))
        if not q_accs:
            continue
        if q_accs in seen_acc_sets:
            continue

        # 构建候选索引集合（基于罪名倒排）
        candidate_indices = set()
        for acc in q_accs:
            if acc in builder.accusation_index:
                cands = builder.accusation_index[acc]
                if len(cands) > candidate_sample_limit:
                    candidate_indices.update(random.sample(cands, candidate_sample_limit))
                else:
                    candidate_indices.update(cands)
        if q_idx in candidate_indices:
            candidate_indices.remove(q_idx)

        scored = []
        for c_idx in candidate_indices:
            c_item = builder.all_data[c_idx]
            clen = len(c_item.get('fact', ''))
            if not (fact_min <= clen <= fact_max):
                continue
            score_info = builder._calculate_comprehensive_score(q_idx, q_item, c_idx, c_item)
            scored.append((score_info['total_score'], c_idx, score_info))

        if len(scored) < positives_per_query:
            # 不满足正样本数，跳过该 query（可改为放宽筛选）
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        topk = scored[:positives_per_query]
        positives = [builder.all_data[c_idx] for _, c_idx, _ in topk]
        score_details = [info for _, _, info in topk]

        out_dataset.append({
            "query": q_item,
            "query_idx": q_idx,
            "positives": positives,
            "positives_count": len(positives),
            "score_details": score_details
        })

        seen_acc_sets.add(q_accs)
        selected_count += 1

    # 保存结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for rec in out_dataset:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f"完成：选出 {len(out_dataset)} 个 query，保存到 {output_path}")

    if run_clean:
        # 临时修改 CleanConfig.INPUT_FILE 并调用清洗
        old_input = getattr(CleanConfig, "INPUT_FILE", None)
        old_out_clean = getattr(CleanConfig, "OUTPUT_CLEAN_FILE", None)
        old_out_dirty = getattr(CleanConfig, "OUTPUT_DIRTY_FILE", None)

        CleanConfig.INPUT_FILE = output_path
        CleanConfig.OUTPUT_CLEAN_FILE = output_path.replace(".jsonl", ".cleaned.jsonl")
        CleanConfig.OUTPUT_DIRTY_FILE = output_path.replace(".jsonl", ".dirty.jsonl")

        try:
            import importlib
            import clean_prepared_datasets
            importlib.reload(clean_prepared_datasets)
            clean_prepared_datasets.clean_data()
        finally:
            # 恢复配置
            if old_input is not None:
                CleanConfig.INPUT_FILE = old_input
            if old_out_clean is not None:
                CleanConfig.OUTPUT_CLEAN_FILE = old_out_clean
            if old_out_dirty is not None:
                CleanConfig.OUTPUT_DIRTY_FILE = old_out_dirty

if __name__ == "__main__":
    build_sample_from_train(run_clean=False)