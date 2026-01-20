import json
import jieba
from collections import defaultdict, Counter
from tqdm import tqdm
import time

# ----------------------------
# 配置
# ----------------------------
KEYWORDS_JSON = "F:\\LegalAgent\\crime_keywords.json"   # 你的罪名关键词文件
DATASET_JSONL = "F:\\LegalAgent\\dataset\\final_all_data\\first_stage\\test.json"            # 你的21万条JSONL数据
TOP_K = 10

# ----------------------------
# 加载关键词并构建倒排索引
# ----------------------------
def load_keywords(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_inverted_index(keywords_dict):
    inverted = defaultdict(set)
    for crime, words in keywords_dict.items():
        for w in words:
            w = w.strip()
            if w:
                inverted[w].add(crime)
    return dict(inverted)

# ----------------------------
# 预测 Top-K 罪名
# ----------------------------
def predict_topk(fact, inverted_index, k=3):
    words = jieba.lcut(fact)
    score = Counter()

    for w in words:
        w = w.strip()
        if w in inverted_index:
            for crime in inverted_index[w]:
                score[crime] += 1

    # 返回 Top-k 罪名列表（按得分降序）
    return [crime for crime, _ in score.most_common(k)]

# ----------------------------
# 主流程
# ----------------------------
def main():
    print("📥 加载关键词...")
    keywords = load_keywords(KEYWORDS_JSON)
    clean_keywords = {
        k: [str(x).strip() for x in v if str(x).strip()]
        for k, v in keywords.items()
    }
    inverted_index = build_inverted_index(clean_keywords)

    print("📊 开始处理数据集...")
    total = 0
    top1_correct = 0
    top3_correct = 0

    start_time = time.time()

    with open(DATASET_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="处理进度"):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            fact = item.get("fact", "").strip()
            true_accusations = item.get("meta", {}).get("accusation", [])
            if not true_accusations or not fact:
                continue

            true_label = true_accusations[0]  # 取主罪名
            preds = predict_topk(fact, inverted_index, k=TOP_K)

            total += 1
            if preds and preds[0] == true_label:
                top1_correct += 1
            if true_label in preds:
                top3_correct += 1

        except Exception as e:
            continue  # 跳过格式错误行

    # 计算准确率
    top1_acc = top1_correct / total if total > 0 else 0
    top3_acc = top3_correct / total if total > 0 else 0

    print(f"\n✅ 总样本数: {total}")
    print(f"🎯 Top-1 准确率: {top1_acc:.4f} ({top1_correct}/{total})")
    print(f"🎯 Top-3 准确率: {top3_acc:.4f} ({top3_correct}/{total})")

    elapsed = time.time() - start_time
    print(f"⏱️  总耗时: {elapsed:.2f} 秒 | 速度: {total/elapsed:.1f} 条/秒")

if __name__ == "__main__":
    main()