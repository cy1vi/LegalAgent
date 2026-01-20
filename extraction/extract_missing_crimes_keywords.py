# extract_missing_crimes_keywords.py
import json
import re
from collections import defaultdict
import os
import argparse

try:
    import hanlp
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
    USE_HANLP = True
except Exception as e:
    print(f"❌ 无法加载 HanLP: {e}")
    exit(1)


def load_stopwords():
    stopwords = set()
    if os.path.exists("stopwords.txt"):
        with open("stopwords.txt", "r", encoding="utf-8") as f:
            stopwords.update(line.strip() for line in f if line.strip())
    else:
        basic = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "个",
                 "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好"}
        judicial = {"经查", "本院认为", "综上", "依法", "判处", "审理", "查明", "上述", "依照",
                    "规定", "应当", "依法予以", "提起公诉", "向本院", "请求依法", "被告人",
                    "被害人", "供述", "证言", "鉴定意见", "证据", "事实清楚", "证据确实",
                    "充分", "构成", "检察院", "公安机关", "投案", "自首", "谅解", "赔偿"}
        stopwords = basic | judicial
    return stopwords


def tokenize_with_hanlp(text, stopwords):
    if not isinstance(text, str) or not text.strip():
        return []
    # 保留中文、英文、数字、点、斜杠（如 IP、编号）
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9./]", " ", text)

    try:
        doc = HanLP(text, tasks=['tok/fine', 'ner/msra'])
        words = doc['tok/fine']
        ner_entities = doc['ner/msra']
        entity_set = set(ent[0].lower() for ent in ner_entities)

        result = []
        for w in words:
            w = w.strip().lower()
            if len(w) < 2:
                continue
            if w in stopwords:
                continue
            if w in entity_set:
                continue
            if re.match(r'^[a-z]*\d+[a-z]*$', w) or w.isdigit():
                continue
            # 过滤泛化法律术语
            if w in {"财物", "行为", "工具", "物品", "人员", "事情", "方式", "手段", "进行", "实施",
                     "过程", "情况", "结果", "目的", "地点", "时间", "内容", "部分", "方面"}:
                continue
            result.append(w)
        return result
    except Exception as e:
        # 回退到 jieba
        try:
            import jieba
            words = jieba.lcut(text)
            return [w.strip().lower() for w in words
                    if len(w.strip()) >= 2 and w.strip().lower() not in stopwords]
        except:
            return []


# ==============================
# 主函数：仅提取指定罪名
# ==============================
def main(jsonl_path, target_crimes, top_k=30, min_samples=1):
    print("🔍 加载停用词...")
    stopwords = load_stopwords()

    target_set = set(target_crimes)
    print(f"🎯 仅提取以下 {len(target_set)} 个罪名的关键词:")
    for c in sorted(target_set):
        print(f"  - {c}")

    # 🔥 关键修改：用 df（文档频次）代替 idx 集合，避免 OOM
    crime_word_df = defaultdict(lambda: defaultdict(int))  # crime -> word -> doc_freq
    crime_doc_count = defaultdict(int)  # crime -> total_docs
    total_docs = 0  # 总“罪名-文档”对数（用于 global normalization）

    print(f"📥 流式读取数据: {jsonl_path}")
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                fact = data.get("fact", "")
                accusation = data.get("meta", {}).get("accusation", [])
                if not fact or not isinstance(accusation, list) or len(accusation) == 0:
                    continue

                # 检查是否有目标罪名（支持多罪名）
                matched_crimes = [c for c in accusation if c in target_set]
                if not matched_crimes:
                    continue

                words = tokenize_with_hanlp(fact, stopwords)
                if not words:
                    continue

                # 样本内去重（一个词在一个样本中只计一次）
                word_set = set(words)

                # 为每个匹配罪名更新统计
                for crime in matched_crimes:
                    for w in word_set:
                        crime_word_df[crime][w] += 1
                    crime_doc_count[crime] += 1
                    total_docs += 1

                if total_docs % 5000 == 0:
                    print(f"  ✅ 已处理 {total_docs} 条目标案件...")

            except Exception as e:
                # 如需调试，可取消注释：
                # print(f"⚠️ 跳过第 {idx} 行: {e}")
                continue

    print(f"✅ 共处理 {total_docs} 条目标案件，涉及 {len(crime_doc_count)} 个罪名")

    # 生成关键词
    crime_keywords = {}
    global_total = total_docs  # 所有目标罪名的总文档数（近似）

    for crime in target_set:
        if crime_doc_count[crime] < min_samples:
            print(f"⚠️  罪名 '{crime}' 样本不足 ({crime_doc_count[crime]} < {min_samples})，使用罪名本身作为关键词")
            crime_keywords[crime] = [crime]
            continue

        doc_count = crime_doc_count[crime]
        scores = {}
        for word, df_crime in crime_word_df[crime].items():
            if df_crime < 1:
                continue
            # 计算该词在所有目标罪名中的总出现文档数
            df_global = sum(crime_word_df[c].get(word, 0) for c in target_set)
            # 防止除零
            score = (df_crime / doc_count) / (df_global / global_total + 1e-9)
            scores[word] = score

        # 取 top-k
        top_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        keywords = [w for w, _ in top_words] if top_words else [crime]
        crime_keywords[crime] = keywords[:top_k]

    # 💾 保存结果
    output_file = "missing_crime_keywords.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(crime_keywords, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 完成！关键词已保存至: {output_file}")
    print("\n预览:")
    for crime, kws in crime_keywords.items():
        print(f"{crime}: {kws[:5]}{'...' if len(kws) > 5 else ''}")


# ==============================
# CLI 入口
# ==============================
if __name__ == "__main__":
    MISSING_CRIMES = [
        "传播淫秽物品",
        "包庇毒品犯罪分子",
        "协助组织卖淫",
        "巨额财产来源不明",
        "引诱、容留、介绍卖淫",
        "强迫卖淫",
        "徇私舞弊不征、少征税款",
        "盗窃、抢夺枪支、弹药、爆炸物、危险物质",
        "组织卖淫",
        "经济犯",
        "非法买卖、运输、携带、持有毒品原植物种子、幼苗",
        "非法收购、运输、出售珍贵、濒危野生动物、珍贵、濒危野生动物制品"
    ]

    parser = argparse.ArgumentParser(description="高效提取指定罪名的关键词（仅处理目标类）")
    parser.add_argument("--jsonl_file", default=r"F:\LegalAgent\dataset\final_all_data\first_stage\train.json", help="案件JSONL路径")
    parser.add_argument("--top_k", type=int, default=25, help="每罪名关键词数量")
    parser.add_argument("--min_samples", type=int, default=1, help="最小样本数（设为1确保全覆盖）")

    args = parser.parse_args()

    main(
        jsonl_path=args.jsonl_file,
        target_crimes=MISSING_CRIMES,
        top_k=args.top_k,
        min_samples=args.min_samples
    )