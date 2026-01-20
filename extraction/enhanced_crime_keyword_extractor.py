# final_crime_keyword_extractor_hanlp.py
import json
import re
from collections import defaultdict
import os
import argparse
import tempfile

# ==============================
# 尝试导入 HanLP（必须安装 hanlp）
# ==============================
try:
    import hanlp
    # 加载多任务模型（包含分词、词性、NER）
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH)
    USE_HANLP = True
    print("✅ HanLP 多任务模型加载成功（含 NER）")
except Exception as e:
    print(f"❌ 无法加载 HanLP: {e}")
    print("请运行: pip install hanlp -i https://pypi.tuna.tsinghua.edu.cn/simple")
    exit(1)

# ==============================
# 1. 停用词加载
# ==============================
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

# ==============================
# 2. 构建法律词典（用于增强关键词）
# ==============================
def build_legal_dict(crimes_txt_path=None):
    legal_words = set()
    if crimes_txt_path and os.path.exists(crimes_txt_path):
        with open(crimes_txt_path, "r", encoding="utf-8") as f:
            for line in f:
                crime = line.strip()
                if crime:
                    parts = re.split(r'[、，\s]+', crime)
                    for part in parts:
                        if len(part) >= 2:
                            legal_words.add(part)
    else:
        builtin = [
            "故意伤害", "盗窃", "抢劫", "诈骗", "危险驾驶", "交通肇事",
            "容留他人吸毒", "贩卖毒品", "强奸", "非法持有枪支", "放火",
            "爆炸", "绑架", "拐卖妇女儿童", "贪污", "受贿", "职务侵占",
            "非法经营", "开设赌场", "合同诈骗", "信用卡诈骗", "洗钱",
            "假冒注册商标", "生产销售假药", "污染环境", "妨害公务",
            "聚众斗殴", "以危险方法危害公共安全", "虚开发票", "逃税",
            "寻衅滋事", "帮助毁灭伪造证据"
        ]
        for crime in builtin:
            parts = re.split(r'[、，\s]+', crime)
            for part in parts:
                if len(part) >= 2:
                    legal_words.add(part)

    terms = [
        "随意殴打", "追逐拦截", "辱骂恐吓", "起哄闹事", "任意损毁", "占用财物",
        "伪造证据", "毁灭证据", "帮助毁灭", "隐匿证据", "作假证明",
        "醉酒驾驶", "血液酒精含量", "持械", "轻伤", "重伤", "死亡", "逃逸",
        "秘密窃取", "冒充", "虚构事实", "非法占有", "计算机系统"
    ]
    legal_words.update(terms)

    dict_path = os.path.join(tempfile.gettempdir(), "legal_dict_temp.txt")
    with open(dict_path, "w", encoding="utf-8") as f:
        for word in sorted(legal_words):
            if len(word) >= 2:
                f.write(f"{word}\n")
    return dict_path

# ==============================
# 3. 使用 HanLP 进行分词 + NER 过滤
# ==============================
def tokenize_with_hanlp(text, stopwords):
    if not isinstance(text, str) or not text.strip():
        return []
    # 保留中文、字母、数字、./（如 mg/100ml）
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9./]", " ", text)
    
    try:
        doc = HanLP(text, tasks=['tok/fine', 'ner/msra'])
        words = doc['tok/fine']
        ner_entities = doc['ner/msra']
        
        # 提取所有被标记为实体的词（转小写）
        entity_set = set(ent[0].lower() for ent in ner_entities)
        
        result = []
        for w in words:
            w = w.strip().lower()
            if len(w) < 2:
                continue
            if w in stopwords:
                continue
            if w in entity_set:  # 过滤人名、地名、组织名等
                continue
            if re.match(r'^[a-z]*\d+[a-z]*$', w) or w.isdigit():
                continue
            if w in {"财物", "行为", "工具", "物品", "人员", "事情", "方式", "手段", "进行", "实施"}:
                continue
            result.append(w)
        return result
    except Exception as e:
        # 回退：简单分词（实际很少触发）
        import jieba
        words = jieba.lcut(text)
        return [w.strip().lower() for w in words if len(w.strip()) >= 2 and w.strip() not in stopwords]

# ==============================
# 4. 法律行为词白名单（用于回退和校验）
# ==============================
LEGAL_BEHAVIOR_KEYWORDS = {
    "寻衅滋事": ["殴打", "辱骂", "恐吓", "追逐", "拦截", "起哄", "闹事", "损毁", "占用", "随意"],
    "帮助毁灭、伪造证据": ["毁灭", "伪造", "证据", "隐匿", "抛弃", "掩埋", "作假", "造假"],
    "故意伤害": ["殴打", "打伤", "捅", "砍", "击打", "伤害", "重伤", "轻伤"],
    "盗窃": ["窃取", "偷", "秘密", "扒窃", "入户", "盗取"],
    "诈骗": ["骗取", "虚构", "隐瞒", "谎称", "冒充", "返利", "投资", "平台"],
    "危险驾驶": ["醉酒", "酒精", "驾驶", "机动车", "血液", "超标"],
    "贩卖毒品": ["贩卖", "毒品", "冰毒", "海洛因", "大麻", "交易"],
    "容留他人吸毒": ["容留", "吸毒", "提供场所", "吸食"],
    "交通肇事": ["肇事", "逃逸", "撞", "致人死亡", "致人重伤", "违反交规"],
    "抢劫": ["抢劫", "暴力", "胁迫", "抢走", "持械"],
}

KEYWORD_FIELD_RULES = {
    "mentions_violence": ["殴打", "打伤", "拳脚", "暴力", "砸", "踢", "捅", "持械"],
    "mentions_impersonation": ["冒充", "假冒", "伪装", "谎称"],
    "mentions_alcohol": ["醉酒", "酒精", "饮酒", "血液酒精", "酒后"],
    "mentions_vehicle": ["机动车", "汽车", "驾驶", "道路", "车辆", "交通", "行驶"],
    "mentions_drugs": ["毒品", "吸毒", "冰毒", "海洛因", "大麻", "容留", "贩毒"],
    "mentions_financial_fraud": ["诈骗", "返利", "投资", "平台", "转账", "理财", "集资"],
    "mentions_secret_theft": ["秘密", "趁人不备", "入户", "窃取", "偷", "扒窃"],
    "mentions_public_disorder": ["公共场所", "起哄", "扰乱", "聚众", "闹事", "滋事"],
    "mentions_firearms": ["枪支", "弹药", "爆炸物", "火药", "持枪"],
    "mentions_computer": ["计算机", "系统", "程序", "黑客", "侵入", "控制"]
}

# ==============================
# 5. 主流程
# ==============================
def main(jsonl_path, crimes_txt_path=None, top_k=30, min_samples=5):
    print("🔍 加载停用词...")
    stopwords = load_stopwords()

    print("📚 构建法律词典...")
    dict_path = build_legal_dict(crimes_txt_path)

    print(f"📥 流式读取数据: {jsonl_path}")

    global_df = defaultdict(int)
    crime_word_docs = defaultdict(lambda: defaultdict(set))
    crime_doc_count = defaultdict(int)
    total_docs = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                fact = data.get("fact", "")
                accusation = data.get("meta", {}).get("accusation", [])
                if fact and accusation:
                    crime = accusation[0]
                    words = tokenize_with_hanlp(fact, stopwords)
                    if not words:
                        continue

                    word_set = set(words)
                    for w in word_set:
                        crime_word_docs[crime][w].add(idx)
                        global_df[w] += 1
                    crime_doc_count[crime] += 1
                    total_docs += 1

                    if total_docs % 20000 == 0:
                        print(f"  ✅ 已处理 {total_docs} 条案件...")

            except json.JSONDecodeError:
                continue

    print(f"✅ 共处理 {total_docs} 条有效案件，涉及 {len(crime_doc_count)} 种罪名")

    valid_crimes = {crime for crime, cnt in crime_doc_count.items() if cnt >= min_samples}
    crime_keywords = {}

    print("📊 计算判别性关键词（基于文档频率）...")
    for crime in valid_crimes:
        doc_count = crime_doc_count[crime]
        scores = {}
        for word, doc_ids in crime_word_docs[crime].items():
            df_crime = len(doc_ids)
            df_global = global_df[word]
            if df_crime < 2:
                continue
            score = (df_crime / doc_count) / (df_global / total_docs + 1e-9)
            scores[word] = score

        top_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        keywords = [w for w, _ in top_words]
        crime_keywords[crime] = keywords

    # ==============================
    # 🔁 回退 + 白名单增强
    # ==============================
    print("🔄 应用白名单增强与回退策略...")
    for crime in list(valid_crimes):
        keywords = crime_keywords.get(crime, [])
        whitelist = LEGAL_BEHAVIOR_KEYWORDS.get(crime, [])
        has_whitelist = any(kw in keywords for kw in whitelist)

        if not has_whitelist or len(keywords) < 3:
            enhanced = whitelist + [crime]
            existing_good = [w for w in keywords if len(w) >= 2 and not w.isdigit()]
            all_candidates = enhanced + existing_good
            seen = set()
            dedup = []
            for w in all_candidates:
                if w not in seen:
                    dedup.append(w)
                    seen.add(w)
            crime_keywords[crime] = dedup[:top_k]

    # ==============================
    # 💾 保存结果
    # ==============================
    print("💾 保存关键词文本...")
    with open("crime_keywords.txt", "w", encoding="utf-8") as f:
        for crime in sorted(crime_keywords.keys()):
            keywords = crime_keywords[crime]
            f.write(f"\n=== {crime} ===\n")
            for word in keywords:
                f.write(f"{word}\n")

    print("🧩 生成字段映射...")
    mapping_output = {}
    for crime, keywords in crime_keywords.items():
        suggested = {}
        for field, triggers in KEYWORD_FIELD_RULES.items():
            matched = [w for w in keywords if w in triggers]
            if matched:
                suggested[field] = matched
        mapping_output[crime] = {
            "raw_keywords": keywords,
            "suggested_fields": suggested
        }

    with open("keyword_to_field_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping_output, f, ensure_ascii=False, indent=2)

    print("✅ 完成！输出文件：")
    print("   - crime_keywords.txt")
    print("   - keyword_to_field_mapping.json")

# ==============================
# 6. CLI 入口
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="最终版罪名关键词提取器（HanLP 版，含 NER 过滤）")
    parser.add_argument("--jsonl_file", default=r"F:\LegalAgent\dataset\final_all_data\first_stage\test.json", help="案件JSONL文件路径")
    parser.add_argument("--crimes_txt", default=r"F:\LegalAgent\dataset\final_all_data\meta\accu.txt", help="罪名列表txt")
    parser.add_argument("--top_k", type=int, default=30, help="每罪名关键词数")
    parser.add_argument("--min_samples", type=int, default=5, help="罪名最小样本数")

    args = parser.parse_args()
    main(
        jsonl_path=args.jsonl_file,
        crimes_txt_path=args.crimes_txt,
        top_k=args.top_k,
        min_samples=args.min_samples
    )