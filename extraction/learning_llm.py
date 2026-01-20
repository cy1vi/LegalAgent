# aggregate_llm_keywords.py
import json
from collections import defaultdict

INPUT_FILE = "F:\\LegalAgent\\output\\comparison_200.jsonl"  # 👈 替换为你的文件路径

# 初始化嵌套 defaultdict
def make_nested_dict():
    return defaultdict(lambda: defaultdict(list))

aggregated = make_nested_dict()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        llm_res = record["llm_extracted"]

        for group_name, group in llm_res.items():
            for field, value in group.items():
                # 过滤掉 null / "null" / 空字符串
                if value is None:
                    continue
                if isinstance(value, str):
                    clean_val = value.strip()
                    if clean_val.lower() == "null" or clean_val == "":
                        continue
                    aggregated[group_name][field].append(clean_val)

# 转为普通 dict（便于 JSON 序列化）
result = {
    "universal_fact": {
        group: dict(fields)
        for group, fields in aggregated.items()
    }
}

# 输出到文件
OUTPUT_FILE = "F:\\LegalAgent\\output\\llm_keyword_summary.json"
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    json.dump(result, out, ensure_ascii=False, indent=2)

print(f"✅ 汇总完成！结果已保存至 {OUTPUT_FILE}")
print("\n📊 示例预览（前3项）：")
for group, fields in list(result["universal_fact"].items())[:2]:
    print(f"\n{group}:")
    for field, vals in list(fields.items())[:2]:
        preview = vals[:3]  # 只看前3个
        print(f"  {field}: {preview}")