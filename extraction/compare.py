# compare_rule_vs_llm.py
import os
import json
import random
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"  
)



PROMPT_TEMPLATE = """你是一个法律信息抽取专家，请严格按以下要求处理：

任务：从案件事实中提取结构化信息。
规则：
1. 输出必须是合法 JSON，结构与示例完全一致。
2. 每个字段：
   - 若事实中明确提及 → 填写**最相关的原文短语**（尽量简短，≤7字）
   - 若未提及 → 填 null
3. 禁止推理、总结、改写，只提取字面内容。

输出结构：
{
  "act": {
    "has_violence": "...",
    "violence_level": "...",
    "has_threat": "...",
    "is_secret": "...",
    "is_deceptive": "...",
    "has_conspiracy": "...",
    "used_tool": "..."
  },
  "object": {
    "is_person": "...",
    "is_property": "...",
    "is_public_order": "...",
    "is_state_interest": "...",
    "property_type": "..."
  },
  "result": {
    "injury": "...",
    "injury_level": "...",
    "death": "...",
    "property_transferred": "...",
    "amount_mentioned": "...",
    "has_restitution": "...",
    "has_confession": "...",
    "has_forgiveness": "..."
  },
  "participation": {
    "has_multiple_offenders": "...",
    "has_organization": "...",
    "role_description": "..."
  },
  "context": {
    "is_indoor": "...",
    "is_public_place": "...",
    "is_night": "...",
    "is_online": "..."
  }
}

"""

def extract_with_llm(fact_text: str) -> dict:
    # ✅ 提前定义 expected —— 这是关键！
    expected = {
        "act": ["has_violence", "violence_level", "has_threat", "is_secret", "is_deceptive", "has_conspiracy", "used_tool"],
        "object": ["is_person", "is_property", "is_public_order", "is_state_interest", "property_type"],
        "result": ["injury", "injury_level", "death", "property_transferred", "amount_mentioned", "has_restitution", "has_confession", "has_forgiveness"],
        "participation": ["has_multiple_offenders", "has_organization", "role_description"],
        "context": ["is_indoor", "is_public_place", "is_night", "is_online"]
    }

    try:
        completion = client.chat.completions.create(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {"role": "system", "content": PROMPT_TEMPLATE},
                {"role": "user", "content": "现在请处理以下事实：" + fact_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1000
        )
        content = completion.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM")
        
        result = json.loads(content)

        # 补全缺失字段为 null
        for group, keys in expected.items():
            if group not in result:
                result[group] = {}
            for k in keys:
                result[group].setdefault(k, None)
        return result

    except Exception as e:
        # 出错时直接返回全 null，不中断流程
        print(f"⚠️ LLM 抽取失败，跳过此条: {str(e)[:150]}")
        return {
            "act": {k: None for k in expected["act"]},
            "object": {k: None for k in expected["object"]},
            "result": {k: None for k in expected["result"]},
            "participation": {k: None for k in expected["participation"]},
            "context": {k: None for k in expected["context"]}
        }

# ======================
# 判断是否有有效信息（用于对比）
# ======================
def has_meaningful_value(val):
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return bool(val.strip())
    return bool(val)

def find_diff_fields(rule_res: dict, llm_res: dict) -> list:
    diff = []
    groups = ["act", "object", "result", "participation", "context"]
    for group in groups:
        rule_group = rule_res.get(group, {})
        llm_group = llm_res[group]
        for key in llm_group:
            v1 = rule_group.get(key)
            v2 = llm_group[key]
            has1 = has_meaningful_value(v1)
            has2 = has_meaningful_value(v2)
            if has1 != has2:
                diff.append(f"{group}.{key}")
    return diff

# ======================
# 主流程
# ======================
def main():
    INPUT_JSONL = "F:\\LegalAgent\\output\\universal_facts.jsonl"      
    OUTPUT_JSONL = "F:\\LegalAgent\\output\\comparison_200.jsonl"
    SAMPLE_SIZE = 200

    # 1. 读取所有行
    print("📂 正在加载 JSONL 数据...")
    records = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    total = len(records)
    print(f"✅ 共加载 {total} 条记录")

    # 2. 随机抽样
    sampled = random.sample(records, min(SAMPLE_SIZE, total))
    print(f"🎲 随机抽取 {len(sampled)} 条进行对比...")

    # 3. 处理每条样本
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out:
        for i, record in enumerate(sampled, 1):
            fact = record["fact"]
            rule_universal = record["universal_fact"]
            
            print(f"[{i}/{len(sampled)}] 抽取中...")
            llm_universal = extract_with_llm(fact)
            time.sleep(0.3)  # 防 API 限流

            # 对比
            diff_fields = find_diff_fields(rule_universal, llm_universal)
            has_diff = len(diff_fields) > 0

            # 构造输出
            output_record = {
                "original_record": record,          # 完整原始记录（含 meta, fact, universal_fact）
                "llm_extracted": llm_universal,     # 大模型抽取结果（字符串/null）
                "has_diff": has_diff,
                "diff_fields": diff_fields
            }
            out.write(json.dumps(output_record, ensure_ascii=False) + "\n")
    
    print(f"✅ 对比完成！结果已保存至 {OUTPUT_JSONL}")
    
    # 统计差异比例
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]
    diff_count = sum(1 for r in results if r["has_diff"])
    print(f"📊 存在差异的样本: {diff_count} / {len(results)} ({100 * diff_count / len(results):.1f}%)")

if __name__ == "__main__":
    main()