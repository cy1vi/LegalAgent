import json
import os
from collections import defaultdict
from tqdm import tqdm
from config import ExtractorConfig

def flatten_json(y):
    """将嵌套字典扁平化: {'act': {'has_violance': true}} -> {'act.has_violance': true}"""
    out = {}
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '.')
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def build_index():
    INPUT_DATA = ExtractorConfig.SCHEMA_PATH
    OUTPUT_DIR = ExtractorConfig.OUTPUT_DIR

    if not os.path.exists(INPUT_DATA):
        print(f"❌ 错误: 找不到输入文件 {INPUT_DATA}")
        return

    print(f"🔍 正在扫描数据: {INPUT_DATA}")

    schema_values = defaultdict(set)
    total_lines = 0

    with open(INPUT_DATA, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Scanning Schema"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                u_fact = item.get("universal_fact", {})
                flat_fact = flatten_json(u_fact)
                for k, v in flat_fact.items():
                    if v is not None:
                        schema_values[k].add(str(v))
                total_lines += 1
            except Exception:
                pass

    print(f"✅ 扫描完成，共 {total_lines} 条数据。")

    one_hot_maps = {}
    schema_fields = sorted(list(schema_values.keys()))

    print("⚙️ 正在构建映射表...")
    for field in schema_fields:
        values = sorted(list(schema_values[field]))
        one_hot_maps[field] = {val: idx for idx, val in enumerate(values)}

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fields_path = os.path.join(OUTPUT_DIR, "schema_fields.json")
    maps_path = os.path.join(OUTPUT_DIR, "one_hot_maps.json")

    with open(fields_path, 'w', encoding='utf-8') as f:
        json.dump(schema_fields, f, ensure_ascii=False, indent=2)
    with open(maps_path, 'w', encoding='utf-8') as f:
        json.dump(one_hot_maps, f, ensure_ascii=False, indent=2)

    print(f"💾 已保存 schema_fields.json 到 {fields_path}")
    print(f"💾 已保存 one_hot_maps.json 到 {maps_path}")

    print("\n⚠️ 注意: 如果你的 sparse_matrix.npz 是旧的，你可能需要重新生成它以匹配新的 Schema 定义。")
    print("如果 main.py 启动后检索报错 'dimension mismatch'，请需要完整的 build_index 脚本。")

if __name__ == "__main__":
    build_index()