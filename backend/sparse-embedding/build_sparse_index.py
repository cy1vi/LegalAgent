import json
import os
from tqdm import tqdm
from scipy import sparse
from collections import defaultdict
from config import ExtractorConfig


def flatten_json(y):
    out = {}
    def flatten(x, name=''):
        if isinstance(x, dict):
            for a in x:
                flatten(x[a], name + a + '.')
        else:
            out[name[:-1]] = x
    flatten(y)
    return out

def load_keywords_map():
    KEYWORDS_PATH = ExtractorConfig.KEYWORDS_FILE
    print(f"📖 加载罪名关键词定义: {KEYWORDS_PATH}")
    with open(KEYWORDS_PATH, 'r', encoding='utf-8') as f:
        kw_data = json.load(f)
    all_keywords = sorted(list(set([kw for keyword_list in kw_data.values() for kw in keyword_list])))
    kw_map = {kw: i for i, kw in enumerate(all_keywords)}
    return kw_map, all_keywords

def build_sparse_matrix():
    SCHEMA_DATA_PATH = ExtractorConfig.SCHEMA_PATH
    KEYWORD_DATA_PATH = ExtractorConfig.KEYWORDS_PATH
    OUTPUT_DB_PATH = ExtractorConfig.DB_PATH
    SCHEMA_FIELDS_PATH = ExtractorConfig.fields_path
    ONE_HOT_MAPS_PATH = ExtractorConfig.maps_path

    if not os.path.exists(SCHEMA_DATA_PATH):
        print(f"❌ 找不到 Schema 文件: {SCHEMA_DATA_PATH}")
        return
    if not os.path.exists(KEYWORD_DATA_PATH):
        print(f"❌ 找不到 Keyword 文件: {KEYWORD_DATA_PATH}")
        return

    print("1. 加载 Schema 配置...")
    if not os.path.exists(SCHEMA_FIELDS_PATH) or not os.path.exists(ONE_HOT_MAPS_PATH):
        print("❌ 缺少 schema_fields.json 或 one_hot_maps.json")
        return

    with open(SCHEMA_FIELDS_PATH, 'r', encoding='utf-8') as f:
        schema_fields = json.load(f)
    with open(ONE_HOT_MAPS_PATH, 'r', encoding='utf-8') as f:
        one_hot_maps = json.load(f)

    kw_map, all_keywords = load_keywords_map()

    schema_offsets = {}
    current_offset = 0
    for field in schema_fields:
        schema_offsets[field] = current_offset
        current_offset += len(one_hot_maps[field])
    schema_dim = current_offset
    keyword_dim = len(all_keywords)
    total_dim = schema_dim + keyword_dim

    print(f"📊 矩阵维度统计:")
    print(f" - Schema 维度: {schema_dim}")
    print(f" - Keyword 维度: {keyword_dim}")
    print(f" - 总维度: {total_dim}")

    rows = []
    cols = []
    data = []
    row_idx = 0

    print(f"2. 正在并行扫描两个文件...")
    print(f" Schema: {SCHEMA_DATA_PATH}")
    print(f" Keyword: {KEYWORD_DATA_PATH}")

    with open(SCHEMA_DATA_PATH, 'r', encoding='utf-8') as f_schema, \
         open(KEYWORD_DATA_PATH, 'r', encoding='utf-8') as f_keyword:
        for line_s, line_k in tqdm(zip(f_schema, f_keyword), desc="Building Matrix"):
            line_s = line_s.strip()
            line_k = line_k.strip()
            if not line_s or not line_k:
                continue
            try:
                item_s = json.loads(line_s)
                item_k = json.loads(line_k)

                u_fact = item_s.get("universal_fact")
                if u_fact:
                    flat_fact = flatten_json(u_fact)
                    for field, value in flat_fact.items():
                        if field in one_hot_maps and str(value) in one_hot_maps[field]:
                            val_idx = one_hot_maps[field][str(value)]
                            col_idx = schema_offsets[field] + val_idx
                            rows.append(row_idx)
                            cols.append(col_idx)
                            data.append(1.0)

                kw_counts = {}
                if "sparse_extraction" in item_k and "keyword_counts" in item_k["sparse_extraction"]:
                    kw_counts = item_k["sparse_extraction"]["keyword_counts"]
                elif "keyword_counts" in item_k:
                    kw_counts = item_k["keyword_counts"]

                if kw_counts:
                    for kw, count in kw_counts.items():
                        if kw in kw_map:
                            col_idx = schema_dim + kw_map[kw]
                            rows.append(row_idx)
                            cols.append(col_idx)
                            data.append(float(count))
                row_idx += 1
            except Exception as e:
                pass

    print(f"✅ 扫描完成。共 {row_idx} 条数据。")

    print("3. 转换并保存矩阵...")
    matrix = sparse.csr_matrix((data, (rows, cols)), shape=(row_idx, total_dim))
    sparse.save_npz(OUTPUT_DB_PATH, matrix)
    print(f"💾 矩阵已保存至: {OUTPUT_DB_PATH}")
    print(f" 文件大小: {os.path.getsize(OUTPUT_DB_PATH) / 1024 / 1024:.2f} MB")

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

if __name__ == "__main__":
    build_sparse_matrix()
    build_index()