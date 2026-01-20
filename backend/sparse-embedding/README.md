 # Sparse Embedding（稀疏检索模块）完整说明文档
 本模块是 LegalAgent 法律智能系统中用于 **稀疏向量构建、索引、检索、关键词与结构化事实抽取** 的完整组件。该模块支撑法律案件相似度检索、法律问答召回、RAG 等核心功能。

 ---

## 一、模块功能概述
本模块实现从 **原始案情文本 → 稀疏特征 → 稀疏向量矩阵 → 在线检索服务** 的全流程，包括：

- 关键词抽取（CrimeKeywordsExtractor）
- 通用事实 Schema 抽取（UniversalFactExtractor）
- one-hot 特征构建（build_sparse_index.py）
- 高维稀疏矩阵生成（CSR / NPZ）
- 稀疏检索（SparseRetriever）
- 语料随机访问（CorpusManager）
- Schema / Keyword 按行索引（SchemaKeywordsManager）
- FastAPI 在线检索服务
- 本地测试工具（quick_test.py）

该系统可独立运行，也可作为更大 RAG 系统中的稀疏召回模块。

---

## 二、目录结构与文件作用
```
sparse-embedding/
├── config.py                    # 全局路径、数据配置
├── main.py                      # FastAPI 检索服务入口（最核心）
├── utils.py                     # 稀疏检索核心 SparseRetriever
├── logger.py                    # 日志系统
├── quick_test.py                # 本地调试工具
├── build_sparse_index.py        # one-hot 构建 & 稀疏矩阵生成
├── universal_fact_extractor.py  # 通用事实 Schema 抽取
├── crime_keywords_extractor.py  # 罪名关键词抽取器
├── requirements.txt             # 依赖列表
├── data/                        # 稀疏矩阵、schema 文件、关键词文件
└── logs/                        # 运行日志
```

### 1. config.py
集中管理所有路径、文件、稀疏矩阵位置。
包含：
- 数据集位置
- schema 文件位置
- keywords 文件位置
- one-hot 映射位置
- 稀疏矩阵 npz 输出位置
- 服务端口

### 2. main.py（检索服务核心）
系统启动时初始化：
- CrimeKeywordsExtractor
- UniversalFactExtractor
- SparseRetriever（加载矩阵）
- CorpusManager（按行读取原始 JSONL 语料）
- SchemaKeywordsManager（按行读取 schema/jsonl）

并提供：
- `/search` 单条检索
- `/batch_search` 批量检索

### 3. utils.py（SparseRetriever）
实现稀疏检索核心：
- 加载矩阵并归一化
- 根据输入 fact 生成 query 向量（Schema + Keyword）
- 稀疏矩阵点积计算余弦相似度
- 返回 top-k

### 4. build_sparse_index.py
构建稀疏矩阵的离线工具：
- 构建 schema_fields（所有字段名）
- 构建 one_hot_maps（离散值 → index）
- 扫描 schema.jsonl + keywords.jsonl 并构建矩阵
- 输出 sparse_matrix.npz

### 5. universal_fact_extractor.py
根据 YAML 规则抽取 universal_fact：
- 主体
- 行为
- 方式方法
- 后果
- 主观故意

### 6. crime_keywords_extractor.py
从案情中提取罪名相关关键词并计数，用作稀疏特征的一部分。

### 7. quick_test.py
本地调试工具，可打印
- Query schema
- Query keywords
- Doc schema
- Doc keywords
- 案件摘要

---

## 三、稀疏矩阵构建流程（离线）
### Step 1：抽取 Schema
```
universal_fact_extractor → universal_fact.jsonl
```
### Step 2：抽取 Keywords
```
crime_keywords_extractor → keyword_counts.jsonl
```
### Step 3：生成 one-hot 映射
```
python build_sparse_index.py
```
输出：
- schema_fields.json
- one_hot_maps.json

### Step 4：构建稀疏矩阵
```
python build_sparse_index.py
```
输出：
- sparse_matrix.npz

---

## 四、启动在线检索服务
```
python main.py
```
默认地址： http://localhost:4240

### 1. 单条检索
```
POST /search
{
	"fact": "被告人持刀抢劫……",
	"top_k": 5
}
```

### 2. 批量检索
```
POST /batch_search
{
	"facts": ["案情 1", "案情 2"],
	"top_k": 5
}
```

---

## 五、SearchResult 字段说明
| 字段名 | 描述 |
|--------|------|
| score | 相似度得分 |
| fact | 案情内容 |
| accusation | 罪名 |
| relevant_articles | 法条 |
| imprisonment | 刑期结构 |
| punish_of_money | 罚金 |
| matched_keywords | 查询关键词命中 |
| query_schema | 查询结构化事实 |
| document_schema | 文档结构化事实 |
| document_keywords | 文档关键词 |
| metadata | 原始 meta、索引等 |

---

## 六、调试方法
```
python quick_test.py
```
可打印：
- Query Schema / Keywords
- Document Schema / Keywords
- 案情摘要

---

## 七、未来扩展方向
- BM25+ scoring
- 稀疏 + 稠密混合检索
- 稀疏矩阵压缩（bit-packed）
- Schema 字段动态权重
- LLM 驱动 Schema 抽取

---

