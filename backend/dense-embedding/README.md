# Legal Fact Dense Retrieval Service (法律案情稠密向量检索服务)

## 📖 项目简介

**Dense Retrieval Service** 是 LegalAgent 项目的核心语义检索组件，基于深度学习模型将法律案情文本转换为稠密向量，通过向量相似度搜索实现智能案例检索。

### 🎯 核心价值
- **语义理解**: 超越关键词匹配，理解案情背后的法律逻辑和事实关系
- **智能检索**: 即使查询与案例没有相同关键词，只要语义相似就能准确召回
- **高性能**: 支持百万级案例库的毫秒级检索响应
- **多模型支持**: 支持 Lawformer（法律专用模型）和 BGE-M3（通用多语言模型）

### 🔑 核心特性
- **双模型支持**: Lawformer（法律领域优化）和 BGE-M3（通用强大模型）
- **多索引算法**: FAISS（精确/近似搜索）和 HNSW（图索引）支持
- **批量处理**: GPU 并行加速的批量检索 API
- **完整元数据**: 返回案情、罪名、法条、刑期等完整信息
- **实时服务**: FastAPI 构建的 RESTful API 服务

---

## 📂 项目结构

```
dense-embedding/
├── main.py                    # FastAPI 服务入口
├── config.py                  # 配置文件（模型路径、数据路径等）
├── indexing.py               # 向量索引封装（FAISS/HNSW）
├── fact_embedding.py         # 向量生成和批量处理
├── logger.py                 # 日志管理模块
├── quick_test.py             # 快速测试脚本
├── requirements.txt          # 依赖包列表
├── README.md                 # 本文档
├── logs/                     # 日志目录
└── __pycache__/             # Python 缓存
```

### 文件说明

| 文件 | 说明 |
|------|------|
| **main.py** | FastAPI 服务主文件，包含 API 路由、模型加载、索引构建等 |
| **config.py** | 全局配置：模型路径、数据路径、索引类型、服务端口等 |
| **indexing.py** | 向量索引抽象层，支持 FAISS 和 HNSW 两种索引算法 |
| **fact_embedding.py** | 批量生成案情向量的工具，支持 Lawformer 和 BGE-M3 |
| **logger.py** | 统一的日志管理，支持彩色输出和文件记录 |
| **quick_test.py** | 客户端测试脚本，验证服务功能 |

---

## 🛠️ 技术架构

### 系统架构图
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Query  │───▶│   FastAPI API   │───▶│   Embedding     │
│   (案情文本)     │    │   (HTTP/JSON)   │    │   Model         │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌────────▼────────┐
│   Search Result │◀───│   Vector Index  │◀───│   Vector        │
│   (检索结果)     │    │   (FAISS/HNSW)  │    │   (768维)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 工作流程
1. **服务启动阶段**:
   - 加载预训练的嵌入模型（Lawformer 或 BGE-M3）
   - 加载预计算的案情向量（.npy 文件）
   - 构建向量索引（FAISS 或 HNSW）
   - 加载原始案情数据用于结果回填

2. **在线检索阶段**:
   - 接收用户查询的案情文本
   - 使用嵌入模型将文本转换为稠密向量
   - 在向量索引中搜索最相似的 Top-K 个向量
   - 根据向量 ID 获取完整的案情元数据
   - 返回格式化的检索结果

---

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 如果需要 GPU 支持
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 配置检查
编辑 `config.py` 确保以下路径正确：
```python
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))

# 模型路径（支持 Lawformer 和 BGE-M3）
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "Lawformer")
BGE_M3_PATH = r"D:\.cache\huggingface\hub\models--BAAI--bge-m3"

# 数据路径
DATA_DIR = os.path.join(PROJECT_ROOT, "backend", "data", "fact_embeddings")
RAW_DATA_PATH = r"F:\LegalAgent\dataset\final_all_data\first_stage\train.json"
```

### 3. 启动服务
```bash
# 启动 FastAPI 服务
python main.py

# 服务将在 http://0.0.0.0:4241 运行
# 访问 http://localhost:4241/docs 查看 API 文档
```

### 4. 快速测试
```bash
# 运行测试脚本
python quick_test.py

# 或使用 curl 测试
curl -X POST "http://localhost:4241/search" \
  -H "Content-Type: application/json" \
  -d '{"fact": "被告人盗窃了一辆电动车，价值3000元", "top_k": 5}'
```

---

## 🔌 API 接口

### 1. 健康检查 `/health`
```bash
GET /health
```
返回服务状态信息。

### 2. 单条检索 `/search`
```bash
POST /search
Content-Type: application/json

{
  "fact": "案情文本内容...",
  "top_k": 10
}
```

**响应示例**:
```json
[
  {
    "fact_id": "12345",
    "score": 0.92,
    "rank": 1,
    "fact": "完整的案情描述...",
    "accusation": ["盗窃罪"],
    "relevant_articles": ["刑法第264条"],
    "imprisonment": {"months": 6}
  }
]
```

### 3. 批量检索 `/batch_search`
```bash
POST /batch_search
Content-Type: application/json

{
  "facts": ["案情A...", "案情B...", "案情C..."],
  "top_k": 10
}
```

**响应**: 返回二维列表 `List[List[SearchResult]]`，每个查询对应一个结果列表。

### 4. 向量生成 `/embed`
```bash
POST /embed
Content-Type: application/json

{
  "text": "需要向量化的文本..."
}
```

**响应**: 返回文本的 768 维向量表示。

---

## ⚙️ 配置说明

### 模型配置
```python
# config.py 中的关键配置

# 嵌入模型类型："lawformer" 或 "bge-m3"
EMBEDDING_MODEL_TYPE = "bge-m3"

# 索引类型："faiss" 或 "hnsw"
INDEX_TYPE = "faiss"

# 服务配置
HOST = "0.0.0.0"
PORT = 4241

# 序列长度
MAX_SEQ_LENGTH = 512
```

### 数据文件
- **向量文件**: `backend/data/fact_embeddings/fact_embeddings.npy` (Lawformer)
- **向量文件**: `backend/data/fact_embeddings/fact_embeddings_bge-m3.npy` (BGE-M3)
- **原始数据**: `dataset/final_all_data/first_stage/train.json`

---

## 🔧 高级功能

### 1. 批量向量生成
使用 `fact_embedding.py` 批量生成案情向量：
```bash
python fact_embedding.py --model_type bge-m3 --batch_size 32
```

### 2. 索引构建
支持两种索引算法：
- **FAISS**: Facebook 的高效相似度搜索库，支持精确和近似搜索
- **HNSW**: 分层可导航小世界图，适合高维向量快速搜索

### 3. 模型切换
在 `config.py` 中修改 `EMBEDDING_MODEL_TYPE` 即可切换模型：
- `"lawformer"`: 法律领域专用模型，768维向量
- `"bge-m3"`: 通用多语言模型，支持稠密、稀疏和多向量检索

---

## 📊 性能指标

### 检索性能
- **响应时间**: < 100ms（单条查询）
- **批量处理**: 100条/秒（GPU 加速）
- **索引大小**: 约 1GB/100万条 768维向量
- **内存占用**: 向量索引 + 原始数据 ≈ 2-3GB/100万条

### 精度指标
- **Recall@10**: 0.85+（在测试集上）
- **MRR**: 0.78+（平均倒数排名）

---

## 🔮 未来规划

### 短期优化
1. **混合检索**: 结合稀疏检索（BM25）和稠密检索，提升召回率
2. **重排序**: 引入 Cross-Encoder 对召回结果进行精细化排序
3. **缓存优化**: 实现查询缓存，提升高频查询响应速度

### 长期规划
1. **向量数据库**: 迁移到 Milvus/Qdrant 等专业向量数据库
2. **增量更新**: 支持实时索引更新，无需重启服务
3. **模型微调**: 使用领域数据微调嵌入模型
4. **多模态**: 支持法律文书图片、PDF 等多模态检索

---

## 🐛 故障排除

### 常见问题

1. **模型加载失败**
   ```
   检查 MODEL_PATH 和 BGE_M3_PATH 配置
   确保 HuggingFace 模型缓存可用
   ```

2. **向量文件不存在**
   ```
   运行 fact_embedding.py 生成向量文件
   检查 DATA_DIR 路径配置
   ```

3. **内存不足**
   ```
   减少索引加载的向量数量
   使用 HNSW 索引（内存效率更高）
   增加系统内存或使用分片索引
   ```

4. **GPU 不可用**
   ```
   检查 CUDA 驱动安装
   修改 config.py 中的 DEVICE 为 "cpu"
   ```

### 日志查看
```bash
# 查看服务日志
tail -f logs/dense_retrieval.log

# 调试模式启动
python main.py --log-level DEBUG
```

---

## 📚 相关资源

- [FAISS 官方文档](https://github.com/facebookresearch/faiss)
- [HNSW 算法论文](https://arxiv.org/abs/1603.09320)
- [BGE-M3 模型介绍](https://github.com/FlagOpen/FlagEmbedding)
- [Lawformer 论文](https://arxiv.org/abs/2105.03865)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
