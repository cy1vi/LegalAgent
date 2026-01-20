# 法律案情检索服务

## 📖 项目简介

本服务是 LegalAgent 项目的检索服务组件，主要用于从案例库中检索相似案例。系统采用双路检索策略：

1. **语义检索** (Dense Retrieval)
   - 使用 BERT 模型进行文本编码
   - 支持案情相似度的语义匹配
   - 适合处理复杂的案情描述

2. **关键词检索** (Sparse Retrieval)
   - 基于 TF-IDF 的文本检索
   - 支持精确的关键词匹配
   - 适合处理罪名、法条等专业术语

## 📂 项目结构

```
retrieval-pipeline/
├── main.py                # FastAPI 服务入口
├── config.py              # 配置文件
├── rerank.py              # 检索逻辑实现
├── quick_test.py          # 测试脚本
└── README.md              # 本文档
```

## 📥 API 接口

### 检索接口

```
POST /api/v1/search
```

请求格式:
```json
{
    "query": {
        "text": "案情描述...",
        "filters": {
            "crime": "故意伤害罪",
            "year": "2020"
        }
    },
    "options": {
        "top_k": 10,
        "min_score": 0.5
    }
}
```

响应格式:
```json
{
    "results": [
        {
            "case_id": "case_001",
            "score": 0.85,
            "content": {
                "fact": "案情描述...",
                "crime": "故意伤害罪",
                "sentence": "判处有期徒刑...",
                "articles": ["第234条"]
            }
        }
    ],
    "total": 100,
    "time_cost": 0.15
}
```

## 🚀 部署说明

### 1. 环境要求
- Python 3.8+
- CUDA 11.7+ (可选，用于GPU加速)
- 8GB+ RAM

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动服务
```bash
# 开发环境
python main.py
```

## ⚙️ 配置说明

主要配置项（config.py）:

```python
# 服务配置
PORT = 8000
WORKERS = 4

# 检索配置
DEFAULT_TOP_K = 10
MIN_SCORE = 0.5

# 模型配置
MODEL_PATH = "./models"
USE_GPU = True
```

## 🔍 常见问题

### 1. 服务启动失败
- 检查端口是否被占用
- 确认模型文件是否存在
- 验证环境依赖是否完整

### 2. 检索性能问题
- 调整 batch_size 参数
- 检查索引是否正确构建
- 考虑使用 GPU 加速


