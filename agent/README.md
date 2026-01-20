# Agent Project - 智能法律助手系统

## 概述
Agent项目是一个基于ReAct范式的智能法律助手系统，专门用于法律事实的密集检索和案件分析。该项目结合了先进的大语言模型(LLM)技术和法律知识库，能够理解复杂的法律案例并提供准确的分析结果。

## 核心功能
- 法律案例分析：使用ReAct（Reasoning + Acting）范式进行推理
- 知识库检索：本地法律知识库检索功能，支持语义相似度匹配
- 信息提取：自动提取案例中的关键信息并结构化输出
- 多LLM提供商支持：支持SiliconFlow、Doubao、Kimi、Moonshot等多种提供商

## 项目结构
```
agent
├── src/                           # 源代码目录
│   ├── agent.py                   # 核心代理类实现
│   ├── main.py                    # 应用入口点
│   ├── config.py                  # 配置管理
│   ├── tools.py                   # 工具定义和实现
│   ├── logger.py                  # 日志配置
│   ├── test_agent.py              # 单元测试
│   ├── test_agent_batch.py        # 批量测试
│   ├── extractor_llm_test.py      # 提取器测试
│   ├── get_datasets.py            # 数据集获取
│   ├── test_extractor_pipeline.py # 提取管道测试
│   ├── .env                       # 环境变量配置
│   ├── logs/                      # 日志目录
│   ├── output/                    # 输出目录
│   └── utils/                     # 工具函数
│       ├── case_extractor.py      # 案例提取器
│       ├── batch_extractor.py     # 批量提取器
│       └── schema_process.py      # 模式处理
├── docs/                          # 文档目录
│   └── api_spec.md                # API规范文档
├── tests/                         # 测试目录
│   ├── extractor_llm_test.py      # 提取器测试
│   └── test_api_client.py         # API客户端测试
├── requirements.txt               # 依赖包列表
└── README.md                      # 项目说明文档
```

## 安装与部署

### 环境要求
- Python 3.8+
- pip包管理器

### 安装步骤
1. 克隆项目：
```bash
git clone <repository-url>
cd agent
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 配置环境变量：
```bash
cp .env.example .env
# 编辑.env文件，填入相应的API密钥和其他配置
```

## 使用方法

### 基本用法
```python
from agent import AgenticRAG
from config import Config

# 初始化配置
config = Config.from_env()
# 创建代理实例
agent = AgenticRAG(config)

# 分析法律案例
case_description = "被告人王某某在KTV内对李某某实施殴打，导致轻伤二级"
response = agent.chat(case_description)
print(response)
```

### API接口
项目提供了RESTful API接口：

- **端点**: `/search`
- **方法**: `POST`
- **功能**: 搜索法律事实
- **请求体**:
```json
{
  "fact": "string",
  "top_k": "integer"
}
```

- **响应体**:
```json
[
  {
    "fact_id": "string",
    "score": "float",
    "rank": "integer",
    "fact": "string",
    "accusation": ["string"],
    "relevant_articles": ["string"],
    "imprisonment": {
      "term": "string",
      "details": "string"
    }
  }
]
```

## 测试
运行所有测试：
```bash
pytest tests/
```

或者运行特定的测试脚本：
```bash
python src/test_agent.py
```

## 配置选项
通过环境变量或配置文件可以设置以下参数：
- LLM提供商和模型选择
- API密钥
- 温度参数
- 最大token数
- 知识库类型和路径

## 扩展性
- 模块化的工具系统，易于添加新功能
- 可配置的LLM提供商，支持多种模型
- 灵活的知识库后端，可根据需求调整

