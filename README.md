# 智能法律助手系统 (LegalAgent)

## 项目概述

本项目是一个基于大语言模型（LLM）的智能法律助手系统，旨在帮助用户理解和分析法律案例。系统采用检索增强生成（RAG）技术，结合结构化信息提取和案例分析，为用户提供相关的法律建议和案例参考。

## 核心架构

系统主要由三个核心模块组成：

1. **Backend（后端检索服务）**：提供混合检索能力，包括稠密检索和稀疏检索
2. **Agent（智能代理）**：处理用户查询，整合检索结果并生成最终回答
3. **Extraction（信息提取）**：从法律文本中提取结构化信息
4. **Rules_YAML（规则定义）**：定义法律事实的结构化模式

## 详细功能说明

### Backend - 检索服务

Backend 模块负责从大量法律案例中检索相关信息，采用双路检索策略：

#### 1. 稠密检索（Dense Retrieval）
- 使用 BERT 模型进行文本编码
- 支持案情相似度的语义匹配
- 适合处理复杂的案情描述

#### 2. 稀疏检索（Sparse Retrieval）
- 基于 TF-IDF 和自定义正则化 schema 的文本检索
- 支持精确的关键词匹配
- 适合处理罪名、法条等专业术语
- 当前 schema 急需优化，效果有待提升

#### 项目结构
```
backend/
├── data/                    # 数据存储目录
├── dense-embedding/         # 稠密向量索引
├── sparse-embedding/        # 稀疏向量索引
├── retrieval-pipeline/      # 检索管道实现
│   ├── main.py             # FastAPI 服务入口
│   ├── config.py           # 配置文件
│   ├── rerank.py           # 检索逻辑实现
│   └── quick_test.py       # 测试脚本
└── evaluate/                # 评估模块
```

### Agent - 智能代理

Agent 模块是系统的交互核心，负责处理用户输入并生成最终的回答：

#### 主要功能
- 接收用户输入的案情描述
- 调用后端检索服务获取相关案例
- 使用 LLM 对检索结果进行结构化处理
- 将结构化后的相似案例及罪名提交给最终处理 LLM 进行判断
- 生成易懂的法律分析结果

#### 当前状态
- 系统基本功能已实现，效果良好
- 实际效果仍有待提高
- LLM 结构化提取能力需要进一步优化
- 目前尚未提取被告身份信息，导致许多错误
- 需要改进处理细节，特别是使用 RAG 技术提取罪名之间的定义特征差异

#### 项目结构
```
agent/
├── src/                    # 源代码目录
│   ├── agent.py           # 核心代理类实现
│   ├── main.py            # 应用入口点
│   ├── config.py          # 配置管理
│   ├── tools.py           # 工具定义和实现
│   ├── logger.py          # 日志配置
│   ├── test_agent.py      # 单元测试
│   ├── test_agent_batch.py # 批量测试
│   ├── extractor_llm_test.py # 提取器测试
│   ├── get_datasets.py    # 数据集获取
│   ├── test_extractor_pipeline.py # 提取管道测试
│   ├── .env              # 环境变量配置
│   ├── logs/             # 日志目录
│   ├── output/           # 输出目录
│   └── utils/            # 工具函数
│       ├── case_extractor.py # 案例提取器
│       ├── batch_extractor.py # 批量提取器
│       └── schema_process.py # 模式处理
├── docs/                   # 文档目录
├── tests/                  # 测试目录
├── requirements.txt        # 依赖包列表
└── README.md               # 项目说明文档
```

### Extraction - 信息提取

Extraction 模块负责从法律文本中提取结构化信息：

#### 主要功能
- 使用 LLM 进行法律事实提取
- 基于 TF-IDF 的罪名词条提取
- 支持多种 LLM 提供商（SiliconFlow、Doubao、Kimi、Moonshot 等）
- 实现了增强的罪名关键词提取器

#### 项目结构
```
extraction/
├── .env                           # 环境变量配置
├── 202_factor_discovery.py        # 因素发现脚本
├── agent.py                       # 提取代理
├── build_one_prompt.py            # 构建单个提示
├── build_prompt.py                # 构建提示模板
├── compare.py                     # 比较脚本
├── config.py                      # 配置文件
├── enhanced_crime_keyword_extractor.py # 增强罪名关键词提取器
├── extract_missing_crimes_keywords.py # 提取缺失罪名关键词
├── factor_discovery_agent.py      # 因素发现代理
├── tf_idf_test.py                 # TF-IDF 测试
├── universal_fact_extractor.py    # 通用事实提取器
└── stopwords.txt                  # 停用词表
```

### Rules_YAML - 规则定义

Rules_YAML 模块定义了法律事实的结构化模式：

#### 主要组件
- `act.yaml` - 行为特征规则
- `context.yaml` - 作案情境规则
- `participation.yaml` - 参与方式规则
- `pattern.yaml` - 行为模式规则
- `result.yaml` - 结果特征规则
- `universal_fact_schema.json` - 通用事实结构定义

#### 项目结构
```
rules_yaml/
├── act.yaml                      # 行为特征规则
├── context.yaml                  # 作案情境规则
├── participation.yaml            # 参与方式规则
├── pattern.yaml                  # 行为模式规则
├── result.yaml                   # 结果特征规则
└── universal_fact_schema.json    # 通用事实结构定义
```

## 当前进展与挑战

### 已实现功能
- 基础的混合检索系统（稠密+稀疏）
- 智能代理框架，支持结构化信息处理
- 法律事实提取和结构化
- 多 LLM 提供商支持

### 存在的问题
1. **LLM 结构化提取优化**：
   - 当前 LLM 在提取结构化信息方面仍有待优化
   - 特别是被告身份信息的提取缺失，导致许多错误
   - 需要进一步完善处理细节

2. **检索效果提升**：
   - 目前检索无法直接达到 recall@5 100% 的效果
   - 导致 Agent 无法处理很多内容
   - 计划加入相似罪名检索功能，而非仅依赖重排序对比

3. **Schema 优化**：
   - 稀疏检索使用的自定义正则化 schema 急需修改
   - 当前效果较差，需要重新设计

4. **罪名特征差异提取**：
   - 需要使用 RAG 技术提取不同罪名之间的定义特征差异
   - 这将有助于更精确的案例匹配和分析

### 解决方案方向
- 改进 LLM 提示工程，提高结构化提取准确性
- 引入相似罪名检索机制，扩展检索范围
- 优化稀疏检索的 schema 设计
- 实现罪名间特征差异的 RAG 提取功能

## 项目结构总览

```
LegalAgent/
├── agent/                        # 智能代理模块
├── Agent_test/                   # 代理测试模块
├── RAG4Law/                      # RAG 实现
├── backend/                      # 后端检索服务
├── extraction/                   # 信息提取模块
├── law_process/                  # 法律数据处理
├── rules_yaml/                   # 规则定义
├── dataset/                      # 数据集
├── model/                        # 模型文件
├── cluster/                      # 聚类分析
├── 202_crimes_keywords.json      # 罪名关键词数据
├── 项目方案.md                    # 详细项目计划（中文）
├── README.md                     # 项目说明文档
└── requirements.txt              # 主项目依赖
```

## 安装与部署

### 环境要求
- Python 3.8 或更高版本
- pip 包管理器

### 安装步骤

1. 克隆项目：
```bash
git clone <repository-url>
cd LegalAgent
```

2. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. 安装主项目依赖：
```bash
pip install -r requirements.txt
```

4. 安装各模块依赖：
```bash
cd agent
pip install -r requirements.txt
cd ../backend
pip install -r requirements.txt
cd ../extraction
pip install -r requirements.txt
```

5. 设置环境变量：
```bash
cp env.example .env
# 编辑 .env 文件，填入相应的 API 密钥和其他配置
```

## 使用方法

### 启动后端检索服务

1. 进入检索管道目录：
```bash
cd backend/retrieval-pipeline
```

2. 启动服务：
```bash
python main.py
```

### 运行智能代理

1. 进入代理目录：
```bash
cd agent/src
```

2. 运行主应用：
```bash
python main.py
```

### 测试与评估

1. 批量测试代理：
```bash
cd Agent_test
python test_agent_batch.py
```

2. 单次查询测试：
```bash
python test_one_query.py
```

### 知识提取流程

1. 运行知识提取管道：
```bash
cd extraction
python enhanced_crime_keyword_extractor.py
```

2. 构建提取提示：
```bash
python build_prompt.py
```

## 配置说明

系统通过环境变量进行配置。复制 `env.example` 到 `.env` 并填写 API 密钥和其他设置。主要配置选项包括：

- `LLM_PROVIDER`: LLM 提供商（如 siliconflow, openai, kimi, doubao）
- `LLM_MODEL`: 使用的具体模型
- 各种提供商的 API 密钥（MOONSHOT_API_KEY, ARK_API_KEY 等）
- 知识库设置（KB_TYPE, host, port 等）

## 当前系统特点

### 优势
- 采用混合检索策略，结合稠密和稀疏检索的优势
- 使用结构化信息提取，提高案例匹配精度
- 支持多种 LLM 提供商，灵活适应不同场景
- 模块化设计，便于维护和扩展

### 待优化方向
- 提高 LLM 结构化提取的准确性，特别是被告身份信息
- 优化稀疏检索的 schema，提升检索效果
- 实现罪名间特征差异的 RAG 提取功能
- 增加相似罪名检索机制，扩大检索覆盖范围

## 技术栈

- Python 3.8+
- PyTorch/TensorFlow (用于模型)
- FastAPI (用于后端服务)
- Elasticsearch (用于检索)
- HanLP (用于中文自然语言处理)
- 各种 LLM 提供商 API

## 未来发展方向

1. **增强结构化提取能力**：改进 LLM 提示工程，提高信息提取准确性
2. **优化检索算法**：引入更先进的检索技术，提升召回率和准确率
3. **扩展法律领域覆盖**：增加更多法律领域的案例和规则
4. **提升用户体验**：优化交互界面和响应速度
5. **加强法律合规性**：确保系统输出符合法律规范和伦理要求

## 贡献指南

我们欢迎对 LegalAgent 项目的贡献！以下是贡献方式：

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/awesome-feature`)
3. 进行修改
4. 提交更改 (`git commit -m 'Add awesome feature'`)
5. 推送到分支 (`git push origin feature/awesome-feature`)
6. 开启 Pull Request

请确保您的代码遵循项目风格指南，并在提交 PR 前通过所有测试。

## 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件。请注意，某些组件可能有不同的许可条款，特别是在集成外部 API 或具有特定使用限制的数据集时。

## 致谢

- CAIL2018 数据集提供法律案例数据
- 开源库使本项目成为可能
- 在开发过程中提供领域专业知识的法律专业人士
