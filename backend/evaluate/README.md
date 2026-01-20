# Legal Agent Evaluation Module

本目录包含用于评估法律文书检索系统（Dense Retrieval Service）性能的工具集。主要用于测试模型在**案情检索**、**罪名预测**、**法条推荐**及**量刑参考**方面的准确性。

## 📂 文件结构说明

```text
backend/evaluate/
├── data/
│   └── gold_standard_eval_tiered.jsonl  # [生成的] 标准评测数据集
├── prepare_datasets.py                  # [脚本] 数据集生成工具
├── run_evaluation.py                    # [脚本] 核心评测执行脚本
└── README.md                            # [文档] 说明文档
```

---

## 🛠️ 数据准备 (Data Preparation)

评测数据由 `prepare_datasets.py` 生成，旨在构建一个具有“标准答案”的测试集。

### 1. 数据来源
*   **原始数据**: `dataset/final_all_data/first_stage/train.json` (全量法律文书数据)。

### 2. 处理逻辑
脚本从全量数据中随机抽取 Query，并根据以下规则在剩余数据中寻找 **Positives (正样本)**：
*   **罪名一致**: 案由完全相同。
*   **法条一致**: 引用的法律条款集合一致。
*   **刑期相近**: 刑期差异在一定范围内。

### 3. 生成结果
生成的 `gold_standard_eval_tiered.jsonl` 每行包含：
*   `query`: 查询案件的案情 (Fact) 和元数据 (Meta)。
*   `positives`: 预先筛选出的最相似案件列表（作为 Gold Standard）。

---

## 📊 评测逻辑 (Evaluation Methodology)

评测由 `run_evaluation.py` 执行，通过调用检索服务 API (`/batch_search`) 进行端到端测试。

### 1. 核心机制
*   **Batch Search**: 使用批量接口加速评测（Batch Size = 16/32）。
*   **Self-Retrieval Filtering (去重)**:
    *   由于 Query 本身存在于数据库中，检索结果的第一名通常是 Query 自己。
    *   **处理方式**: 脚本会计算 Query 和 Result 的文本哈希，**自动剔除**与 Query 内容完全一致的结果，确保评估的是模型“举一反三”的能力。
*   **Top-K 截断**:
    *   向服务请求 Top-20 数据。
    *   剔除自身后，截取 Top-3, Top-5, Top-10 进行指标计算。
    *   **Positives 截断**: 为了适应 Top-10 检索场景，Recall 的分母（标准答案数量）被固定截断为 **10**。

### 2. 评估任务 (Tasks)
脚本同时评估四个维度的表现：

| 任务 | 说明 | 判定标准 |
| :--- | :--- | :--- |
| **Gold** | 标准集命中 | 检索结果是否属于预定义的 `positives` 列表（通过哈希比对）。 |
| **Accusation** | 罪名预测 | 检索结果的罪名集合是否与 Query 完全一致。 |
| **Article** | 法条推荐 | 检索结果的法条集合是否与 Query 完全一致。 |
| **Imprisonment** | 刑期参考 | 检索结果的刑期是否在容忍度内 (±20% 或 ±6个月)。 |

---

## 📈 指标说明 (Metrics)

评测报告 (`evaluate_summary.txt`) 包含以下指标：

### 1. Precision@K (查准率)
*   **含义**: Top-K 结果中有多少比例是正确的？
*   **公式**: `匹配数量 / K`
*   **用途**: 衡量检索结果的**准确性**（用户看到的 10 个结果里有几个是对的）。

### 2. Recall@K (召回率)
*   **含义**: 在 Top-K 结果中，找到了多少个目标正样本？
*   **公式**: `匹配数量 / 10` (分母固定为 10)
*   **用途**: 衡量检索结果的**覆盖率**（是否漏掉了关键案例）。

### 3. Hit Rate@K (命中率)
*   **含义**: Top-K 结果中**至少有一个**正确结果的 Query 比例。
*   **用途**: 衡量系统的**保底能力**。

### 4. F1 Score
*   **Macro-F1**: 先计算每个 Query 的 F1，再求平均。平等对待每个 Query。
*   **Micro-F1**: 基于全局的总命中数和总检索数计算 F1。平等对待每个样本。

---

## 🚀 使用指南

### 第一步：启动检索服务
在进行评测前，必须确保后端服务已启动并加载了模型和索引。
```powershell
# 在终端 1 中运行
python main.py
```

### 第二步：(可选) 生成新数据集
如果你想重新采样测试数据：
```powershell
# 在终端 2 中运行
python datasets_v4.py
```

### 第三步：运行评测
```powershell
# 在终端 2 中运行
python run_evaluation_v4.py
```

评测完成后，详细报告将生成于同目录下的 `evaluate_summary.txt`。