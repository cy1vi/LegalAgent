import os

class EvalConfig:
    """
    Evaluation 模块统一配置文件。
    所有路径、参数、超参都集中到这里管理。
    """
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "final_all_data", "first_stage", "train.json")
    # EVAL_DATASET_PATH = os.path.join(
    #     PROJECT_ROOT,
    #     "backend",
    #     "evaluate",
    #     "data",
    #     "gold_standard_eval_tiered.jsonl"
    # )
    EVAL_DATASET_PATH = os.path.join(
        PROJECT_ROOT,
        "backend",
        "evaluate",
        "eval_dataset_v4.jsonl"
    )
    REPORT_OUTPUT_PATH = os.path.join(
        PROJECT_ROOT,
        "backend",
        "evaluate",
        "evaluate_summary.txt"
    )
    os.makedirs(os.path.dirname(EVAL_DATASET_PATH), exist_ok=True)

    # === 服务配置 ===
    SERVICE_HOST = "localhost"
    SERVICE_PORT = 4241
    SERVICE_URL = f"http://{SERVICE_HOST}:{SERVICE_PORT}/search"
    BATCH_SERVICE_URL = f"http://{SERVICE_HOST}:{SERVICE_PORT}/batch_search"

    # === 评测参数 ===
    # batch size 用于评测脚本 run_evaluation.py
    BATCH_SIZE = 2

    # Top-K（一般用于 search / batch_search 请求）
    TOP_K = 10
    TOP_K_FOR_REQUEST = 12  # 评测时先请求更多数据，避免过滤自身后不够

    # 每个 query 使用多少条 Positives（gold standard）
    MAX_POSITIVES = 10

    # 是否过滤 Query 自身
    FILTER_SELF_RETRIEVAL = True

    # K 值列表
    K_LIST = [3, 5, 10]

    NUM_queries = 500
    NUM_positives_per_query = 12

class CleanConfig:
    # 原始数据路径
    INPUT_FILE = r'F:\LegalAgent\backend\evaluate\data\gold_standard_eval_tiered.jsonl'
    # 清洗后保存路径
    OUTPUT_CLEAN_FILE = r'F:\LegalAgent\backend\evaluate\data\train_cleaned.json'
    # 被剔除的数据保存路径
    OUTPUT_DIRTY_FILE = r'F:\LegalAgent\backend\evaluate\data\train_dirty.json'
    
    ACCUSATION_MAP_PATH = r'F:\LegalAgent\dataset\final_all_data\meta\accu.txt'
    MODEL_PATH = r'F:\LegalAgent\legal-article-classifier\checkpoints\best_model.pt'
    LAWFORMER_PATH = r"D:\.cache\huggingface\hub\Lawformer" 
    
    MAX_LEN = 512
    BATCH_SIZE = 32 
    
    PRED_THRESHOLD = 0.5 
    HIGH_CONFIDENCE_THRESHOLD = 0.85 