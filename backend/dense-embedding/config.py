import os
import torch

class Config:
    VERBOSE = True 

    # 当前文件所在目录: .../backend/dense-embedding
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 项目根目录: .../LegalAgent (假设结构为 backend/dense-embedding/../../)
    PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))
    
    # 1. 模型路径
    MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "Lawformer")
    # 嵌入模型类型: "lawformer" 或 "bge-m3"
    EMBEDDING_MODEL_TYPE = "bge-m3"  # 默认使用lawformer
    
    # BGE-M3 模型路径 (如果需要本地路径)
    BGE_M3_PATH = r"D:\.cache\huggingface\hub\models--BAAI--bge-m3"
      
    # 2. 数据路径
    DATA_DIR = os.path.join(PROJECT_ROOT, "backend", "data", "fact_embeddings")
    if EMBEDDING_MODEL_TYPE == "lawformer": 
        EMBEDDING_FILE = os.path.join(DATA_DIR, "fact_embeddings.npy")
    else:
        EMBEDDING_FILE = os.path.join(DATA_DIR, "fact_embeddings_bge-m3.npy")
    INDEX_DIR = os.path.join(DATA_DIR, "indexes")
    LOG_DIR = os.path.join(_CURRENT_DIR, "logs")
    RAW_DATA_PATH = r"F:\LegalAgent\dataset\final_all_data\first_stage\train.json"

    SCHEMA_path = r"F:\LegalAgent\backend\sparse-embedding\data\train_universal_schema.jsonl"
    KEYWORDS_path = r"F:\LegalAgent\backend\sparse-embedding\data\train_sparse_features.jsonl"
    # ---------------------------------------------------------
    # 模型参数 (Model Parameters)
    # ---------------------------------------------------------
    MAX_SEQ_LENGTH = 512
    DEEVICE: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ServiceConfig:
    # ---------------------------------------------------------
    # 服务配置 (Service Configuration)
    # ---------------------------------------------------------
    HOST = "0.0.0.0"
    PORT = 4241
    
    # 索引类型: "hnsw" 或 "faiss"
    INDEX_TYPE = "faiss" 
    
    @classmethod
    def get_index_path(cls):
        """获取完整的索引文件路径"""
        return os.path.join(Config.INDEX_DIR, f"fact_index_{Config.EMBEDDING_MODEL_TYPE.lower()}.{cls.INDEX_TYPE}")

# 确保必要的目录存在
os.makedirs(Config.INDEX_DIR, exist_ok=True)
os.makedirs(Config.LOG_DIR, exist_ok=True)