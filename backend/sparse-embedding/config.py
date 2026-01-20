from pathlib import Path

class GlobalConfig:
    BASE_DIR = Path(r"f:\LegalAgent")
    BACKEND_DIR = BASE_DIR / "backend" / "sparse-embedding"
    DATA_DIR = BACKEND_DIR / "data"
    OUTPUT_DIR = BACKEND_DIR / "data"
    LOG_DIR = BACKEND_DIR / "logs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PORT = 4240 

class ExtractorConfig(GlobalConfig):
    KEYWORDS_FILE = GlobalConfig.DATA_DIR / "202_crimes_keywords.json"
    INPUT_DATASET = GlobalConfig.BASE_DIR / "dataset" / "final_all_data" / "first_stage" / "train.json"
    OUTPUT_DATASET = GlobalConfig.OUTPUT_DIR / "train_sparse_features.jsonl"
    RULES_YAML_PATH = GlobalConfig.BASE_DIR / "rules_yaml"
    SCHEMA_PATH = GlobalConfig.OUTPUT_DIR / "train_universal_schema.jsonl" 
    KEYWORDS_PATH = GlobalConfig.OUTPUT_DIR / "train_sparse_features.jsonl" 
    DB_PATH = GlobalConfig.DATA_DIR / "sparse_matrix.npz"
    maps_path = GlobalConfig.DATA_DIR / "one_hot_maps.json"
    fields_path = GlobalConfig.DATA_DIR / "schema_fields.json"
    SAMPLE_LIMIT = None

class RetrieverConfig(GlobalConfig):
    """稀疏检索阶段配置"""
    DB_PATH = GlobalConfig.DATA_DIR / "sparse_matrix.npz"
    KEYWORDS_PATH = ExtractorConfig.KEYWORDS_FILE