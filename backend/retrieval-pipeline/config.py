from enum import Enum
from typing import Optional
import os
import torch
from dataclasses import dataclass

class BaseConfig:
    VERBOSE = True
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(_CURRENT_DIR, "logs")

class SearchMode(str, Enum):
    DENSE  = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"

@dataclass
class RerankerConfig:
    model_path = "BAAI/bge-reranker-v2-m3"
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16: bool = True
    batch_size: int = 1
    max_length: int = 1024


@dataclass
class ServicesConfig:
    dense_service_url: str = "http://localhost:4241"
    sparse_service_url: str = "http://localhost:4240"

class PipelineConfig:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        debug: bool = True,
        default_top_k: int = 10,
        rerank_top_k: int = 30
    ):
        self.host = host
        self.port = port
        self.debug = debug
        self.default_top_k = default_top_k
        self.rerank_top_k = rerank_top_k
        self.reranker = RerankerConfig()
        self.services = ServicesConfig()

    @classmethod
    def from_env(cls) -> 'PipelineConfig':
        """从环境变量加载配置"""
        return cls(
            host=os.getenv("PIPELINE_HOST", "0.0.0.0"),
            port=int(os.getenv("PIPELINE_PORT", "8000")),
            debug=os.getenv("DEBUG", "true").lower() == "true",
            default_top_k=int(os.getenv("DEFAULT_TOP_K", "20")),
            rerank_top_k=int(os.getenv("RERANK_TOP_K", "100"))
        )