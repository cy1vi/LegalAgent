"""Configuration for Agentic RAG System"""
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum


# 获取当前文件所在目录
current_dir = Path(__file__).parent.absolute()
env_path = current_dir / '.env'
load_dotenv(env_path)

class Config:
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOG_DIR = os.path.join(_CURRENT_DIR, "logs")

class Provider(str, Enum):
    """Supported LLM providers"""
    SILICONFLOW = "siliconflow"
    DOUBAO = "doubao"
    KIMI = "kimi"
    MOONSHOT = "moonshot"
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GROQ = "groq"
    TOGETHER = "together"
    DEEPSEEK = "deepseek"


class KnowledgeBaseType(str, Enum):
    """Knowledge base backend types"""
    LOCAL = "local"  # Local retrieval pipeline



@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: str = "openrouter"  # Default provider
    model: Optional[str] = None  # Will use provider defaults if not specified
    api_key: Optional[str] = None  # Will read from env if not provided
    temperature: float = 0.7
    max_tokens: int = 100000
    stream: bool = True
    
    # Provider-specific defaults
    PROVIDER_DEFAULTS = {
        "siliconflow": {
            "model": "Qwen/Qwen3-235B-A22B-Thinking-2507",
            "base_url": "https://api.siliconflow.cn/v1"
        },
        "doubao": {
            "model": "doubao-seed-1-6-thinking-250715",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3"
        },
        "kimi": {
            "model": "kimi-k2-0905-preview",
            "base_url": "https://api.moonshot.cn/v1"
        },
        "moonshot": {
            "model": "kimi-k2-0905-preview",
            "base_url": "https://api.moonshot.cn/v1"
        },
        "openrouter": {
            "model": "anthropic/claude-3-sonnet",
            "base_url": "https://openrouter.ai/api/v1"
        },
        "openai": {
            "model": "gpt-4o-2024-11-20",
            "base_url": "https://api.openai.com/v1"
        },
        "groq": {
            "model": "llama-3.3-70b-versatile",
            "base_url": "https://api.groq.com/openai/v1"
        },
        "together": {
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "base_url": "https://api.together.xyz"
        },
        "deepseek": {
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1"
        }
    }
    
    @classmethod
    def get_api_key(cls, provider: str) -> Optional[str]:
        """Get API key from environment"""
        env_mappings = {
            "siliconflow": "SILICONFLOW_API_KEY",
            "doubao": "ARK_API_KEY",
            "kimi": "MOONSHOT_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "groq": "GROQ_API_KEY",
            "together": "TOGETHER_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY"
        }
        api_key = os.getenv(env_mappings.get(provider.lower(), ""))
        if api_key:
            return api_key.strip('"\'')
        return None
    
    def get_client_config(self) -> tuple[Dict[str, Any], str]:
        """Get OpenAI client configuration and model name"""
        provider_lower = self.provider.lower()
        defaults = self.PROVIDER_DEFAULTS.get(provider_lower, {})
        
        # 直接从环境变量获取API key
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip('"\'')
        if not api_key:
            raise ValueError(f"API key required for provider '{provider_lower}'")
        
        # Get API key
        api_key = self.api_key or self.get_api_key(provider_lower)
        if not api_key:
            raise ValueError(f"API key required for provider '{provider_lower}'")
        
        # Build config
        model = self.model or defaults.get("model", "openai/gpt-4-turbo-preview")
        config = {
            "api_key": api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "default_headers": {
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "LegalAgent"
            }
        }
        
        return config, model


@dataclass
class KnowledgeBaseConfig:
    """Knowledge base configuration"""
    type: KnowledgeBaseType = KnowledgeBaseType.LOCAL
    
    # Local retrieval pipeline config
    host: str = "localhost"
    port: int = 8000
    top_k: int = 3
    timeout: int = 30
    
    
    
@dataclass 
class AgentConfig:
    """Agent configuration"""
    max_iterations: int = 2  # Max reasoning iterations
    enable_reasoning_trace: bool = True
    enable_citations: bool = True
    conversation_history_limit: int = 25  # Max conversation turns to keep
    verbose: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    dataset_path: str = "evaluation/legal_qa_dataset.json"
    results_path: str = "evaluation/results"
    metrics: list = field(default_factory=lambda: ["accuracy", "relevance", "citation_quality"])


@dataclass
class Config:
    """Main configuration"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    knowledge_base: KnowledgeBaseConfig = field(default_factory=KnowledgeBaseConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        config = cls()
        
        # Override from env
        if provider := os.getenv("LLM_PROVIDER"):
            config.llm.provider = provider
        if model := os.getenv("LLM_MODEL"):
            config.llm.model = model
        
        return config
