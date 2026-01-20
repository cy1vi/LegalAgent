"""Agentic RAG System with ReAct Pattern"""

import json
import logging
from typing import List, Dict, Any, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
from openai import OpenAI
from config import Config
import dotenv
# Load environment variables from .env file
dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Represents a message in the conversation"""
    role: str  # "user", "assistant", "tool"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LegalCaseSchemaExtractor:
    """Legal case schema extraction agent using ReAct pattern"""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the legal case schema extraction agent"""
        self.config = config or Config.from_env()
        
        # Initialize LLM client
        self._init_llm_client()
        
        # Conversation history
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Legal case schema structure
        self.schema_structure = {
            "case_id": "案件编号",
            "case_type": "案件类型",
            "court": "审理法院",
            "judges": "审判人员",
            "parties": {
                "plaintiff": "原告",
                "defendant": "被告",
                "third_party": "第三人"
            },
            "facts": "案件事实",
            "claims": "诉讼请求",
            "arguments": "争议焦点",
            "evidence": "证据清单",
            "judgment": "判决结果",
            "legal_basis": "法律依据",
            "date_filed": "立案日期",
            "date_judgment": "判决日期"
        }
        
        logger.info(f"Initialized LegalCaseSchemaExtractor with provider: {self.config.llm.provider}")
    
    def _init_llm_client(self):
        """Initialize the LLM client based on provider"""
        client_config, model = self.config.llm.get_client_config()
        
        # Extract base_url if present
        base_url = client_config.pop("base_url", None)
        
        # Create OpenAI client
        if base_url:
            self.client = OpenAI(base_url=base_url, **client_config)
        else:
            self.client = OpenAI(**client_config)
        
        self.model = model
        logger.info(f"Using model: {self.model}")
    
    def _get_system_prompt(self, prompt_override: Optional[str] = None) -> str:
        """Get system prompt for legal case schema extraction"""
        if prompt_override:
            return prompt_override
        else:
            schema_desc = json.dumps(self.schema_structure, ensure_ascii=False, indent=2)
            return f"""你是一个专业的法律案件信息提取助手。你的任务是从用户提供的法律文书中提取结构化信息，并按照指定的schema格式输出。

请严格按照以下ReAct模式工作：
1. Thought: 分析用户提供的法律文书内容，确定需要提取的信息。
2. Action: 使用工具从文档中提取所需信息。
3. Observation: 观察提取到的信息是否完整准确。
4. Final Answer: 按照以下JSON schema格式输出提取结果：

{schema_desc}

注意事项：
- 所有字段都应尽量填写，如果原文中没有相关信息，请填入"未提及"或空字符串
- 日期格式统一为YYYY-MM-DD
- 多个人员或多项内容用列表表示
- 保持客观，只提取原文信息，不要添加推论
- 如果不确定某些信息，标注为"待确认"
"""

    def _build_messages(self, user_query: str, prompt_override: Optional[str] = None) -> List[Dict[str, Any]]:
        """Build messages for the LLM including conversation history and chosen system prompt"""
        system_prompt = self._get_system_prompt(prompt_override=prompt_override)
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (limited)
        history_limit = self.config.agent.conversation_history_limit
        if len(self.conversation_history) > history_limit:
            messages.extend(self.conversation_history[-history_limit:])
        else:
            messages.extend(self.conversation_history)

        messages.append({"role": "user", "content": user_query})
        return messages

    def _stream_response(self, content: str) -> Generator[str, None, None]:
        """Stream response content"""
        # Simple character streaming for demonstration
        for char in content:
            yield char
    
    def extract_schema(self, legal_document: str, prompt_override: Optional[str] = None, stream: bool = None) -> Any:
        """
        Extract legal case schema from legal document.
        
        Args:
            legal_document: The legal document text to analyze
            stream: Whether to stream the response
            
        Returns:
            The extracted schema (dict or generator for streaming)
        """
        if stream is None:
            stream = self.config.llm.stream
        
        try:
            # Build prompt with legal document
            system_prompt = self._get_system_prompt(prompt_override=prompt_override)
            
            user_prompt = (
                "请从以下法律文书中提取案件信息并按要求的schema格式输出：\n\n"
                "文书内容：\n"
                + legal_document +
                "\n\n请严格按照schema格式输出JSON结果，不要添加其他解释文字。"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            if stream:
                response_stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                    stream=True
                )
                
                def response_generator():
                    for chunk in response_stream:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                
                return response_generator()
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                    stream=False
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error in schema extraction: {e}")
            error_msg = f"Error processing document: {str(e)}"
            if stream:
                return self._stream_response(error_msg)
            else:
                return error_msg
    def analyze_case(self, system_prompt: str, user_content: str, stream: bool = None) -> Any:
        """
        通用的案件分析方法，不强制添加 Schema 提取指令
        """
        if stream is None:
            stream = self.config.llm.stream

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            
            if stream:
                # ...复用流式逻辑...
                response_stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                    stream=True
                )
                def response_generator():
                    for chunk in response_stream:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                return response_generator()
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                    stream=False
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error in case analysis: {e}")
            return f"Error: {str(e)}"

    def validate_schema(self, extracted_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the extracted schema against expected structure.
        
        Args:
            extracted_schema: The extracted schema to validate
            
        Returns:
            Validation results with missing or incorrect fields
        """
        validation_result = {
            "valid": True,
            "missing_fields": [],
            "extra_fields": [],
            "issues": []
        }
        
        # Check for missing required fields
        for key in self.schema_structure:
            if key not in extracted_schema:
                validation_result["missing_fields"].append(key)
                validation_result["valid"] = False
        
        # Check for extra fields
        for key in extracted_schema:
            if key not in self.schema_structure:
                validation_result["extra_fields"].append(key)
        
        return validation_result
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
