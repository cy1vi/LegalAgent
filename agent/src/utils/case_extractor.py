"""Case information extraction module."""

import json
import logging
import os
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CaseExtractor:
    """Legal case information extractor."""
    
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize the extractor with API configuration."""
        self.api_url = api_url or os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("LLM_MODEL")

    def extract_case_info(self, fact_text: str, crime_str: str) -> Dict[str, Any]:
        """Extract structured information from case text.
        
        Args:
            fact_text: The case fact description text
            crime_str: The crime type string
            
        Returns:
            Dict containing extracted information with keys:
            - subject: The subject of the case
            - action: The criminal action
            - result: The consequence
            - others: Other relevant information
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "Legal Case Extractor"
            }
            
            prompt = (
                "请从以下法律事实中提取关键信息，若文字过多可压缩，尽量提取原文内容，并以纯JSON格式返回，仅包含以下字段：\n"
                "- subject: 主体（如'刘某某'）\n"
                "- action: 行为（简要描述其实施的具体行为）\n"
                "- result: 结果（如'造成森林破坏'或'导致财产损失'等）\n"
                "- others: 其他对于定罪判刑有关的内容\n\n"
                f"罪名如下：{crime_str}\n"
                "法律事实如下：\n\n"
                f"{fact_text}"
            )
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个法律信息提取助手。请严格按JSON格式输出，只包含以下四个字段：subject（主体）、action（行为）、result（结果）、others（其他定罪相关内容）。只输出纯 JSON。"
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1024
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                logger.error(f"API Error {response.status_code}: {response.text}")
                return self._get_default_structure()
                
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON: {content}")
                return self._get_default_structure()
                
        except Exception as e:
            logger.error(f"Error in case extraction: {str(e)}")
            return self._get_default_structure()
            
    def _get_default_structure(self) -> Dict[str, str]:
        """Return default structure when extraction fails."""
        return {
            "subject": "",
            "action": "",
            "result": "",
            "others": ""
        }