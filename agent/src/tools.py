"""Tool definitions and concrete implementations for the agent."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from config import KnowledgeBaseConfig


logger = logging.getLogger(__name__)


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return tool schemas for the LLM."""

    return [
        {
            "type": "function",
            "function": {
                "name": "knowledge_base_search",
                "description": "根据用户问题搜索本地检索服务并返回最相关的法条/案例片段",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "需要检索的自然语言问题",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回的候选数量，不传则使用服务端默认值",
                            "minimum": 1,
                            "maximum": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        # 预留更多工具定义
    ]


class KnowledgeBaseTools:
    """Wrapper around the retrieval pipeline and future tools."""

    def __init__(self, config: KnowledgeBaseConfig):
        self.config = config
        self.base_url = f"http://{config.host}:{config.port}"

    def knowledge_base_search(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """搜索相似案例."""
        try:
            payload = {
                "query": query,
                "mode": "hybrid",
                "top_k": top_k or self.config.top_k,
                "rerank": True
            }
            response = requests.post(
                f"{self.base_url}/search",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return {"results": []}

        results = data.get("results") or data.get("reranked_results") or []
        normalized: List[Dict[str, Any]] = []

        for rank, item in enumerate(results, start=1):
            normalized.append(
                {
                    "rank": rank,
                    "doc_id": item.get("doc_id") or item.get("fact_id") or item.get("id"),
                    "chunk_id": item.get("chunk_id"),
                    "content": item.get("content")
                    or item.get("fact")
                    or item.get("text")
                    or item.get("summary"),
                    "score": item.get("score") or item.get("rerank_score"),
                    "metadata": item,
                }
            )

        return normalized

    # 未来可以新增更多工具方法
