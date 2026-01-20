"""Agent entry point with ReAct + tool support."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional

from openai import OpenAI

from config import Config, KnowledgeBaseConfig
from tools import KnowledgeBaseTools, get_tool_definitions
from utils.case_extractor import CaseExtractor
from utils.batch_extractor import BatchCaseExtractor

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Conversation message container."""

    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AgenticRAG:
    """Legal assistant agent that follows the ReAct paradigm."""

    def __init__(
        self,
        config: Optional[Config] = None,
        enable_memory: bool = False,
    ) -> None:
        self.config = config or Config.from_env()
        self.enable_memory = enable_memory

        self._init_llm_client()

        kb_cfg = getattr(self.config, "knowledge_base", None)
        if kb_cfg is None:
            kb_cfg = KnowledgeBaseConfig()
        self.kb_tools = KnowledgeBaseTools(kb_cfg)

        self.conversation_history: List[Dict[str, Any]] = []
        self.tools = get_tool_definitions()
        self.case_extractor = CaseExtractor()
        self.batch_extractor = BatchCaseExtractor(batch_size=5, max_workers=3)
        
        # 设置系统提示词
        self.system_prompt = """你是一个法律助手，专门帮助分析案件并进行定罪。
请根据用户描述的案件和参考案例，分析案件性质并给出定罪建议。
分析时请注意:
1. 仔细对比案件事实与参考案例的相似性
2. 考虑案件中的关键要素(犯罪行为、后果、情节等)
3. 给出明确的罪名建议
4. 简要说明定罪理由

输出格式:
建议罪名：xxx
定罪理由：xxx
参考案例分析：xxx"""

        logger.info("AgenticRAG ready | provider=%s | model=%s", self.config.llm.provider, self.model)

    def _init_llm_client(self) -> None:
        """Create the OpenAI-compatible client."""

        client_config, model = self.config.llm.get_client_config()
        base_url = client_config.pop("base_url", None)

        if base_url:
            self.client = OpenAI(base_url=base_url, **client_config)
        else:
            self.client = OpenAI(**client_config)

        self.model = model

    # ---------------------------------------------------------------------
    # Prompt & messaging helpers
    # ---------------------------------------------------------------------
    def _get_system_prompt(self, prompt_override: Optional[str] = None) -> str:
        if prompt_override:
            return prompt_override

        return (
            "你是一名严格依赖检索结果的法律助手，遵循ReAct流程：\n"
            "1. Thought：分析问题并决定是否需要工具。\n"
            "2. Action：调用合适的工具（例如 knowledge_base_search）。\n"
            "3. Observation：总结工具返回的信息。\n"
            "4. Response：基于证据作答，必要时引用出处。\n"
            "若检索不到答案，请明确告知。"
        )

    def _build_messages(self, user_query: str, prompt_override: Optional[str] = None) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt if prompt_override is None else prompt_override}
        ]

        if self.enable_memory and self.conversation_history:
            messages.extend(self.conversation_history)

        # 获取相似案例
        search_results = self.kb_tools.knowledge_base_search(user_query)
        
        # 构建包含案例信息的提示词
        similar_cases = ""
        for i, result in enumerate(search_results.get("results", []), 1):
            fact = result.get('fact', '')
            accusation = result.get('accusation', '')
            
            # 提取案例关键信息
            extracted_info = self.case_extractor.extract_case_info(fact, accusation)
            
            similar_cases += f"\n参考案例{i}：\n"
            similar_cases += f"案件描述：{fact}\n"
            similar_cases += "结构化信息：\n"
            similar_cases += f"- 主体：{extracted_info.get('subject', '')}\n"
            similar_cases += f"- 行为：{extracted_info.get('action', '')}\n"
            similar_cases += f"- 结果：{extracted_info.get('result', '')}\n"
            similar_cases += f"- 其他要素：{extracted_info.get('others', '')}\n"
            similar_cases += f"判决结果：{accusation}\n"
            similar_cases += f"相似度得分：{result.get('score', 0.0):.4f}\n"
        
        analysis_prompt = (
            f"请分析以下案件：\n{user_query}\n\n"
            f"相似案例参考（已进行结构化处理）：{similar_cases}\n\n"
            "请根据当前案件描述和参考案例的结构化信息进行分析。重点关注：\n"
            "1. 案件主体、行为和结果的对比\n"
            "2. 相似案例的量刑情况\n"
            "3. 根据相似度进行合理参考\n"
            "请给出完整的定罪分析和可能的量刑建议。"
        )
        messages.append({"role": "user", "content": analysis_prompt})
        history_limit = self.config.agent.conversation_history_limit
        messages.extend(self.conversation_history[-history_limit:])

        messages.append({"role": "user", "content": user_query})
        return messages

    # ---------------------------------------------------------------------
    # Tool handling
    # ---------------------------------------------------------------------
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if tool_name == "knowledge_base_search":
                query = arguments.get("query", "").strip()
                if not query:
                    return {"status": "error", "message": "query 字段不能为空"}

                results = self.kb_tools.knowledge_base_search(query)

                if not results:
                    return {"status": "no_results", "message": "未找到相关文档"}

                formatted = []
                for idx, item in enumerate(results, start=1):
                    formatted.append(
                        {
                            "rank": idx,
                            "doc_id": item.get("doc_id") or item.get("id"),
                            "chunk_id": item.get("chunk_id"),
                            "content": item.get("content")
                            or item.get("text")
                            or item.get("summary", ""),
                            "score": item.get("score"),
                            "metadata": item.get("metadata", {}),
                        }
                    )

                return {
                    "status": "success",
                    "results": formatted[:3],  
                    "total_found": len(formatted),
                    "raw_results": formatted,
                }

            # 为未来扩展的工具预留接口
            dynamic_tool = getattr(self.kb_tools, tool_name, None)
            if callable(dynamic_tool):
                return dynamic_tool(**arguments)

            return {"status": "error", "message": f"未知工具: {tool_name}"}

        except Exception as exc:  # noqa: BLE001
            logger.exception("工具执行失败: %s", tool_name)
            return {"status": "error", "message": str(exc)}

    # ---------------------------------------------------------------------
    # Public APIs
    # ---------------------------------------------------------------------
    def chat(self, user_query: str) -> str:
        """Process a user query and return a response."""
        messages = self._build_messages(user_query)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.exception("Chat completion failed")
            return f"错误：{str(e)}"
            
    def query(
        self,
        user_query: str,
        stream: Optional[bool] = None,
        prompt_override: Optional[str] = None,
    ) -> Any:
        """Use ReAct loop to answer a standalone question."""

        stream = self.config.llm.stream if stream is None else stream
        messages = self._build_messages(user_query, prompt_override)

        for iteration in range(1, self.config.agent.max_iterations + 1):
            if self.config.agent.verbose:
                logger.info("Iteration %s", iteration)

            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tools if self.tools else None,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens,
                    stream=False,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("LLM 调用失败")
                return self._stream_response(str(exc)) if stream else str(exc)

            choice = response.choices[0]
            message = choice.message

            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
            }

            if message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ]

            messages.append(assistant_msg)

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    args_payload = tool_call.function.arguments or "{}"
                    try:
                        parsed_args = json.loads(args_payload)
                    except json.JSONDecodeError:
                        parsed_args = {}

                    tool_result = self._execute_tool(tool_call.function.name, parsed_args)
                    result_for_llm = {k: v for k, v in tool_result.items() if k != "raw_results"}

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result_for_llm, ensure_ascii=False),
                        }
                    )
                continue

            # 没有工具调用：到达最终回答
            if self.enable_memory:
                self.conversation_history.extend(
                    [
                        {"role": "user", "content": user_query},
                        assistant_msg,
                    ]
                )

            final_answer = message.content or ""
            return self._stream_response(final_answer) if stream else final_answer

        warning_msg = "未在限制轮次内完成推理，请尝试重述问题。"
        return self._stream_response(warning_msg) if stream else warning_msg

    def query_non_agentic(
        self,
        user_query: str,
        stream: Optional[bool] = None,
        prompt_override: Optional[str] = None,
    ) -> Any:
        """Fallback mode: simple retrieval + answer."""

        stream = self.config.llm.stream if stream is None else stream

        search_results = self.kb_tools.knowledge_base_search(user_query)
        context_parts: List[str] = []

        for idx, result in enumerate(search_results[:3], start=1):
            context_parts.append(
                f"[Doc {idx}] (ID: {result.get('doc_id')})\n{result.get('content', result.get('text', ''))}"
            )

        context_text = "\n\n".join(context_parts) if context_parts else "暂无检索结果。"

        system_prompt = prompt_override or (
            "你将看到知识库检索到的内容，只能基于这些内容回答并引用文献。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"检索内容:\n{context_text}\n\n问题: {user_query}\n请基于检索内容回答。",
            },
        ]

        try:
            if stream:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.config.llm.temperature,
                    stream=True,
                )

                def generator() -> Generator[str, None, None]:
                    for chunk in completion:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            yield delta.content

                return generator()

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.config.llm.temperature,
            )
            return response.choices[0].message.content or ""

        except Exception as exc:  # noqa: BLE001
            logger.exception("Non-agentic 查询失败")
            return self._stream_response(str(exc)) if stream else str(exc)

    def reset_conversation(self) -> None:
        self.conversation_history.clear()

    def _stream_response(self, content: str) -> Generator[str, None, None]:
        for char in content:
            yield char