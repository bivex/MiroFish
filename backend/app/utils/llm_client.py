"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI

from ..config import Config


class LLMClient:
    """LLM客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def prefers_native_tools(self) -> bool:
        """是否优先使用原生 tool-calling（当前主要用于 Groq 兼容）。"""
        return "groq.com" in (self.base_url or "").lower()

    def _strip_think_blocks(self, content: Any) -> str:
        text = content if isinstance(content, str) else ""
        return re.sub(r'<think>[\s\S]*?</think>', '', text).strip()

    def _parse_tool_arguments(self, arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str) or not arguments.strip():
            return {}
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw_arguments": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}

    def _format_native_tool_calls(self, tool_calls: Any) -> str:
        if not tool_calls:
            return ""

        chunks = []
        for call in tool_calls:
            function = getattr(call, "function", None)
            tool_name = getattr(function, "name", None)
            if not tool_name:
                continue
            payload = {
                "name": tool_name,
                "parameters": self._parse_tool_arguments(getattr(function, "arguments", None)),
            }
            chunks.append(f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>")
        return "\n\n".join(chunks)

    def _should_retry_without_native_tools(self, exc: Exception, tools: Optional[List[Dict[str, Any]]]) -> bool:
        if not tools or not self.prefers_native_tools():
            return False
        message = str(exc)
        return "Tool call validation failed" in message or "tool_use_failed" in message
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Any] = None,
    ) -> str:
        """
        发送聊天请求
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）
            
        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if response_format:
            kwargs["response_format"] = response_format
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if not self._should_retry_without_native_tools(exc, tools):
                raise

            fallback_kwargs = dict(kwargs)
            fallback_kwargs.pop("tools", None)
            fallback_kwargs.pop("tool_choice", None)
            response = self.client.chat.completions.create(**fallback_kwargs)

        message = response.choices[0].message
        content = self._strip_think_blocks(getattr(message, "content", None))
        native_tool_text = self._format_native_tool_calls(getattr(message, "tool_calls", None))
        if native_tool_text:
            return "\n\n".join(part for part in [content, native_tool_text] if part)
        return content
    
    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")

