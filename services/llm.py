"""
LLM Service Wrapper - Using qwen-max
"""
from typing import List, Dict, Any, Optional
from openai import OpenAI
from config import settings


class LLMService:
    """LLM Service Class, wrapping qwen-max calls"""

    def __init__(self):
        if not settings.DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY is not set, please configure it in the .env file")

        self.client = OpenAI(
            api_key=settings.DASHSCOPE_API_KEY,
            base_url=settings.LLM_BASE_URL
        )
        self.model = settings.LLM_MODEL

    def chat(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.7,
            max_tokens: int = 4096,
            tools: Optional[List[Dict]] = None,
            tool_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send chat request

        Args:
            messages: List of messages
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens
            tools: List of tool definitions
            tool_choice: Tool selection strategy

        Returns:
            API response
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        response = self.client.chat.completions.create(**kwargs)
        return response

    def simple_chat(self, prompt: str, system_prompt: str = None) -> str:
        """
        Simple chat interface

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Returns:
            Assistant response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.chat(messages)
        return response.choices[0].message.content

    def chat_with_tools(
            self,
            messages: List[Dict[str, str]],
            tools: List[Dict],
            tool_choice: str = "auto"
    ) -> Dict[str, Any]:
        """
        Chat with tool calling

        Args:
            messages: List of messages
            tools: Tool definitions
            tool_choice: Tool selection strategy

        Returns:
            API response
        """
        response = self.chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice
        )
        return response


# Singleton pattern
_llm_service = None


def get_llm_service() -> LLMService:
    """Get the LLM service singleton instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
