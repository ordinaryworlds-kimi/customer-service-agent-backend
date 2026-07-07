"""阿里云通义千问（DashScope OpenAI 兼容接口）客户端封装。"""

import logging
import time
from collections.abc import AsyncIterator, Iterator

from openai import OpenAI

from app.config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Qwen3 混合思考模型：非流式调用必须显式关闭 thinking，否则 DashScope 返回 400。
_QWEN_EXTRA_BODY = {"enable_thinking": False}


class QwenClient:
    """通义千问同步/流式对话客户端。"""

    def __init__(self) -> None:
        """初始化 DashScope 客户端，读取配置中的 API Key、Base URL 与模型名。"""
        self.client = OpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
        )
        self.model = settings.qwen_model

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3) -> str:
        """发起非流式对话补全。"""
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            extra_body=_QWEN_EXTRA_BODY,
        )
        content = response.choices[0].message.content or ""
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "[qwen_chat] completed model=%s temperature=%s duration_ms=%s response_len=%s",
            self.model,
            temperature,
            duration_ms,
            len(content),
        )
        return content

    def stream_chat(
        self, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> Iterator[str]:
        """发起流式对话补全。"""
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
            extra_body=_QWEN_EXTRA_BODY,
        )
        token_count = 0
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                token_count += len(delta.content)
                yield delta.content
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "[qwen_stream] completed model=%s temperature=%s duration_ms=%s chars=%s",
            self.model,
            temperature,
            duration_ms,
            token_count,
        )

    async def astream_chat(
        self, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> AsyncIterator[str]:
        """异步流式对话（包装同步 SDK，供 SSE 使用）。"""
        for token in self.stream_chat(messages, temperature):
            yield token


_qwen: QwenClient | None = None


def get_qwen() -> QwenClient:
    """获取通义千问客户端单例。"""
    global _qwen
    if _qwen is None:
        _qwen = QwenClient()
    return _qwen
