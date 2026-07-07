"""智谱 GLM 大模型客户端封装。"""

import logging
import time
from collections.abc import AsyncIterator, Iterator
from zhipuai import ZhipuAI
from app.config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class GLMClient:
    """GLM 同步/流式对话客户端。"""

    def __init__(self) -> None:
        """初始化智谱 API 客户端，读取配置中的 API Key 与模型名。"""
        self.client = ZhipuAI(api_key=settings.zhipuai_api_key)
        self.model = settings.glm_model

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3) -> str:
        """发起非流式对话补全。
        Args:
            messages: OpenAI 格式的消息列表，含 role 与 content。
            temperature: 采样温度，越高越发散。
        Returns:
            str: 模型回复文本，无内容时返回空字符串。
        """
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        content = response.choices[0].message.content or ""
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "[glm_chat] completed model=%s temperature=%s duration_ms=%s response_len=%s",
            self.model,
            temperature,
            duration_ms,
            len(content),
        )
        return content

    def stream_chat(
        self, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> Iterator[str]:
        """发起流式对话补全。
        Args:
            messages: OpenAI 格式的消息列表。
            temperature: 采样温度。
        Yields:
            str: 逐 token 输出的文本片段。
        """
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        token_count = 0
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                token_count += len(delta.content)
                yield delta.content
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "[glm_stream] completed model=%s temperature=%s duration_ms=%s chars=%s",
            self.model,
            temperature,
            duration_ms,
            token_count,
        )

    async def astream_chat(
        self, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> AsyncIterator[str]:
        """异步流式对话（包装同步 SDK，供 SSE 使用）。
        Args:
            messages: OpenAI 格式的消息列表。
            temperature: 采样温度。
        Yields:
            str: 逐 token 输出的文本片段。
        """
        for token in self.stream_chat(messages, temperature):
            yield token


_glm: GLMClient | None = None


def get_glm() -> GLMClient:
    """获取 GLM 客户端单例。
    Returns:
        GLMClient: 全局共享的 GLM 客户端实例。
    """
    global _glm
    if _glm is None:
        _glm = GLMClient()
    return _glm
