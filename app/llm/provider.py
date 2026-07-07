"""LLM 提供商工厂，按配置选择智谱 GLM 或阿里云通义千问。"""

from typing import Protocol

from app.config.settings import get_settings
from app.llm.glm import GLMClient, get_glm
from app.llm.qwen import QwenClient, get_qwen


class LLMClient(Protocol):
    """各 LLM 客户端共用的 chat / stream 接口。"""

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3) -> str: ...

    async def astream_chat(
        self, messages: list[dict[str, str]], temperature: float = 0.3
    ): ...


def get_llm() -> GLMClient | QwenClient:
    """按 ``llm_provider`` 配置返回对应 LLM 客户端单例。"""
    provider = get_settings().llm_provider.lower()
    if provider == "qwen":
        return get_qwen()
    if provider == "glm":
        return get_glm()
    raise ValueError(f"Unsupported llm_provider: {provider!r}. Use 'glm' or 'qwen'.")
