"""customer-service-agent 测试公共 fixture。

提供 Mock GLM 客户端、会员上下文、数据库会话等，
使测试无需连接真实 MySQL / Milvus / 智谱 API 即可运行。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.models.schemas import MemberContext
from app.tools.registry import ToolContext


@dataclass
class MockConversation:
    """模拟会话 ORM 对象，供编排层测试使用。"""

    id: int = 1
    member_id: int = 100
    member_username: str = "testuser"
    title: str = "测试会话"
    status: int = 1
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


class MockGLMClient:
    """可配置的 GLM 客户端，按预设队列返回 chat / stream 响应。"""

    def __init__(
        self,
        chat_responses: list[str] | None = None,
        stream_tokens: list[str] | None = None,
    ) -> None:
        # chat_responses: 每次 glm.chat() 依次弹出的返回值
        self.chat_responses = list(chat_responses or [])
        # stream_tokens: glm.astream_chat() 逐 token 产出的内容
        self.stream_tokens = list(stream_tokens or ["汇总", "回复"])
        # 记录调用历史，便于断言 Prompt 是否正确传入
        self.chat_calls: list[list[dict[str, str]]] = []
        self.stream_calls: list[list[dict[str, str]]] = []

    def chat(
        self, messages: list[dict[str, str]], temperature: float = 0.7
    ) -> str:
        self.chat_calls.append(messages)
        if not self.chat_responses:
            return '{"final":"默认回复"}'
        return self.chat_responses.pop(0)

    async def astream_chat(
        self, messages: list[dict[str, str]], temperature: float = 0.7
    ) -> AsyncIterator[str]:
        self.stream_calls.append(messages)
        for token in self.stream_tokens:
            yield token


@pytest.fixture
def member() -> MemberContext:
    """标准测试会员，模拟已登录 mall-portal 用户。"""
    return MemberContext(
        member_id=100,
        username="testuser",
        nickname="测试用户",
        token="mock-jwt-token",
    )


@pytest.fixture
def tool_ctx(member: MemberContext) -> ToolContext:
    """Tool 执行上下文，绑定测试会员与固定会话 ID。"""
    return ToolContext(
        token=member.token,
        member_id=member.member_id,
        member_username=member.username,
        conversation_id=1,
    )


@pytest.fixture
def mock_conversation() -> MockConversation:
    """返回 id=1 的模拟会话。"""
    return MockConversation()


@pytest.fixture
def mock_db_session(mocker, mock_conversation):
    """拦截 SessionLocal，避免测试写入真实数据库。"""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.get.return_value = mock_conversation
    session.add = MagicMock()
    session.commit = MagicMock()
    session.refresh = MagicMock(
        side_effect=lambda obj: setattr(obj, "id", mock_conversation.id)
        if not getattr(obj, "id", None)
        else None
    )
    session.scalars.return_value.all.return_value = []
    session.scalars.return_value.first.return_value = None

    mocker.patch("app.workflows.graph.SessionLocal", return_value=session)
    mocker.patch("app.agents.runner.SessionLocal", return_value=session)
    return session


@pytest.fixture
def mock_graph_deps(mocker, mock_conversation):
    """拦截编排层外部依赖：会话、记忆、RAG、消息持久化。"""
    mocker.patch(
        "app.workflows.graph.get_or_create_conversation",
        return_value=mock_conversation,
    )
    mocker.patch("app.workflows.graph.save_message")
    mocker.patch("app.workflows.graph.persist_tool_logs")
    mocker.patch("app.workflows.graph.extract_and_save_memory")
    mocker.patch("app.workflows.graph.load_short_term_messages", return_value=[])
    mocker.patch("app.workflows.graph.load_long_term_memory", return_value="暂无长期记忆。")
    mocker.patch("app.workflows.graph.format_rag_context", return_value="RAG 参考内容")


@pytest.fixture
def mock_glm_factory(mocker):
    """返回工厂函数，用于将 get_glm 替换为 MockGLMClient。"""

    def _factory(
        chat_responses: list[str] | None = None,
        stream_tokens: list[str] | None = None,
    ) -> MockGLMClient:
        client = MockGLMClient(chat_responses, stream_tokens)
        mocker.patch("app.workflows.graph.get_llm", return_value=client)
        return client

    return _factory
