"""Expert Agent ReAct 执行器测试。

覆盖 product / order / aftersale 三类 Agent 的
直接回复、Tool 调用链路与 max_rounds 降级逻辑。
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.tools import StructuredTool

from app.agents.runner import run_expert_agent
from app.prompts.templates import (
    AFTERSALE_AGENT_PROMPT,
    ORDER_AGENT_PROMPT,
    PRODUCT_AGENT_PROMPT,
)
from app.tools.registry import (
    AFTERSALE_TOOL_NAMES,
    ORDER_TOOL_NAMES,
    PRODUCT_TOOL_NAMES,
    ToolContext,
    build_tools,
)
from app.workflows.graph import _filter_tools
from tests.conftest import MockGLMClient


@pytest.fixture
def expert_ctx() -> ToolContext:
    """Expert Agent 执行所需的 Tool 上下文。"""
    return ToolContext(
        token="mock-token",
        member_id=100,
        member_username="testuser",
        conversation_id=1,
    )


@pytest.fixture
def mock_runner_db(mocker):
    """拦截 runner 中的 AgentTrace 写入，避免真实 DB 操作。"""
    session = MagicMock()
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)
    session.add = MagicMock()
    session.commit = MagicMock()
    mocker.patch("app.agents.runner.SessionLocal", return_value=session)
    return session


def _tools_for_agent(agent: str, ctx: ToolContext) -> list[StructuredTool]:
    """按 agent 名称筛选对应 Tool 子集。"""
    all_tools = build_tools(ctx)
    if agent == "product":
        return _filter_tools(all_tools, PRODUCT_TOOL_NAMES)
    if agent == "order":
        return _filter_tools(all_tools, ORDER_TOOL_NAMES)
    return _filter_tools(all_tools, AFTERSALE_TOOL_NAMES)


def _prompt_for_agent(agent: str) -> str:
    """返回各 Agent 对应的系统 Prompt。"""
    return {
        "product": PRODUCT_AGENT_PROMPT,
        "order": ORDER_AGENT_PROMPT,
        "aftersale": AFTERSALE_AGENT_PROMPT,
    }[agent]


@pytest.mark.parametrize("agent", ["product", "order", "aftersale"])
@pytest.mark.asyncio
async def test_expert_agent_direct_final_response(
    mocker, expert_ctx, mock_runner_db, agent
):
    """LLM 首轮直接返回 final 时，各 Expert Agent 应产出对应回复并注入正确 Prompt。"""
    expected = f"{agent} agent 回复内容"
    mock_glm = MockGLMClient(chat_responses=[json.dumps({"final": expected})])
    mocker.patch("app.agents.runner.get_llm", return_value=mock_glm)

    result = await run_expert_agent(
        agent_name=agent,
        system_prompt=_prompt_for_agent(agent),
        instruction=f"测试 {agent} 子任务",
        tools=_tools_for_agent(agent, expert_ctx),
        rag_context="RAG 上下文",
        history=[],
        ctx=expert_ctx,
    )

    assert result == expected
    assert mock_glm.chat_calls
    system_content = mock_glm.chat_calls[0][0]["content"]
    assert _prompt_for_agent(agent) in system_content


@pytest.mark.asyncio
async def test_product_agent_tool_then_final(mocker, expert_ctx, mock_runner_db):
    """Product Agent：先调用 query_product 工具，再根据返回结果生成 final 回复。"""
    tool_result = {"products": [{"name": "iPhone 15", "price": 5999}]}
    mock_tool = MagicMock(spec=StructuredTool)
    mock_tool.name = "query_product"
    mock_tool.description = "搜索商品"
    mock_tool.ainvoke = AsyncMock(return_value=json.dumps(tool_result))

    mock_glm = MockGLMClient(
        chat_responses=[
            json.dumps({"tool": "query_product", "args": {"keyword": "iPhone"}}),
            json.dumps({"final": "iPhone 15 售价 5999 元，库存充足。"}),
        ]
    )
    mocker.patch("app.agents.runner.get_llm", return_value=mock_glm)

    result = await run_expert_agent(
        agent_name="product",
        system_prompt=PRODUCT_AGENT_PROMPT,
        instruction="查询 iPhone 价格",
        tools=[mock_tool],
        rag_context="",
        history=[],
        ctx=expert_ctx,
    )

    assert result == "iPhone 15 售价 5999 元，库存充足。"
    mock_tool.ainvoke.assert_awaited_once_with({"keyword": "iPhone"})
    assert len(mock_glm.chat_calls) == 2


@pytest.mark.asyncio
async def test_order_agent_tool_then_final(mocker, expert_ctx, mock_runner_db):
    """Order Agent：调用 query_logistics 查询物流，并基于结果生成回复。"""
    mock_tool = MagicMock(spec=StructuredTool)
    mock_tool.name = "query_logistics"
    mock_tool.description = "查询物流"
    mock_tool.ainvoke = AsyncMock(
        return_value=json.dumps(
            {"order_id": 42, "delivery_company": "顺丰", "delivery_sn": "SF123"}
        )
    )

    mock_glm = MockGLMClient(
        chat_responses=[
            json.dumps({"tool": "query_logistics", "args": {"order_id": 42}}),
            json.dumps({"final": "您的订单由顺丰承运，单号 SF123。"}),
        ]
    )
    mocker.patch("app.agents.runner.get_llm", return_value=mock_glm)

    result = await run_expert_agent(
        agent_name="order",
        system_prompt=ORDER_AGENT_PROMPT,
        instruction="查询订单 42 物流",
        tools=[mock_tool],
        rag_context="",
        history=[],
        ctx=expert_ctx,
    )

    assert "顺丰" in result
    mock_tool.ainvoke.assert_awaited_once_with({"order_id": 42})


@pytest.mark.asyncio
async def test_aftersale_agent_tool_then_final(mocker, expert_ctx, mock_runner_db):
    """Aftersale Agent：调用 calculate_refund 估算退款金额。"""
    mock_tool = MagicMock(spec=StructuredTool)
    mock_tool.name = "calculate_refund"
    mock_tool.description = "估算退款"
    mock_tool.ainvoke = AsyncMock(
        return_value=json.dumps(
            {"product_real_price": 100.0, "product_count": 2, "refund_amount": 200.0}
        )
    )

    mock_glm = MockGLMClient(
        chat_responses=[
            json.dumps(
                {"tool": "calculate_refund", "args": {"product_real_price": 100, "product_count": 2}}
            ),
            json.dumps({"final": "预计退款 200 元，最终以审核为准。"}),
        ]
    )
    mocker.patch("app.agents.runner.get_llm", return_value=mock_glm)

    result = await run_expert_agent(
        agent_name="aftersale",
        system_prompt=AFTERSALE_AGENT_PROMPT,
        instruction="估算退货 2 件商品的退款",
        tools=[mock_tool],
        rag_context="",
        history=[],
        ctx=expert_ctx,
    )

    assert "200" in result
    mock_tool.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_expert_agent_fallback_after_max_rounds(
    mocker, expert_ctx, mock_runner_db
):
    """连续 max_rounds 轮无法解析有效 JSON 时，应触发 fallback 直接生成最终回复。"""
    mock_glm = MockGLMClient(
        chat_responses=[
            "invalid response",
            "still invalid",
            "still invalid",
            "still invalid",
            "fallback 最终回复",
        ]
    )
    mocker.patch("app.agents.runner.get_llm", return_value=mock_glm)

    result = await run_expert_agent(
        agent_name="product",
        system_prompt=PRODUCT_AGENT_PROMPT,
        instruction="测试 fallback",
        tools=[],
        rag_context="",
        history=[],
        ctx=expert_ctx,
        max_rounds=4,
    )

    assert result == "fallback 最终回复"
    assert len(mock_glm.chat_calls) == 5
