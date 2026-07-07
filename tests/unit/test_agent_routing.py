"""Expert Agent 工具路由测试。

验证 graph.py 中 agent → tools/prompt 的映射逻辑，
确保各 Expert Agent 只获得其职责范围内的 Tool。
"""

from __future__ import annotations

import pytest

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


def _resolve_agent_tools(agent: str, all_tools: list):
    """复刻 graph.py 中的 agent → tools/prompt 路由逻辑，供测试断言。"""
    if agent == "product":
        return _filter_tools(all_tools, PRODUCT_TOOL_NAMES), PRODUCT_AGENT_PROMPT
    if agent == "order":
        return _filter_tools(all_tools, ORDER_TOOL_NAMES), ORDER_AGENT_PROMPT
    return _filter_tools(all_tools, AFTERSALE_TOOL_NAMES), AFTERSALE_AGENT_PROMPT


class TestAgentToolRouting:
    """各 Expert Agent 工具集与 Prompt 路由用例集。"""

    @pytest.fixture
    def all_tools(self):
        """构建全部 Tool，作为路由筛选的输入源。"""
        ctx = ToolContext(
            token="token",
            member_id=1,
            member_username="user",
            conversation_id=1,
        )
        return build_tools(ctx)

    @pytest.mark.parametrize(
        "agent,expected_names,prompt",
        [
            ("product", PRODUCT_TOOL_NAMES, PRODUCT_AGENT_PROMPT),
            ("order", ORDER_TOOL_NAMES, ORDER_AGENT_PROMPT),
            ("aftersale", AFTERSALE_TOOL_NAMES, AFTERSALE_AGENT_PROMPT),
        ],
    )
    def test_each_agent_gets_correct_tools(self, all_tools, agent, expected_names, prompt):
        """每个 Agent 应获得专属 Tool 子集及对应的系统 Prompt。"""
        tools, resolved_prompt = _resolve_agent_tools(agent, all_tools)
        tool_names = {t.name for t in tools}
        assert tool_names == expected_names
        assert resolved_prompt == prompt

    def test_agents_do_not_share_tools(self, all_tools):
        """三类 Agent 的 Tool 集合应互不重叠，防止越权调用。"""
        product_tools = {t.name for t in _resolve_agent_tools("product", all_tools)[0]}
        order_tools = {t.name for t in _resolve_agent_tools("order", all_tools)[0]}
        aftersale_tools = {
            t.name for t in _resolve_agent_tools("aftersale", all_tools)[0]
        }
        assert product_tools.isdisjoint(order_tools)
        assert order_tools.isdisjoint(aftersale_tools)
        assert product_tools.isdisjoint(aftersale_tools)
