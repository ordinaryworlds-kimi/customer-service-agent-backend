"""Supervisor 编排工作流集成测试。

验证 run_chat / stream_chat 能否根据 Supervisor 计划
正确调度 product、order、aftersale 三类 Expert Agent。
"""

from __future__ import annotations
import json
import pytest
from app.models.schemas import MemberContext
from app.workflows.graph import run_chat, stream_chat


def _supervisor_plan(tasks: list[dict], need_rag: bool = False) -> str:
    """构造 Supervisor LLM 返回的任务计划 JSON 字符串。"""
    return json.dumps({"tasks": tasks, "need_rag": need_rag}, ensure_ascii=False)


@pytest.fixture
def track_expert_agents(mocker):
    """拦截 run_expert_agent，记录每次调用的 agent_name 与 system_prompt。"""
    calls: list[dict] = []

    async def _fake_run_expert_agent(**kwargs):
        agent_name = kwargs["agent_name"]
        calls.append(kwargs)
        return f"{agent_name} 处理结果"

    mocker.patch(
        "app.workflows.graph.run_expert_agent",
        side_effect=_fake_run_expert_agent,
    )
    return calls


class TestRunChatAgentDispatch:
    """非流式 run_chat 的 Agent 调度用例集。"""

    @pytest.mark.asyncio
    async def test_dispatches_product_agent(
        self, member, mock_graph_deps, mock_glm_factory, track_expert_agents
    ):
        """Supervisor 计划仅含 product 任务时，应调度商品 Expert Agent 并汇总回复。"""
        mock_glm_factory(
            chat_responses=[
                _supervisor_plan(
                    [{"agent": "product", "instruction": "推荐一款手机"}]
                ),
                "为您推荐 iPhone 15。",
            ]
        )

        result = await run_chat(member, None, "有什么手机推荐？")

        assert result == "为您推荐 iPhone 15。"
        assert len(track_expert_agents) == 1
        assert track_expert_agents[0]["agent_name"] == "product"
        assert "商品 Expert Agent" in track_expert_agents[0]["system_prompt"]

    @pytest.mark.asyncio
    async def test_dispatches_order_agent(
        self, member, mock_graph_deps, mock_glm_factory, track_expert_agents
    ):
        """Supervisor 计划含 order 任务时，应调度订单 Expert Agent。"""
        mock_glm_factory(
            chat_responses=[
                _supervisor_plan([{"agent": "order", "instruction": "查最近订单"}]),
                "您有一笔待发货订单。",
            ]
        )

        result = await run_chat(member, None, "我的订单到哪了？")

        assert len(track_expert_agents) == 1
        assert track_expert_agents[0]["agent_name"] == "order"
        assert "订单 Expert Agent" in track_expert_agents[0]["system_prompt"]
        assert "order 处理结果" in result or "待发货" in result

    @pytest.mark.asyncio
    async def test_dispatches_aftersale_agent(
        self, member, mock_graph_deps, mock_glm_factory, track_expert_agents
    ):
        """Supervisor 计划含 aftersale 任务时，应调度售后 Expert Agent。"""
        mock_glm_factory(
            chat_responses=[
                _supervisor_plan(
                    [{"agent": "aftersale", "instruction": "申请退货退款"}]
                ),
                "已为您说明退货流程。",
            ]
        )

        await run_chat(member, None, "我要退货")

        assert len(track_expert_agents) == 1
        assert track_expert_agents[0]["agent_name"] == "aftersale"
        assert "售后 Expert Agent" in track_expert_agents[0]["system_prompt"]

    @pytest.mark.asyncio
    async def test_dispatches_all_three_agents_in_one_turn(
        self, member, mock_graph_deps, mock_glm_factory, track_expert_agents
    ):
        """多意图场景下 Supervisor 应依次调度 product → order → aftersale。"""
        mock_glm_factory(
            chat_responses=[
                _supervisor_plan(
                    [
                        {"agent": "product", "instruction": "查 iPhone 库存"},
                        {"agent": "order", "instruction": "查订单物流"},
                        {"agent": "aftersale", "instruction": "估算退款金额"},
                    ]
                ),
                "综合回复：库存充足，物流已发出，预计退款 200 元。",
            ]
        )

        await run_chat(
            member,
            None,
            "iPhone 有货吗？我的订单物流呢？退货能退多少？",
        )

        assert len(track_expert_agents) == 3
        agents = [c["agent_name"] for c in track_expert_agents]
        assert agents == ["product", "order", "aftersale"]

    @pytest.mark.asyncio
    async def test_unknown_agent_falls_back_to_aftersale(
        self, member, mock_graph_deps, mock_glm_factory, track_expert_agents
    ):
        """未知 agent 名称时，graph.py 的 else 分支应降级使用售后 Agent 的 Prompt 与 Tool。"""
        mock_glm_factory(
            chat_responses=[
                _supervisor_plan([{"agent": "unknown", "instruction": "处理请求"}]),
                "已处理。",
            ]
        )

        await run_chat(member, None, "帮我处理一下")

        assert track_expert_agents[0]["agent_name"] == "unknown"
        assert "售后 Expert Agent" in track_expert_agents[0]["system_prompt"]


class TestStreamChatAgentDispatch:
    """流式 stream_chat 的 Agent 调度与 SSE 事件用例集。"""

    @pytest.mark.asyncio
    async def test_stream_events_cover_all_agents(
        self, member, mock_graph_deps, mock_glm_factory, track_expert_agents
    ):
        """流式模式下应依次产出 meta/status/token/done 事件，并调度全部三类 Agent。"""
        mock_glm_factory(
            chat_responses=[
                _supervisor_plan(
                    [
                        {"agent": "product", "instruction": "查库存"},
                        {"agent": "order", "instruction": "查物流"},
                        {"agent": "aftersale", "instruction": "查售后"},
                    ]
                ),
            ],
            stream_tokens=["您好", "，", "已为您处理。"],
        )

        events = []
        async for event in stream_chat(member, None, "综合咨询"):
            events.append(event)

        event_types = [e["event"] for e in events]
        assert event_types[0] == "meta"
        assert event_types[-1] == "done"
        assert "token" in event_types

        status_messages = [
            e["data"]["message"]
            for e in events
            if e["event"] == "status" and "Agent" in e["data"].get("message", "")
        ]
        assert any("product Agent" in msg for msg in status_messages)
        assert any("order Agent" in msg for msg in status_messages)
        assert any("aftersale Agent" in msg for msg in status_messages)

        assert len(track_expert_agents) == 3
        assert events[-1]["data"]["content"] == "您好，已为您处理。"

    @pytest.mark.asyncio
    async def test_stream_single_product_agent(
        self, member, mock_graph_deps, mock_glm_factory, track_expert_agents
    ):
        """单 Agent 流式对话应只调度一次 product，并以 done 事件结束。"""
        mock_glm_factory(
            chat_responses=[
                _supervisor_plan([{"agent": "product", "instruction": "推荐商品"}]),
            ],
            stream_tokens=["推荐", "商品A"],
        )

        events = [e async for e in stream_chat(member, None, "推荐商品")]

        assert len(track_expert_agents) == 1
        assert track_expert_agents[0]["agent_name"] == "product"
        assert events[-1]["event"] == "done"
