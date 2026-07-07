"""Supervisor 任务计划解析测试。

验证 graph._parse_supervisor_plan 能否正确解析
Supervisor LLM 输出的 JSON 任务计划，并在异常时安全降级。
"""

import json

import pytest

from app.workflows.graph import _parse_supervisor_plan


class TestParseSupervisorPlan:
    """Supervisor 计划 JSON 解析用例集。"""

    def test_valid_json(self):
        """合法 JSON 应原样解析出 agent 与 need_rag 字段。"""
        plan = {
            "tasks": [{"agent": "order", "instruction": "查询最近订单"}],
            "need_rag": False,
        }
        raw = json.dumps(plan, ensure_ascii=False)
        result = _parse_supervisor_plan(raw)
        assert result["tasks"][0]["agent"] == "order"
        assert result["need_rag"] is False

    def test_json_wrapped_in_markdown(self):
        """LLM 用 markdown 代码块包裹 JSON 时，应能提取内部 JSON 并解析。"""
        raw = '```json\n{"tasks":[{"agent":"aftersale","instruction":"申请退货"}],"need_rag":true}\n```'
        result = _parse_supervisor_plan(raw)
        assert result["tasks"][0]["agent"] == "aftersale"

    def test_invalid_json_fallback_to_product(self):
        """无法解析时降级为单 product 任务，避免编排流程中断。"""
        raw = "这不是合法 JSON"
        result = _parse_supervisor_plan(raw)
        assert result["tasks"][0]["agent"] == "product"
        assert result["tasks"][0]["instruction"] == raw
        assert result["need_rag"] is True

    def test_multi_agent_plan(self):
        """多任务计划应保留 product / order / aftersale 的调度顺序。"""
        plan = {
            "tasks": [
                {"agent": "product", "instruction": "查 iPhone 库存"},
                {"agent": "order", "instruction": "查物流"},
                {"agent": "aftersale", "instruction": "估算退款"},
            ],
            "need_rag": True,
        }
        result = _parse_supervisor_plan(json.dumps(plan))
        agents = [t["agent"] for t in result["tasks"]]
        assert agents == ["product", "order", "aftersale"]
