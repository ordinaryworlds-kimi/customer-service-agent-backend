"""Expert Agent Tool 调用 JSON 解析测试。

验证 runner._parse_tool_call 能否从 LLM 输出中
提取 {"tool":...} 或 {"final":...} 指令。
"""

import json

import pytest

from app.agents.runner import _parse_tool_call


class TestParseToolCall:
    """ReAct 循环中 Tool/Final JSON 解析用例集。"""

    def test_parse_tool_call(self):
        """标准 tool 调用 JSON 应完整解析出工具名与参数。"""
        raw = '{"tool":"query_product","args":{"keyword":"手机"}}'
        result = _parse_tool_call(raw)
        assert result == {"tool": "query_product", "args": {"keyword": "手机"}}

    def test_parse_final_response(self):
        """final 字段表示 Agent 已完成，应直接解析为最终回复。"""
        raw = '{"final":"该商品库存充足，推荐购买。"}'
        result = _parse_tool_call(raw)
        assert result == {"final": "该商品库存充足，推荐购买。"}

    def test_json_embedded_in_text(self):
        """LLM 在 JSON 前后附加说明文字时，应通过正则提取首个 JSON 对象。"""
        raw = '分析完毕，调用工具：\n{"tool":"query_order","args":{"status":-1}}'
        result = _parse_tool_call(raw)
        assert result["tool"] == "query_order"

    def test_invalid_json_returns_none(self):
        """纯文本无 JSON 时返回 None，触发 ReAct 重试提示。"""
        assert _parse_tool_call("plain text without json") is None

    def test_json_without_tool_or_final_returns_none(self):
        """JSON 缺少 tool/final 键时视为无效，不进入工具调用分支。"""
        assert _parse_tool_call('{"message":"hello"}') is None
