"""Expert Agent ReAct 执行器，支持 Tool 调用与追踪。"""

import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from app.llm.provider import get_llm
from app.models.db import AgentTrace, SessionLocal
from app.tools.registry import ToolContext

logger = logging.getLogger(__name__)


def _tool_catalog(tools: list[StructuredTool]) -> str:
    """生成 Tool 名称与描述清单，供 LLM Prompt 使用。
    Args:
        tools: 当前 Agent 可用的 Tool 列表。
    Returns:
        str: 每行一个 Tool 的文本描述。
    """
    lines = []
    for t in tools:
        desc = t.description or ""
        args_schema = getattr(t, "args_schema", None)
        if args_schema:
            props = []
            schema = args_schema.model_json_schema()
            for name, meta in schema.get("properties", {}).items():
                props.append(f"{name}: {meta.get('description', '')}")
            param_hint = "; ".join(props)
            lines.append(f"- {t.name}({param_hint}): {desc}")
        else:
            lines.append(f"- {t.name}: {desc}")
    return "\n".join(lines)


def _parse_tool_call(text: str) -> dict[str, Any] | None:
    """从 LLM 输出中解析 JSON 格式的 tool 或 final 指令。
    Args:
        text: LLM 原始输出文本。
    Returns:
        dict[str, Any] | None: 解析成功且含 tool 键时返回 dict，否则 None。
    """
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        if "tool" in data or "final" in data:
            return data
    except json.JSONDecodeError:
        return None
    return None


async def _invoke_tool(
    tools: dict[str, StructuredTool], name: str, args: dict[str, Any]
) -> str:
    """调用指定 Tool 并返回 JSON 字符串结果。
    Args:
        tools: 工具名到 StructuredTool 的映射。
        name: 工具名称。
        args: 工具参数字典。
    Returns:
        str: Tool 返回结果的 JSON 字符串，未知工具时含 error 字段。
    """
    tool = tools.get(name)
    if not tool:
        logger.warning("[tool_invoke] unknown tool=%s args=%s", name, args)
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    logger.info("[tool_invoke] tool=%s args=%s", name, args)
    try:
        return await tool.ainvoke(args)
    except ValidationError as exc:
        error = json.dumps(
            {
                "error": "参数校验失败",
                "detail": exc.errors(include_url=False),
                "hint": "请检查参数类型与取值是否符合工具说明，修正后重试。",
            },
            ensure_ascii=False,
        )
        logger.warning("[tool_invoke] validation failed tool=%s args=%s error=%s", name, args, error)
        return error


def _trace(
    conversation_id: int | None,
    member_id: int,
    agent_name: str,
    step: str,
    inp: str,
    out: str,
    duration_ms: int,
) -> None:
    """写入 Agent 执行追踪记录。
    Args:
        conversation_id: 会话 ID，可为空。
        member_id: 会员 ID。
        agent_name: Agent 名称，如 product、order。
        step: 步骤名称，如 final、fallback。
        inp: 输入摘要。
        out: 输出摘要。
        duration_ms: 耗时（毫秒）。
    """
    with SessionLocal() as db:
        db.add(
            AgentTrace(
                conversation_id=conversation_id,
                member_id=member_id,
                agent_name=agent_name,
                step_name=step,
                input_summary=inp[:2000],
                output_summary=out[:2000],
                duration_ms=duration_ms,
            )
        )
        db.commit()


async def run_expert_agent(
    agent_name: str,
    system_prompt: str,
    instruction: str,
    tools: list[StructuredTool],
    rag_context: str,
    history: list[dict[str, str]],
    ctx: ToolContext,
    max_rounds: int = 4,
) -> str:
    """运行单个 Expert Agent，循环调用 Tool 直至产出最终回复。
    Args:
        agent_name: Agent 标识，用于日志与追踪。
        system_prompt: 系统 Prompt。
        instruction: 本子任务的具体指令。
        tools: 可用 Tool 列表。
        rag_context: RAG 检索到的参考知识文本。
        history: 会话历史消息。
        ctx: Tool 执行上下文（Token、会员信息等）。
        max_rounds: 最大 Tool 调用轮数。
    Returns:
        str: Agent 面向用户的最终中文回复。
    """
    llm = get_llm()
    tool_map = {t.name: t for t in tools}
    catalog = _tool_catalog(tools)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                f"{system_prompt}\n\n"
                f"可用工具：\n{catalog}\n\n"
                f'调用工具时只输出 JSON：{{"tool":"工具名","args":{{...}}}}\n'
                f'完成时只输出 JSON：{{"final":"你的中文回复"}}\n'
                f"RAG 参考：\n{rag_context}"
            ),
        },
        *history[-6:],
        {"role": "user", "content": instruction},
    ]
    start = time.perf_counter()
    logger.info(
        "[expert_agent] started agent=%s conversation_id=%s instruction=%r tools=%s",
        agent_name,
        ctx.conversation_id,
        instruction,
        list(tool_map.keys()),
    )
    for round_num in range(1, max_rounds + 1):
        raw = llm.chat(messages, temperature=0.2)
        parsed = _parse_tool_call(raw)
        logger.info(
            "[expert_round] agent=%s conversation_id=%s round=%s/%s raw=%r parsed=%s",
            agent_name,
            ctx.conversation_id,
            round_num,
            max_rounds,
            raw[:500],
            parsed,
        )
        if not parsed:
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": "请按 JSON 格式输出 tool 或 final。"}
            )
            continue
        if "final" in parsed:
            result = str(parsed["final"])
            duration_ms = int((time.perf_counter() - start) * 1000)
            logger.info(
                "[expert_final] agent=%s conversation_id=%s round=%s duration_ms=%s result_len=%s",
                agent_name,
                ctx.conversation_id,
                round_num,
                duration_ms,
                len(result),
            )
            _trace(
                ctx.conversation_id,
                ctx.member_id,
                agent_name,
                "final",
                instruction,
                result,
                duration_ms,
            )
            return result
        tool_name = parsed.get("tool")
        args = parsed.get("args") or {}
        tool_result = await _invoke_tool(tool_map, str(tool_name), args)
        logger.info(
            "[tool_result] agent=%s conversation_id=%s tool=%s result_len=%s",
            agent_name,
            ctx.conversation_id,
            tool_name,
            len(tool_result),
        )
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {"role": "user", "content": f"工具 {tool_name} 返回：{tool_result}"}
        )
    fallback = llm.chat(
        messages + [{"role": "user", "content": "请直接给出最终中文回复。"}],
        temperature=0.3,
    )
    duration_ms = int((time.perf_counter() - start) * 1000)
    logger.warning(
        "[expert_fallback] agent=%s conversation_id=%s duration_ms=%s max_rounds=%s",
        agent_name,
        ctx.conversation_id,
        duration_ms,
        max_rounds,
    )
    _trace(
        ctx.conversation_id,
        ctx.member_id,
        agent_name,
        "fallback",
        instruction,
        fallback,
        duration_ms,
    )
    return fallback
