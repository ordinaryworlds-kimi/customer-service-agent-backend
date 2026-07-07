"""Supervisor 多 Agent 编排工作流。"""

import json
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.tools import StructuredTool

from app.agents.runner import run_expert_agent
from app.llm.provider import get_llm
from app.memory.store import (
    extract_and_save_memory,
    load_long_term_memory,
    load_short_term_messages,
)
from app.models.db import AgentTrace, Conversation, Message, SessionLocal, ToolLog
from app.models.schemas import MemberContext
from app.prompts.templates import (
    AFTERSALE_AGENT_PROMPT,
    ORDER_AGENT_PROMPT,
    PRODUCT_AGENT_PROMPT,
    SUMMARY_PROMPT,
    SUPERVISOR_PROMPT,
)
from app.rag.milvus_store import format_rag_context
from app.tools.registry import (
    AFTERSALE_TOOL_NAMES,
    ORDER_TOOL_NAMES,
    PRODUCT_TOOL_NAMES,
    ToolContext,
    build_tools,
)

logger = logging.getLogger(__name__)


def _parse_supervisor_plan(text: str) -> dict[str, Any]:
    """解析 Supervisor LLM 输出的任务计划 JSON。
    Args:
        text: Supervisor 原始输出，应含 tasks 与 need_rag 字段。
    Returns:
        dict[str, Any]: 任务计划；解析失败时降级为单 product 任务。
    """
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning(
                "[supervisor_plan] JSON parse failed, fallback to product agent raw=%r",
                text[:500],
            )
            pass
    return {"tasks": [{"agent": "product", "instruction": text}], "need_rag": True}


def _should_use_rag(plan: dict[str, Any]) -> bool:
    """判断是否需要向量检索：仅 product 类任务才启用 RAG。"""
    if not plan.get("need_rag", True):
        return False
    tasks = plan.get("tasks") or []
    return any(task.get("agent", "product") == "product" for task in tasks)


def _resolve_rag_context(user_message: str, plan: dict[str, Any]) -> str:
    """按任务类型决定是否检索 Milvus，不可用时由 format_rag_context 降级。"""
    if not _should_use_rag(plan):
        if plan.get("need_rag"):
            logger.info("[rag] skipped for non-product tasks")
        return "未启用 RAG。"
    return format_rag_context(user_message)


def _filter_tools(
    all_tools: list[StructuredTool], names: set[str]
) -> list[StructuredTool]:
    """按名称集合筛选 Tool 子集。
    Args:
        all_tools: 全部已注册 Tool。
        names: 允许的工具名集合。
    Returns:
        list[StructuredTool]: 筛选后的 Tool 列表。
    """
    return [t for t in all_tools if t.name in names]


def get_or_create_conversation(
    member: MemberContext, conversation_id: int | None, first_message: str
) -> Conversation:
    """获取已有会话或创建新会话。
    Args:
        member: 当前会员上下文。
        conversation_id: 会话 ID，为空则新建。
        first_message: 首条用户消息，用于生成会话标题。
    Returns:
        Conversation: 会话 ORM 对象。
    """
    with SessionLocal() as db:
        if conversation_id:
            conv = db.get(Conversation, conversation_id)
            if conv and conv.member_id == member.member_id:
                return conv
        title = first_message[:30] + ("..." if len(first_message) > 30 else "")
        conv = Conversation(
            member_id=member.member_id, member_username=member.username, title=title
        )
        db.add(conv)
        db.commit()
        db.refresh(conv)
        return conv


def save_message(
    conversation_id: int, role: str, content: str, agent_name: str | None = None
) -> None:
    """持久化一条会话消息。
    Args:
        conversation_id: 会话 ID。
        role: 消息角色，如 user、assistant。
        content: 消息正文。
        agent_name: 处理的 Agent 名称，可选。
    """
    with SessionLocal() as db:
        db.add(
            Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                agent_name=agent_name,
            )
        )
        db.commit()


def persist_tool_logs(ctx: ToolContext) -> None:
    """将 ToolContext 中累积的 Tool 调用日志写入数据库。
    Args:
        ctx: Tool 执行上下文，含 tool_logs 列表。
    """
    if not ctx.tool_logs:
        return
    with SessionLocal() as db:
        for log in ctx.tool_logs:
            db.add(
                ToolLog(
                    conversation_id=ctx.conversation_id,
                    member_id=ctx.member_id,
                    tool_name=log["tool_name"],
                    tool_input=log.get("tool_input"),
                    tool_output=log.get("tool_output"),
                    success=log.get("success", 1),
                    duration_ms=log.get("duration_ms"),
                )
            )
        db.commit()


async def run_chat(
    member: MemberContext, conversation_id: int | None, user_message: str
) -> str:
    """非流式执行完整客服对话流程。
    流程：Supervisor 拆任务 → 各 Expert Agent 执行 → Supervisor 汇总回复。
    Args:
        member: 当前会员上下文。
        conversation_id: 会话 ID，可为空。
        user_message: 用户输入。
    Returns:
        str: 最终客服回复文本。
    """
    conv = get_or_create_conversation(member, conversation_id, user_message)
    save_message(conv.id, "user", user_message)
    ctx = ToolContext(
        token=member.token,
        member_id=member.member_id,
        member_username=member.username,
        conversation_id=conv.id,
    )
    all_tools = build_tools(ctx)
    history = load_short_term_messages(conv.id, limit=20)
    long_memory = load_long_term_memory(member.member_id)
    llm = get_llm()
    sup_start = time.perf_counter()
    plan_raw = llm.chat(
        [
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {
                "role": "user",
                "content": f"用户长期记忆：\n{long_memory}\n\n用户消息：{user_message}",
            },
        ],
        temperature=0.1,
    )
    plan = _parse_supervisor_plan(plan_raw)
    rag_context = _resolve_rag_context(user_message, plan)
    with SessionLocal() as db:
        db.add(
            AgentTrace(
                conversation_id=conv.id,
                member_id=member.member_id,
                agent_name="supervisor",
                step_name="plan",
                input_summary=user_message[:2000],
                output_summary=plan_raw[:2000],
                duration_ms=int((time.perf_counter() - sup_start) * 1000),
            )
        )
        db.commit()
    agent_outputs: list[str] = []
    for task in plan.get("tasks", []):
        agent = task.get("agent", "product")
        instruction = task.get("instruction") or user_message
        if agent == "product":
            tools = _filter_tools(all_tools, PRODUCT_TOOL_NAMES)
            prompt = PRODUCT_AGENT_PROMPT
        elif agent == "order":
            tools = _filter_tools(all_tools, ORDER_TOOL_NAMES)
            prompt = ORDER_AGENT_PROMPT
        else:
            tools = _filter_tools(all_tools, AFTERSALE_TOOL_NAMES)
            prompt = AFTERSALE_AGENT_PROMPT
        result = await run_expert_agent(
            agent_name=agent,
            system_prompt=prompt,
            instruction=instruction,
            tools=tools,
            rag_context=rag_context,
            history=history,
            ctx=ctx,
        )
        agent_outputs.append(f"[{agent}] {result}")
    summary_input = "\n\n".join(agent_outputs) if agent_outputs else "无 Agent 结果"
    final = llm.chat(
        [
            {"role": "system", "content": SUMMARY_PROMPT},
            {
                "role": "user",
                "content": f"用户问题：{user_message}\n\n各 Agent 结果：\n{summary_input}",
            },
        ],
        temperature=0.4,
    )
    save_message(conv.id, "assistant", final, agent_name="supervisor")
    persist_tool_logs(ctx)
    extract_and_save_memory(member.member_id, user_message, final)
    return final


async def stream_chat(
    member: MemberContext, conversation_id: int | None, user_message: str
) -> AsyncIterator[dict[str, Any]]:
    """流式执行客服对话，逐步 yield SSE 事件。
    Args:
        member: 当前会员上下文。
        conversation_id: 会话 ID，可为空。
        user_message: 用户输入。
    Yields:
        dict[str, Any]: 事件字典，含 event（meta/status/token/done）与 data 字段。
    """
    conv = get_or_create_conversation(member, conversation_id, user_message)
    logger.info(
        "[stream_chat] started member_id=%s conversation_id=%s message=%r",
        member.member_id,
        conv.id,
        user_message,
    )
    yield {"event": "meta", "data": {"conversation_id": conv.id}}
    yield {"event": "status", "data": {"message": "Supervisor 正在分析意图..."}}
    ctx = ToolContext(
        token=member.token,
        member_id=member.member_id,
        member_username=member.username,
        conversation_id=conv.id,
    )
    save_message(conv.id, "user", user_message)
    all_tools = build_tools(ctx)
    history = load_short_term_messages(conv.id, limit=20)
    long_memory = load_long_term_memory(member.member_id)
    llm = get_llm()
    sup_start = time.perf_counter()
    plan_raw = llm.chat(
        [
            {"role": "system", "content": SUPERVISOR_PROMPT},
            {
                "role": "user",
                "content": f"用户长期记忆：\n{long_memory}\n\n用户消息：{user_message}",
            },
        ],
        temperature=0.1,
    )
    plan = _parse_supervisor_plan(plan_raw)
    logger.info(
        "[supervisor_plan] parsed member_id=%s conversation_id=%s need_rag=%s tasks=%s raw=%r",
        member.member_id,
        conv.id,
        plan.get("need_rag"),
        plan.get("tasks"),
        plan_raw[:500],
    )
    with SessionLocal() as db:
        db.add(
            AgentTrace(
                conversation_id=conv.id,
                member_id=member.member_id,
                agent_name="supervisor",
                step_name="plan",
                input_summary=user_message[:2000],
                output_summary=plan_raw[:2000],
                duration_ms=int((time.perf_counter() - sup_start) * 1000),
            )
        )
        db.commit()
    rag_context = _resolve_rag_context(user_message, plan)
    agent_outputs: list[str] = []
    tasks = plan.get("tasks") or [{"agent": "product", "instruction": user_message}]
    for i, task in enumerate(tasks, 1):
        agent = task.get("agent", "product")
        instruction = task.get("instruction") or user_message
        logger.info(
            "[dispatch_expert] member_id=%s conversation_id=%s task=%s/%s agent=%s instruction=%r",
            member.member_id,
            conv.id,
            i,
            len(tasks),
            agent,
            instruction,
        )
        yield {
            "event": "status",
            "data": {"message": f"正在处理子任务 {i}/{len(tasks)}：{agent} Agent"},
        }
        if agent == "product":
            tools = _filter_tools(all_tools, PRODUCT_TOOL_NAMES)
            prompt = PRODUCT_AGENT_PROMPT
        elif agent == "order":
            tools = _filter_tools(all_tools, ORDER_TOOL_NAMES)
            prompt = ORDER_AGENT_PROMPT
        else:
            tools = _filter_tools(all_tools, AFTERSALE_TOOL_NAMES)
            prompt = AFTERSALE_AGENT_PROMPT
        result = await run_expert_agent(
            agent_name=agent,
            system_prompt=prompt,
            instruction=instruction,
            tools=tools,
            rag_context=rag_context,
            history=history,
            ctx=ctx,
        )
        agent_outputs.append(f"[{agent}] {result}")
        logger.info(
            "[expert_done] member_id=%s conversation_id=%s agent=%s result_len=%s",
            member.member_id,
            conv.id,
            agent,
            len(result),
        )
    yield {"event": "status", "data": {"message": "正在汇总回复..."}}
    logger.info(
        "[summary_stream] started member_id=%s conversation_id=%s",
        member.member_id,
        conv.id,
    )
    summary_input = "\n\n".join(agent_outputs)
    messages = [
        {"role": "system", "content": SUMMARY_PROMPT},
        {
            "role": "user",
            "content": f"用户问题：{user_message}\n\n各 Agent 结果：\n{summary_input}",
        },
    ]
    final_parts: list[str] = []
    async for token in llm.astream_chat(messages, temperature=0.4):
        final_parts.append(token)
        yield {"event": "token", "data": {"content": token}}
    final = "".join(final_parts)
    save_message(conv.id, "assistant", final, agent_name="supervisor")
    persist_tool_logs(ctx)
    extract_and_save_memory(member.member_id, user_message, final)
    logger.info(
        "[stream_chat] completed member_id=%s conversation_id=%s reply_len=%s",
        member.member_id,
        conv.id,
        len(final),
    )
    yield {"event": "done", "data": {"conversation_id": conv.id, "content": final}}
