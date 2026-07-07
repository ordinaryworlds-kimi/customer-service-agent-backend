"""长期记忆与短期会话记忆管理。"""

import json
from typing import Any
from sqlalchemy import select
from app.llm.provider import get_llm
from app.models.db import Memory, Message, SessionLocal


def load_long_term_memory(member_id: int) -> str:
    """加载用户长期记忆，格式化为 Prompt 文本。
    Args:
        member_id: mall 会员 ID。
    Returns:
        str: 多行文本，无记忆时返回「暂无长期记忆。」
    """
    with SessionLocal() as db:
        rows = db.scalars(
            select(Memory)
            .where(Memory.member_id == member_id)
            .order_by(Memory.updated_at.desc())
            .limit(20)
        ).all()
    if not rows:
        return "暂无长期记忆。"
    lines = [f"- [{m.memory_type}] {m.memory_key}: {m.memory_value}" for m in rows]
    return "\n".join(lines)


def upsert_memory(
    member_id: int, memory_type: str, key: str, value: str, source: str = "agent"
) -> None:
    """新增或更新一条长期记忆。
    Args:
        member_id: mall 会员 ID。
        memory_type: 记忆类型，如 preference、address、product。
        key: 记忆键，同一用户下与 type 组合唯一。
        value: 记忆内容。
        source: 来源标识，默认 agent。
    """
    with SessionLocal() as db:
        existing = db.scalars(
            select(Memory).where(
                Memory.member_id == member_id,
                Memory.memory_type == memory_type,
                Memory.memory_key == key,
            )
        ).first()
        if existing:
            existing.memory_value = value
            existing.source = source
        else:
            db.add(
                Memory(
                    member_id=member_id,
                    memory_type=memory_type,
                    memory_key=key,
                    memory_value=value,
                    source=source,
                )
            )
        db.commit()


def load_short_term_messages(
    conversation_id: int, limit: int = 20
) -> list[dict[str, str]]:
    """加载会话最近若干轮消息，供 LLM 上下文使用。
    Args:
        conversation_id: 会话 ID。
        limit: 最多加载的消息条数。
    Returns:
        list[dict[str, str]]: 按时间正序的消息列表，每项含 role 与 content。
    """
    with SessionLocal() as db:
        rows = db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        ).all()
    rows = list(reversed(rows))
    return [
        {
            "role": r.role if r.role in ("user", "assistant") else "assistant",
            "content": r.content,
        }
        for r in rows
    ]


def extract_and_save_memory(
    member_id: int, user_message: str, assistant_reply: str
) -> None:
    """使用 LLM 从本轮对话中提取值得长期保存的用户信息。
    Args:
        member_id: mall 会员 ID。
        user_message: 用户本轮输入。
        assistant_reply: 客服本轮回复。
    Note:
        解析失败或无有效信息时静默跳过，不抛出异常。
    """
    llm = get_llm()
    prompt = f"""从以下对话中提取值得长期记住的用户信息（偏好、常购品类、收货习惯等）。
若无有价值信息，返回空 JSON 数组 []。
只返回 JSON，格式：[{{"type":"preference|address|product|habit|other","key":"...","value":"..."}}]
用户：{user_message}
客服：{assistant_reply}
"""
    try:
        raw = llm.chat([{"role": "user", "content": prompt}], temperature=0.1)
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start < 0 or end <= start:
            return
        items: list[dict[str, Any]] = json.loads(raw[start:end])
        for item in items:
            upsert_memory(
                member_id,
                str(item.get("type", "other")),
                str(item.get("key", "info"))[:100],
                str(item.get("value", ""))[:2000],
            )
    except Exception:
        pass
