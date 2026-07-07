"""聊天与会话相关 API 路由。"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from app.auth.jwt import get_current_member
from app.models.db import Conversation, Message, SessionLocal
from app.models.schemas import ChatRequest, ConversationOut, MemberContext, MessageOut
from app.rag.milvus_store import sync_knowledge
from app.workflows.graph import run_chat, stream_chat

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat")
async def chat(
    body: ChatRequest, member: MemberContext = Depends(get_current_member)
) -> dict:
    """非流式聊天接口，等待完整回复后返回。
    Args:
        body: 聊天请求，含 conversation_id 与 message。
        member: 当前登录会员（依赖注入）。
    Returns:
        dict: 含 conversation_id 与 reply 字段。
    Raises:
        HTTPException: Agent 执行异常时返回 500。
    """
    try:
        reply = await run_chat(member, body.conversation_id, body.message)
        conv_id = body.conversation_id
        if not conv_id:
            with SessionLocal() as db:
                conv = db.scalars(
                    select(Conversation)
                    .where(Conversation.member_id == member.member_id)
                    .order_by(Conversation.updated_at.desc())
                ).first()
                conv_id = conv.id if conv else None
        return {"conversation_id": conv_id, "reply": reply}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/chat/stream")
async def chat_stream(
    body: ChatRequest, member: MemberContext = Depends(get_current_member)
) -> EventSourceResponse:
    """SSE 流式聊天接口，逐 token 推送回复。
    Args:
        body: 聊天请求。
        member: 当前登录会员。
    Returns:
        EventSourceResponse: SSE 事件流，事件类型含 meta、status、token、done、error。
    """

    logger.info(
        "[chat_stream] request started member_id=%s conversation_id=%s message=%r",
        member.member_id,
        body.conversation_id,
        body.message,
    )

    async def event_generator():
        """生成 SSE 事件序列。"""
        try:
            async for event in stream_chat(member, body.conversation_id, body.message):
                if event["event"] == "done":
                    logger.info(
                        "[chat_stream] request completed member_id=%s conversation_id=%s",
                        member.member_id,
                        event["data"].get("conversation_id"),
                    )
                yield {
                    "event": event["event"],
                    "data": json.dumps(event["data"], ensure_ascii=False),
                }
        except Exception as exc:
            logger.exception(
                "[chat_stream] request failed member_id=%s conversation_id=%s",
                member.member_id,
                body.conversation_id,
            )
            yield {
                "event": "error",
                "data": json.dumps({"message": str(exc)}, ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(
    member: MemberContext = Depends(get_current_member),
) -> list[Conversation]:
    """获取当前用户的会话列表。
    Args:
        member: 当前登录会员。
    Returns:
        list[Conversation]: 最近 50 条会话，按更新时间倒序。
    """
    with SessionLocal() as db:
        rows = db.scalars(
            select(Conversation)
            .where(Conversation.member_id == member.member_id)
            .order_by(Conversation.updated_at.desc())
            .limit(50)
        ).all()
    return rows


@router.get(
    "/conversations/{conversation_id}/messages", response_model=list[MessageOut]
)
def list_messages(
    conversation_id: int, member: MemberContext = Depends(get_current_member)
) -> list[Message]:
    """获取指定会话的消息历史。
    Args:
        conversation_id: 会话 ID。
        member: 当前登录会员。
    Returns:
        list[Message]: 消息列表，按时间正序。
    Raises:
        HTTPException: 会话不存在或不属于当前用户时返回 404。
    """
    with SessionLocal() as db:
        conv = db.get(Conversation, conversation_id)
        if not conv or conv.member_id != member.member_id:
            raise HTTPException(status_code=404, detail="会话不存在")
        rows = db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        ).all()
    return rows


@router.post("/rag/sync")
def rag_sync(_: MemberContext = Depends(get_current_member)) -> dict:
    """从 mall 库同步商品与帮助文档到 Milvus 向量库。
    Args:
        _: 当前登录会员（仅用于鉴权）。
    Returns:
        dict: 含 message 与 indexed（索引条数）字段。
    Raises:
        HTTPException: Milvus 或数据库异常时返回 500。
    """
    try:
        result = sync_knowledge()
        return {"message": "同步完成", **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG 同步失败: {exc}") from exc
