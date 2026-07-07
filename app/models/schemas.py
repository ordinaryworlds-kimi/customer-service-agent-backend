"""Pydantic 请求/响应模型定义。"""

from datetime import datetime
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天请求体。"""

    conversation_id: int | None = Field(
        default=None, description="会话 ID，为空则创建新会话"
    )
    message: str = Field(..., min_length=1, max_length=4000, description="用户消息内容")


class ConversationOut(BaseModel):
    """会话列表项响应。"""

    id: int = Field(description="会话 ID")
    title: str | None = Field(default=None, description="会话标题")
    status: int = Field(description="状态：0 关闭，1 进行中")
    created_at: datetime = Field(description="创建时间")
    updated_at: datetime = Field(description="更新时间")
    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    """消息列表项响应。"""

    id: int = Field(description="消息 ID")
    role: str = Field(description="角色：user/assistant/system/tool")
    content: str = Field(description="消息正文")
    agent_name: str | None = Field(default=None, description="处理的 Agent 名称")
    created_at: datetime = Field(description="创建时间")
    model_config = {"from_attributes": True}


class MemberContext(BaseModel):
    """当前登录会员上下文，供 Agent 与 Tool 使用。"""

    member_id: int = Field(description="mall 会员 ID")
    username: str = Field(description="会员用户名")
    nickname: str | None = Field(default=None, description="会员昵称")
    token: str = Field(description="mall-portal JWT Token")
