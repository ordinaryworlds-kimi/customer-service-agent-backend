"""JWT 鉴权模块，与 mall-portal Hutool HS512 Token 兼容。"""

import logging
import time
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request
from sqlalchemy import text

from app.config.settings import get_settings
from app.models.db import MallSessionLocal
from app.models.schemas import MemberContext

settings = get_settings()
logger = logging.getLogger(__name__)


def _extract_token(request: Request) -> str:
    """从 HTTP 请求头中提取 JWT Token。
    Args:
        request: FastAPI 请求对象。
    Returns:
        str: 不含 Bearer 前缀的 Token 字符串。
    Raises:
        HTTPException: 未携带 Authorization 头时返回 401。
    """
    auth = request.headers.get("Authorization", "")
    prefix = settings.jwt_bearer_prefix
    if auth.startswith(prefix):
        return auth[len(prefix) :].strip()
    if auth:
        return auth.strip()
    raise HTTPException(status_code=401, detail="未登录，请先登录")


def decode_username(token: str) -> str:
    """解析 JWT 并提取用户名。
    Args:
        token: mall-portal 签发的 JWT（不含 Bearer 前缀）。
    Returns:
        str: 用户名（JWT sub 字段）。
    Raises:
        HTTPException: Token 无效、过期或缺少 sub 时返回 401。
    """
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS512", "HS256"],
            options={"verify_exp": False},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Token 无效或已过期") from exc
    exp = payload.get("exp")
    if exp is not None:
        exp_val = int(exp)
        # mall-portal Hutool JWT 使用毫秒时间戳
        now_ms = int(time.time() * 1000)
        if exp_val > 10**11 and exp_val < now_ms:
            raise HTTPException(status_code=401, detail="Token 已过期")
        if exp_val <= 10**11 and exp_val < time.time():
            raise HTTPException(status_code=401, detail="Token 已过期")
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Token 缺少用户信息")
    return str(username)


def get_member_by_username(username: str) -> dict[str, Any]:
    """从 mall 库查询会员信息。
    Args:
        username: 会员用户名。
    Returns:
        dict[str, Any]: 包含 id、username、nickname、phone 等字段。
    Raises:
        HTTPException: 用户不存在或已禁用时返回 401。
    """
    with MallSessionLocal() as session:
        row = (
            session.execute(
                text(
                    "SELECT id, username, nickname, phone FROM ums_member "
                    "WHERE username = :username AND status = 1 LIMIT 1"
                ),
                {"username": username},
            )
            .mappings()
            .first()
        )
    if not row:
        raise HTTPException(status_code=401, detail="用户不存在或已禁用")
    return dict(row)


async def get_current_member(request: Request) -> MemberContext:
    """FastAPI 依赖：解析当前登录会员上下文。
    Args:
        request: FastAPI 请求对象。
    Returns:
        MemberContext: 当前会员 ID、用户名、昵称及原始 Token。
    """
    token = _extract_token(request)
    username = decode_username(token)
    member = get_member_by_username(username)
    logger.info(
        "[auth] member authenticated member_id=%s username=%s",
        member["id"],
        username,
    )
    return MemberContext(
        member_id=int(member["id"]),
        username=str(member["username"]),
        nickname=member.get("nickname"),
        token=token,
    )


CurrentMember = Depends(get_current_member)
