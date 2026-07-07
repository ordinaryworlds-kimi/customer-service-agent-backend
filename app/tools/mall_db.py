"""mall 业务库直连查询（售后等）。"""

from typing import Any

from sqlalchemy import text

from app.models.db import MallSessionLocal

# 退货申请状态码与中文描述映射
RETURN_STATUS_MAP = {0: "待处理", 1: "退货中", 2: "已完成", 3: "已拒绝"}


def query_return_applies(
    member_username: str, order_id: int | None = None
) -> list[dict[str, Any]]:
    """查询会员的退货申请列表。
    Args:
        member_username: 会员用户名，与 oms_order_return_apply.member_username 对应。
        order_id: 可选，按订单 ID 过滤。
    Returns:
        list[dict[str, Any]]: 退货申请记录列表，最多 20 条，按申请时间倒序。
    """
    sql = """
        SELECT id, order_id, order_sn, product_id, product_name, return_amount,
               status, reason, description, create_time, handle_time, handle_note
        FROM oms_order_return_apply
        WHERE member_username = :username
    """
    params: dict[str, Any] = {"username": member_username}
    if order_id is not None:
        sql += " AND order_id = :order_id"
        params["order_id"] = order_id
    sql += " ORDER BY create_time DESC LIMIT 20"
    with MallSessionLocal() as session:
        rows = session.execute(text(sql), params).mappings().all()
    return [dict(r) for r in rows]


def query_return_apply_by_id(
    apply_id: int, member_username: str
) -> dict[str, Any] | None:
    """按 ID 查询单条退货申请（校验归属）。
    Args:
        apply_id: 退货申请 ID。
        member_username: 会员用户名，用于校验数据归属。
    Returns:
        dict[str, Any] | None: 退货申请详情，不存在或不属于该用户时返回 None。
    """
    with MallSessionLocal() as session:
        row = (
            session.execute(
                text("""
                SELECT id, order_id, order_sn, product_id, product_name, return_amount,
                       status, reason, description, create_time, handle_time, handle_note
                FROM oms_order_return_apply
                WHERE id = :id AND member_username = :username
                LIMIT 1
                """),
                {"id": apply_id, "username": member_username},
            )
            .mappings()
            .first()
        )
    return dict(row) if row else None
