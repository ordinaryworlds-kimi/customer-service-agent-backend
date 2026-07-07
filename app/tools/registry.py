"""Tool 注册与执行上下文，封装 LangChain StructuredTool。"""

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.tools.mall_client import MallPortalClient
from app.tools.mall_db import RETURN_STATUS_MAP, query_return_applies
from app.tools.utils import to_camel_dict, to_json_safe

logger = logging.getLogger(__name__)


@dataclass
class ToolContext:
    """Tool 执行上下文，贯穿单次对话的所有 Tool 调用。
    Attributes:
        token: mall-portal JWT Token。
        member_id: 会员 ID。
        member_username: 会员用户名。
        conversation_id: 当前会话 ID，可为空。
        tool_logs: 本次对话累积的 Tool 调用日志。
    """

    token: str
    member_id: int
    member_username: str
    conversation_id: int | None = None
    tool_logs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def client(self) -> MallPortalClient:
        """获取绑定当前 Token 的 mall-portal HTTP 客户端。"""
        return MallPortalClient(self.token)

    def log_tool(
        self, name: str, inp: dict, output: Any, success: bool, duration_ms: int
    ) -> None:
        """记录一次 Tool 调用到内存日志列表。
        Args:
            name: Tool 名称。
            inp: 输入参数。
            output: 返回结果。
            success: 是否成功。
            duration_ms: 耗时（毫秒）。
        """
        raw_output = (
            output
            if isinstance(output, (dict, list))
            else {"result": str(output)}
        )
        self.tool_logs.append(
            {
                "tool_name": name,
                "tool_input": to_json_safe(inp),
                "tool_output": to_json_safe(raw_output),
                "success": 1 if success else 0,
                "duration_ms": duration_ms,
            }
        )


async def _run_tool(
    ctx: ToolContext,
    name: str,
    inp: dict,
    fn: Callable[[], Awaitable[Any]],
) -> str:
    """执行 Tool 回调并记录日志，统一返回 JSON 字符串。
    Args:
        ctx: Tool 执行上下文。
        name: Tool 名称。
        inp: 输入参数（用于日志）。
        fn: 无参异步回调，返回 Tool 实际结果。
    Returns:
        str: 成功时为结果 JSON；失败时含 error 字段的 JSON。
    """
    start = time.perf_counter()
    logger.info(
        "[tool_execute] started tool=%s conversation_id=%s input=%s",
        name,
        ctx.conversation_id,
        inp,
    )
    try:
        result = await fn()
        duration_ms = int((time.perf_counter() - start) * 1000)
        ctx.log_tool(name, inp, result, True, duration_ms)
        logger.info(
            "[tool_execute] completed tool=%s conversation_id=%s success=true duration_ms=%s",
            name,
            ctx.conversation_id,
            duration_ms,
        )
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        ctx.log_tool(
            name,
            inp,
            {"error": str(exc)},
            False,
            duration_ms,
        )
        logger.error(
            "[tool_execute] failed tool=%s conversation_id=%s duration_ms=%s error=%s",
            name,
            ctx.conversation_id,
            duration_ms,
            exc,
        )
        return json.dumps({"error": str(exc)}, ensure_ascii=False)


def build_tools(ctx: ToolContext) -> list[StructuredTool]:
    """为当前用户会话构建全部 StructuredTool 实例。
    Args:
        ctx: Tool 执行上下文。
    Returns:
        list[StructuredTool]: 已注册的 LangChain Tool 列表。
    """

    class ProductSearchInput(BaseModel):
        keyword: str = Field(description="商品搜索关键词，如 iPhone 14")
        page_size: int = Field(default=5, description="返回数量")

    class ProductDetailInput(BaseModel):
        product_id: int = Field(
            description="商品数字 ID（整数），须从 query_product 返回结果的 id 字段获取，不能填商品名称"
        )

    class OrderListInput(BaseModel):
        status: int = Field(
            default=-1,
            description="订单状态：-1全部 0待付款 1待发货 2已发货 3已完成 4已关闭",
        )

    class OrderDetailInput(BaseModel):
        order_id: int = Field(description="数据库订单ID（短整数，如42），不是订单编号")

    class OrderSnInput(BaseModel):
        order_sn: str = Field(description="订单编号 order_sn，如202607060100000020")

    class UpdateAddressInput(BaseModel):
        order_id: int = Field(description="订单ID")
        receiver_name: str = Field(description="收货人姓名")
        receiver_phone: str = Field(description="收货人电话")
        receiver_province: str = Field(description="省份")
        receiver_city: str = Field(description="城市")
        receiver_region: str = Field(description="区/县")
        receiver_detail_address: str = Field(description="详细地址")
        receiver_post_code: str = Field(default="", description="邮编")

    class LogisticsInput(BaseModel):
        order_id: int = Field(description="订单ID")

    class CouponInput(BaseModel):
        use_status: int | None = Field(default=0, description="0未使用 1已使用 2已过期")

    class ReturnApplyInput(BaseModel):
        order_id: int = Field(description="订单ID")
        product_id: int = Field(description="退货商品ID")
        order_sn: str = Field(description="订单编号")
        return_name: str = Field(description="退货人姓名")
        return_phone: str = Field(description="退货人电话")
        product_pic: str = Field(default="", description="商品图片URL")
        product_name: str = Field(description="商品名称")
        product_brand: str = Field(default="", description="商品品牌")
        product_attr: str = Field(default="", description="商品销售属性")
        product_count: int = Field(default=1, description="退货数量")
        product_price: float = Field(description="商品单价")
        product_real_price: float = Field(description="商品实际支付单价")
        reason: str = Field(description="退货原因")
        description: str = Field(default="", description="问题描述")

    class AfterSaleQueryInput(BaseModel):
        order_id: int | None = Field(default=None, description="可选，按订单筛选")

    class RefundCalcInput(BaseModel):
        product_real_price: float = Field(description="商品实际支付单价")
        product_count: int = Field(default=1, description="退货数量")

    async def query_product(keyword: str, page_size: int = 5) -> str:
        """搜索商品。
        Args:
            keyword: 搜索关键词。
            page_size: 返回条数。
        Returns:
            str: 商品搜索结果 JSON。
        """
        return await _run_tool(
            ctx,
            "query_product",
            {"keyword": keyword, "page_size": page_size},
            lambda: ctx.client.search_products(keyword=keyword, page_size=page_size),
        )

    async def query_stock(product_id: int) -> str:
        """查询商品库存（含 SKU 库存）。
        Args:
            product_id: 商品 ID。
        Returns:
            str: 库存信息 JSON。
        """

        async def _fn():
            detail = await ctx.client.product_detail(product_id)
            stock = detail.get("stock")
            skus = detail.get("skuStockList") or []
            return {"product_id": product_id, "stock": stock, "sku_stock_list": skus}

        return await _run_tool(ctx, "query_stock", {"product_id": product_id}, _fn)

    async def query_price(product_id: int) -> str:
        """查询商品价格信息。
        Args:
            product_id: 商品 ID。
        Returns:
            str: 价格信息 JSON。
        """

        async def _fn():
            detail = await ctx.client.product_detail(product_id)
            return {
                "product_id": product_id,
                "name": detail.get("name"),
                "price": detail.get("price"),
                "promotion_price": detail.get("promotionPrice"),
                "original_price": detail.get("originalPrice"),
            }

        return await _run_tool(ctx, "query_price", {"product_id": product_id}, _fn)

    async def query_coupon(use_status: int | None = 0) -> str:
        """查询会员优惠券。
        Args:
            use_status: 使用状态，0 未使用，1 已使用，2 已过期。
        Returns:
            str: 优惠券列表 JSON。
        """
        return await _run_tool(
            ctx,
            "query_coupon",
            {"use_status": use_status},
            lambda: ctx.client.list_coupons(use_status),
        )

    async def query_order(status: int = -1) -> str:
        """查询当前用户订单列表。
        Args:
            status: 订单状态筛选，-1 表示全部。
        Returns:
            str: 订单分页列表 JSON。
        """
        return await _run_tool(
            ctx,
            "query_order",
            {"status": status},
            lambda: ctx.client.list_orders(status=status),
        )

    async def query_order_detail(order_id: int) -> str:
        """按数据库订单 ID 查询订单详情（非订单编号）。
        Args:
            order_id: 数据库主键 ID。
        Returns:
            str: 订单详情 JSON。
        """
        return await _run_tool(
            ctx,
            "query_order_detail",
            {"order_id": order_id},
            lambda: ctx.client.order_detail(order_id),
        )

    async def query_order_status(order_sn: str) -> str:
        """按订单编号 order_sn 查询订单状态。
        Args:
            order_sn: 用户提供的订单编号，如 202607060100000020。
        Returns:
            str: 含 status、statusText 等字段的 JSON。
        """
        return await _run_tool(
            ctx,
            "query_order_status",
            {"order_sn": order_sn},
            lambda: ctx.client.order_status_by_sn(order_sn),
        )

    async def modify_address(**kwargs) -> str:
        """修改待发货订单收货地址。
        Args:
            **kwargs: 地址字段（order_id、receiver_name 等 snake_case）。
        Returns:
            str: mall-portal 返回结果 JSON。
        """
        payload = to_camel_dict(kwargs)
        return await _run_tool(
            ctx, "modify_address", kwargs, lambda: ctx.client.update_receiver(payload)
        )

    async def query_logistics(order_id: int) -> str:
        """查询订单物流信息（仅返回订单内快递公司/单号）。
        Args:
            order_id: 订单 ID。
        Returns:
            str: 物流信息 JSON。
        """

        async def _fn():
            detail = await ctx.client.order_detail(order_id)
            return {
                "order_id": order_id,
                "order_sn": detail.get("orderSn"),
                "status": detail.get("status"),
                "delivery_company": detail.get("deliveryCompany") or "暂无",
                "delivery_sn": detail.get("deliverySn") or "暂无",
            }

        return await _run_tool(ctx, "query_logistics", {"order_id": order_id}, _fn)

    async def create_refund(**kwargs) -> str:
        """提交退货申请。
        Args:
            **kwargs: 退货申请字段（snake_case）。
        Returns:
            str: 申请结果 JSON。
        """
        kwargs.setdefault("member_username", ctx.member_username)
        payload = to_camel_dict(kwargs)
        return await _run_tool(
            ctx,
            "create_refund",
            kwargs,
            lambda: ctx.client.create_return_apply(payload),
        )

    async def calculate_refund(
        product_real_price: float, product_count: int = 1
    ) -> str:
        """估算退款金额（实际支付单价 × 数量）。
        Args:
            product_real_price: 商品实际支付单价。
            product_count: 退货数量。
        Returns:
            str: 含 refund_amount 的 JSON。
        """
        amount = Decimal(str(product_real_price)) * product_count
        result = {
            "product_real_price": product_real_price,
            "product_count": product_count,
            "refund_amount": float(amount),
        }
        ctx.log_tool("calculate_refund", result, result, True, 0)
        return json.dumps(result, ensure_ascii=False)

    async def query_after_sale(order_id: int | None = None) -> str:
        """查询售后/退货申请进度（直连 mall 库）。
        Args:
            order_id: 可选，按订单 ID 过滤。
        Returns:
            str: 退货申请列表 JSON，含 status_text 中文状态。
        """

        async def _fn():
            rows = query_return_applies(ctx.member_username, order_id)
            for row in rows:
                row["status_text"] = RETURN_STATUS_MAP.get(row.get("status"), "未知")
            return rows

        return await _run_tool(ctx, "query_after_sale", {"order_id": order_id}, _fn)

    return [
        StructuredTool.from_function(
            coroutine=query_product,
            name="query_product",
            args_schema=ProductSearchInput,
        ),
        StructuredTool.from_function(
            coroutine=query_stock, name="query_stock", args_schema=ProductDetailInput
        ),
        StructuredTool.from_function(
            coroutine=query_price, name="query_price", args_schema=ProductDetailInput
        ),
        StructuredTool.from_function(
            coroutine=query_coupon, name="query_coupon", args_schema=CouponInput
        ),
        StructuredTool.from_function(
            coroutine=query_order, name="query_order", args_schema=OrderListInput
        ),
        StructuredTool.from_function(
            coroutine=query_order_detail,
            name="query_order_detail",
            args_schema=OrderDetailInput,
        ),
        StructuredTool.from_function(
            coroutine=query_order_status,
            name="query_order_status",
            args_schema=OrderSnInput,
        ),
        StructuredTool.from_function(
            coroutine=modify_address,
            name="modify_address",
            args_schema=UpdateAddressInput,
        ),
        StructuredTool.from_function(
            coroutine=query_logistics,
            name="query_logistics",
            args_schema=LogisticsInput,
        ),
        StructuredTool.from_function(
            coroutine=create_refund, name="create_refund", args_schema=ReturnApplyInput
        ),
        StructuredTool.from_function(
            coroutine=calculate_refund,
            name="calculate_refund",
            args_schema=RefundCalcInput,
        ),
        StructuredTool.from_function(
            coroutine=query_after_sale,
            name="query_after_sale",
            args_schema=AfterSaleQueryInput,
        ),
    ]


# 各 Expert Agent 可用的 Tool 名称集合
PRODUCT_TOOL_NAMES = {"query_product", "query_stock", "query_price", "query_coupon"}
ORDER_TOOL_NAMES = {
    "query_order",
    "query_order_detail",
    "query_order_status",
    "modify_address",
    "query_logistics",
}
AFTERSALE_TOOL_NAMES = {"create_refund", "calculate_refund", "query_after_sale"}
