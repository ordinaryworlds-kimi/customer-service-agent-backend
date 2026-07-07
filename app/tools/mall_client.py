"""mall-portal HTTP API 客户端。"""

import logging
import time
from typing import Any

import httpx

from app.config.settings import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class MallPortalClient:
    """封装对 mall-portal 前台商城 API 的异步 HTTP 调用。"""

    def __init__(self, token: str) -> None:
        """初始化客户端。
        Args:
            token: mall-portal JWT Token（不含 Bearer 前缀）。
        """
        self.token = token
        self.base_url = settings.mall_portal_base_url.rstrip("/")
        self.headers = {"Authorization": settings.authorization_header(token)}

    async def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """发送 HTTP 请求并解析 CommonResult 响应。
        Args:
            method: HTTP 方法，如 GET、POST。
            path: API 路径，如 /order/list。
            **kwargs: 传递给 httpx 的额外参数（params、json 等）。
        Returns:
            Any: CommonResult.data 字段内容。
        Raises:
            httpx.HTTPStatusError: HTTP 状态码非 2xx。
            RuntimeError: 业务 code 非 200。
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            start = time.perf_counter()
            logger.info("[mall_portal] request started method=%s path=%s", method, path)
            resp = await client.request(
                method,
                f"{self.base_url}{path}",
                headers=self.headers,
                **kwargs,
            )
            duration_ms = int((time.perf_counter() - start) * 1000)
            resp.raise_for_status()
            data = resp.json()
            business_code = data.get("code")
            logger.info(
                "[mall_portal] request completed method=%s path=%s http_status=%s business_code=%s duration_ms=%s",
                method,
                path,
                resp.status_code,
                business_code,
                duration_ms,
            )
        if business_code != 200:
            raise RuntimeError(data.get("message") or "mall-portal 请求失败")
        return data.get("data")

    async def search_products(
        self,
        keyword: str | None = None,
        page_num: int = 1,
        page_size: int = 5,
        sort: int = 0,
    ) -> dict[str, Any]:
        """搜索商品。
        Args:
            keyword: 搜索关键词，可为空。
            page_num: 页码，从 1 开始。
            page_size: 每页条数。
            sort: 排序方式，0 相关度 1 新品 2 销量 3 价格升 4 价格降。
        Returns:
            dict[str, Any]: 分页商品列表（CommonPage 结构）。
        """
        params: dict[str, Any] = {
            "pageNum": page_num,
            "pageSize": page_size,
            "sort": sort,
        }
        if keyword:
            params["keyword"] = keyword
        return await self._request("GET", "/product/search", params=params)

    async def product_detail(self, product_id: int) -> dict[str, Any]:
        """获取商品详情（含 SKU 库存）。
        Args:
            product_id: 商品 ID。
        Returns:
            dict[str, Any]: 商品详情对象。
        """
        return await self._request("GET", f"/product/detail/{product_id}")

    async def list_orders(
        self, status: int = -1, page_num: int = 1, page_size: int = 5
    ) -> dict[str, Any]:
        """分页查询当前用户订单。
        Args:
            status: 订单状态，-1 全部，0 待付款，1 待发货，2 已发货，3 已完成，4 已关闭。
            page_num: 页码。
            page_size: 每页条数。
        Returns:
            dict[str, Any]: 分页订单列表。
        """
        return await self._request(
            "GET",
            "/order/list",
            params={"status": status, "pageNum": page_num, "pageSize": page_size},
        )

    async def order_detail(self, order_id: int) -> dict[str, Any]:
        """获取订单详情。
        Args:
            order_id: 订单 ID。
        Returns:
            dict[str, Any]: 订单详情（含订单项、物流信息）。
        """
        return await self._request("GET", f"/order/detail/{order_id}")

    async def order_status_by_sn(self, order_sn: str) -> dict[str, Any]:
        """按订单编号查询当前会员订单状态。
        Args:
            order_sn: 订单编号，如 202607060100000020。
        Returns:
            dict[str, Any]: 订单状态摘要（含 status、statusText 等）。
        """
        return await self._request("GET", f"/order/status/{order_sn}")

    async def update_receiver(self, payload: dict[str, Any]) -> Any:
        """修改待发货订单收货地址。
        Args:
            payload: camelCase 格式的地址参数（orderId、receiverName 等）。
        Returns:
            Any: 更新影响的行数或 mall-portal 返回 data。
        """
        return await self._request("POST", "/order/updateReceiver", json=payload)

    async def list_coupons(self, use_status: int | None = None) -> list[Any]:
        """查询会员优惠券列表。
        Args:
            use_status: 使用状态，0 未使用，1 已使用，2 已过期；None 表示不限。
        Returns:
            list[Any]: 优惠券列表。
        """
        params = {}
        if use_status is not None:
            params["useStatus"] = use_status
        result = await self._request("GET", "/member/coupon/list", params=params)
        return result or []

    async def create_return_apply(self, payload: dict[str, Any]) -> Any:
        """提交退货申请。
        Args:
            payload: camelCase 格式的退货申请参数。
        Returns:
            Any: 新建退货申请 ID 或 mall-portal 返回 data。
        """
        return await self._request("POST", "/returnApply/create", json=payload)

    async def list_addresses(self) -> list[Any]:
        """查询会员收货地址列表。
        Returns:
            list[Any]: 收货地址列表。
        """
        result = await self._request("GET", "/member/address/list")
        return result or []
