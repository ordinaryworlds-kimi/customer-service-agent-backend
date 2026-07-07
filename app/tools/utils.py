"""工具函数：Python snake_case 与 Java camelCase 字段名转换。"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any


def to_json_safe(value: Any) -> Any:
    """递归将值转为 JSON 可序列化类型（用于 DB JSON 列持久化）。
    Args:
        value: 任意 Python 对象。
    Returns:
        Any: 可被标准 json.dumps 序列化的值。
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [to_json_safe(v) for v in value]
    return value


def snake_to_camel(name: str) -> str:
    """将 snake_case 字符串转为 camelCase。
    Args:
        name: 下划线命名的字段名，如 receiver_name。
    Returns:
        str: 驼峰命名的字段名，如 receiverName。
    """
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def to_camel_dict(data: dict) -> dict:
    """将字典所有键从 snake_case 转为 camelCase。
    Args:
        data: 原始参数字典。
    Returns:
        dict: 键名已转为 camelCase 的新字典。
    """
    return {snake_to_camel(k): v for k, v in data.items()}
