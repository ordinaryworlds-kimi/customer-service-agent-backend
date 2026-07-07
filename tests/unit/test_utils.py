"""to_json_safe 等工具函数单元测试。"""

import json
from datetime import datetime
from decimal import Decimal

from app.tools.utils import to_json_safe


def test_to_json_safe_decimal_and_datetime():
    value = {
        "return_amount": Decimal("3585.50"),
        "create_time": datetime(2026, 7, 6, 12, 30, 0),
        "items": [{"price": Decimal("99.9")}],
    }
    safe = to_json_safe(value)
    json.dumps(safe)
    assert safe["return_amount"] == 3585.5
    assert safe["create_time"] == "2026-07-06T12:30:00"
    assert safe["items"][0]["price"] == 99.9
