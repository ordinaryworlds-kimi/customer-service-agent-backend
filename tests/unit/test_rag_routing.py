"""RAG 启用条件与降级行为测试。"""

from app.workflows.graph import _resolve_rag_context, _should_use_rag


class TestShouldUseRag:
    def test_disabled_when_need_rag_false(self) -> None:
        plan = {
            "tasks": [{"agent": "product", "instruction": "查商品"}],
            "need_rag": False,
        }
        assert _should_use_rag(plan) is False

    def test_skipped_for_order_only_even_if_need_rag_true(self) -> None:
        plan = {
            "tasks": [{"agent": "order", "instruction": "查物流"}],
            "need_rag": True,
        }
        assert _should_use_rag(plan) is False

    def test_enabled_for_product_task(self) -> None:
        plan = {
            "tasks": [{"agent": "product", "instruction": "推荐手机"}],
            "need_rag": True,
        }
        assert _should_use_rag(plan) is True

    def test_enabled_when_mixed_tasks_include_product(self) -> None:
        plan = {
            "tasks": [
                {"agent": "product", "instruction": "查库存"},
                {"agent": "order", "instruction": "查订单"},
            ],
            "need_rag": True,
        }
        assert _should_use_rag(plan) is True


def test_resolve_rag_context_skips_milvus_for_order_plan(mocker) -> None:
    mock_format = mocker.patch("app.workflows.graph.format_rag_context")
    plan = {
        "tasks": [{"agent": "order", "instruction": "查订单"}],
        "need_rag": True,
    }

    result = _resolve_rag_context("查订单", plan)

    assert result == "未启用 RAG。"
    mock_format.assert_not_called()


def test_resolve_rag_context_calls_milvus_for_product_plan(mocker) -> None:
    mock_format = mocker.patch(
        "app.workflows.graph.format_rag_context",
        return_value="RAG 参考内容",
    )
    plan = {
        "tasks": [{"agent": "product", "instruction": "查商品"}],
        "need_rag": True,
    }

    result = _resolve_rag_context("查商品", plan)

    assert result == "RAG 参考内容"
    mock_format.assert_called_once_with("查商品")


def test_format_rag_context_graceful_when_milvus_down(mocker) -> None:
    from app.rag.milvus_store import format_rag_context

    mocker.patch(
        "app.rag.milvus_store.search_knowledge",
        side_effect=RuntimeError("Milvus unavailable"),
    )

    result = format_rag_context("查商品")

    assert "不可用" in result
