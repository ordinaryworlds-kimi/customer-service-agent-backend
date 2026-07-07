"""Authorization 请求头构造与解析测试。"""

from app.config.settings import Settings


def test_authorization_header_normalizes_bearer_prefix() -> None:
    """无论 jwt_token_head 是否带空格，出站头都应为 ``Bearer <token>``。"""
    settings = Settings(jwt_token_head="Bearer")
    assert settings.authorization_header("jwt-token") == "Bearer jwt-token"


def test_jwt_bearer_prefix_always_has_trailing_space() -> None:
    settings = Settings(jwt_token_head="Bearer ")
    assert settings.jwt_bearer_prefix == "Bearer "
