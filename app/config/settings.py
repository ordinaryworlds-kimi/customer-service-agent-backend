"""应用配置模块，从环境变量或 .env 文件加载配置。"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """客服 Agent 应用配置项。
    所有字段均可通过同名环境变量覆盖（不区分大小写）。
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    llm_provider: str = "qwen"
    zhipuai_api_key: str = "your-zhipu-api-key-here"
    glm_model: str = "glm-4.7-flash"
    dashscope_api_key: str = "your-dashscope-api-key-here"
    qwen_model: str = "qwen3-32b"
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    agent_host: str = "0.0.0.0"
    agent_port: int = 8090
    agent_db_host: str = "localhost"
    agent_db_port: int = 3308
    agent_db_user: str = "root"
    agent_db_password: str = "111111"
    agent_db_name: str = "mall_agent"
    mall_db_host: str = "localhost"
    mall_db_port: int = 3308
    mall_db_user: str = "root"
    mall_db_password: str = "111111"
    mall_db_name: str = "mall"
    mall_portal_base_url: str = "http://localhost:8085"
    jwt_secret: str = "mall-portal-secret"
    jwt_token_head: str = "Bearer "
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "mall_product_knowledge"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    log_level: str = "INFO"

    @property
    def agent_db_url(self) -> str:
        """mall_agent 库的 SQLAlchemy 连接 URL。"""
        return (
            f"mysql+pymysql://{self.agent_db_user}:{self.agent_db_password}"
            f"@{self.agent_db_host}:{self.agent_db_port}/{self.agent_db_name}?charset=utf8mb4"
        )

    @property
    def mall_db_url(self) -> str:
        """mall 业务库的 SQLAlchemy 连接 URL。"""
        return (
            f"mysql+pymysql://{self.mall_db_user}:{self.mall_db_password}"
            f"@{self.mall_db_host}:{self.mall_db_port}/{self.mall_db_name}?charset=utf8mb4"
        )

    @property
    def cors_origin_list(self) -> list[str]:
        """解析 CORS 允许来源为列表。"""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def jwt_bearer_prefix(self) -> str:
        """mall-portal 要求的 Authorization 前缀，固定为 ``Bearer ``（含空格）。"""
        return f"{self.jwt_token_head.rstrip()} "

    def authorization_header(self, token: str) -> str:
        """构造 mall-portal 兼容的 Authorization 请求头。"""
        return f"{self.jwt_bearer_prefix}{token.strip()}"


@lru_cache
def get_settings() -> Settings:
    """获取全局配置单例（进程内缓存）。
    Returns:
        Settings: 应用配置实例。
    """
    return Settings()
