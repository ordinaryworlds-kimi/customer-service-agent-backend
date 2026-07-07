"""FastAPI 应用入口模块。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.config.logging_config import setup_logging
from app.config.settings import get_settings

setup_logging()
settings = get_settings()
app = FastAPI(title="Mall Customer Service Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])


@app.get("/health")
def health() -> dict[str, str]:
    """健康检查接口。
    Returns:
        dict[str, str]: 包含 status 字段，值为 ok 表示服务正常。
    """
    return {"status": "ok"}
