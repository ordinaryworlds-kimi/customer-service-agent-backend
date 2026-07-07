"""命令行入口模块。"""

import uvicorn

from app.config.settings import get_settings


def main() -> None:
    """启动 FastAPI 开发服务器。
    读取配置中的 host/port，并以热重载模式运行 uvicorn。
    """
    s = get_settings()
    uvicorn.run("app.main:app", host=s.agent_host, port=s.agent_port, reload=True)
