# 企业级 AI Agent 电商客服系统

技术栈：Supervisor 多 Agent + GLM + Milvus RAG + FastAPI + Vue3

## 前置依赖

本项目依赖开源电商系统 **[mall](https://github.com/macrozheng/mall)** 后端，请先完成 mall 的部署与配置，再启动本客服 Agent。

mall 提供商品、订单、会员、售后等业务能力；本 Agent 通过 mall-portal API（默认 `http://localhost:8085`）及 mall 数据库与之集成。至少需要：

- **mall-portal** 已启动并可正常登录
- **MySQL** 已导入 mall 数据（`mall` 库）
- **Redis** 已启动（与 mall 共用）

mall 环境搭建可参考其官方文档：[mall 在 Windows 环境下的部署](https://www.macrozheng.com/mall/deploy/windows_deploy.html)（或 [GitHub 仓库 README](https://github.com/macrozheng/mall)）。

## 快速开始

### 1. 初始化 mall_agent 库

```bash
mysql -h localhost -P 3308 -u root -p < sql/mall_agent.sql
```

### 2. Milvus

默认连接 `localhost:19530`，请先启动 Milvus 实例。

### 3. 配置 .env

```bash
copy .env.example .env
```

填入 `ZHIPUAI_API_KEY`；若使用 `glm-4.7-flash`，设置 `GLM_MODEL=glm-4.7-flash`。

### 4. 启动后端

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python run.py
# 或安装后使用：cs-agent
```

### 5. 同步 RAG

```bash
curl -X POST http://localhost:8090/api/v1/rag/sync -H "Authorization: Bearer <token>"
```

### 6. 前端

```bash
cd ../customer-service-agent-frontend
npm install
npm run dev
```
