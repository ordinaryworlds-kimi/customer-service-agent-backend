# 企业级 AI Agent 电商客服系统

基于 **Supervisor 多 Agent** 架构的智能电商客服，对接开源商城 [mall](https://github.com/macrozheng/mall)，支持商品咨询、订单查询、售后处理等场景。技术栈：Supervisor 多 Agent + GLM / 通义千问 + Milvus RAG + FastAPI + Vue3。

## 功能概览

- **多 Agent 协作**：Supervisor 分析用户意图，将任务分派给商品 / 订单 / 售后三类 Expert Agent
- **Tool 调用**：Expert Agent 通过 ReAct 模式调用 11 个业务 Tool，代理访问 mall-portal API 与 mall 数据库
- **RAG 知识检索**：商品与帮助文档向量化存入 Milvus，商品类咨询自动检索增强
- **会话与记忆**：短期对话历史 + LLM 提取的长期用户偏好，持久化至 `mall_agent` 库
- **流式对话**：SSE 逐 token 推送回复，前端实时渲染

## 系统架构

整体由三层组成：Vue3 前端 → FastAPI Agent 后端 → mall-portal 及外部存储。

```
┌─────────────────────────────────────────────────────────────────┐
│  浏览器 (localhost:5173)                                        │
│  customer-service-agent-frontend                                │
│  LoginView / ChatView  ── SSE 流式聊天                          │
└────────────┬───────────────────────────────┬────────────────────┘
             │ /portal-api/*                 │ /agent-api/*
             ▼                               ▼
┌────────────────────────┐    ┌──────────────────────────────────┐
│  mall-portal :8085     │    │  customer-service-agent :8090    │
│  (Spring Boot + JWT)   │◄───│  FastAPI + Supervisor Workflow    │
│  商品 / 订单 / 会员 API │    │  ┌──────────┐  ┌───────────────┐  │
└────────────┬───────────┘    │  │Supervisor│→ │Expert Agents  │  │
             │                │  │ 意图分析  │  │product/order/ │  │
             ▼                │  └──────────┘  │   aftersale   │  │
┌────────────────────────┐    │       │        └───────┬───────┘  │
│  mall 库 (MySQL)       │◄───┼───────┼────────────────┼──────────┤
│  会员 / 商品 / 售后     │    │       ▼                ▼          │
└────────────────────────┘    │  Milvus RAG    GLM / Qwen LLM     │
                              └──────────┬───────────────────────┘
                                         ▼
                              ┌────────────────────────┐
                              │  mall_agent 库 (MySQL) │
                              │  会话 / 消息 / 记忆     │
                              └────────────────────────┘
```

### 对话处理流程

1. 前端携带 mall-portal JWT 调用 `/api/v1/chat/stream`
2. 后端校验 JWT，查询 `mall.ums_member` 确认用户身份
3. **Supervisor** 分析意图，输出任务计划 `{ tasks, need_rag }`
4. 按需从 **Milvus** 检索商品/帮助知识（RAG）
5. 各 **Expert Agent** 以 ReAct 模式调用 Tool（最多 4 轮），访问 mall-portal 或 mall 库
6. **Supervisor 汇总** 生成最终回复，SSE 流式返回；同步写入会话、Tool 日志与用户记忆

### 多 Agent 分工

| Agent | 职责 | 可用 Tool |
|-------|------|-----------|
| `product` | 商品搜索、库存、价格、优惠券 | `query_product`, `query_stock`, `query_price`, `query_coupon` |
| `order` | 订单列表、详情、改地址、物流 | `query_order`, `query_order_detail`, `modify_address`, `query_logistics` |
| `aftersale` | 退款申请、退款计算、售后查询 | `create_refund`, `calculate_refund`, `query_after_sale` |

## 项目结构

本仓库为 **Agent 后端**；前端位于同级目录 `customer-service-agent-frontend`。

```
mall/
├── customer-service-agent-backend/     # 本仓库（FastAPI Agent 后端）
│   ├── app/
│   │   ├── main.py                     # FastAPI 入口，CORS、路由注册
│   │   ├── cli.py                      # uvicorn 启动 CLI（cs-agent 命令）
│   │   ├── api/
│   │   │   └── chat.py                 # 聊天 / 会话 / RAG 同步 API
│   │   ├── auth/
│   │   │   └── jwt.py                  # JWT 解析，mall 会员校验
│   │   ├── workflows/
│   │   │   └── graph.py                # Supervisor 编排主流程
│   │   ├── agents/
│   │   │   └── runner.py               # Expert Agent ReAct 执行器
│   │   ├── tools/
│   │   │   ├── registry.py             # Tool 注册与分组
│   │   │   ├── mall_client.py          # mall-portal HTTP 客户端
│   │   │   └── mall_db.py              # mall 库直连查询
│   │   ├── rag/
│   │   │   └── milvus_store.py         # Milvus 向量检索与知识同步
│   │   ├── memory/
│   │   │   └── store.py                # 短期历史与长期记忆
│   │   ├── llm/
│   │   │   ├── provider.py             # LLM 提供商切换（glm / qwen）
│   │   │   ├── glm.py                  # 智谱 GLM 客户端
│   │   │   └── qwen.py                 # 通义千问客户端
│   │   ├── prompts/
│   │   │   └── templates.py            # Supervisor / Expert / 汇总 Prompt
│   │   ├── models/
│   │   │   ├── db.py                   # SQLAlchemy ORM（mall_agent + mall）
│   │   │   └── schemas.py              # Pydantic 请求/响应模型
│   │   └── config/
│   │       └── settings.py             # 环境变量与配置
│   ├── sql/
│   │   └── mall_agent.sql              # Agent 专用库建表脚本
│   ├── tests/                          # 单元测试与集成测试
│   ├── docs/
│   │   └── API_CALL_CHAIN.md           # 前后端调用链路详细文档
│   ├── run.py                          # 开发启动脚本
│   ├── pyproject.toml
│   └── .env.example
│
└── customer-service-agent-frontend/    # Vue3 前端
    └── src/
        ├── views/
        │   ├── LoginView.vue           # 登录（对接 mall-portal SSO）
        │   └── ChatView.vue            # 聊天主界面（SSE）
        ├── api/index.ts                # HTTP / SSE 请求封装
        ├── stores/auth.ts              # Token 持久化
        └── router/index.ts             # 路由与登录守卫
```

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

主要配置项：

| 配置项 | 说明 |
|--------|------|
| `LLM_PROVIDER` | 大模型提供商：`glm` 或 `qwen` |
| `ZHIPUAI_API_KEY` / `DASHSCOPE_API_KEY` | 对应 LLM 的 API Key |
| `GLM_MODEL` / `QWEN_MODEL` | 模型名称，如 `glm-4.7-flash` |
| `JWT_SECRET` | 须与 mall-portal 一致（默认 `mall-portal-secret`） |
| `AGENT_DB_*` / `MALL_DB_*` | Agent 库与 mall 库连接信息 |
| `MILVUS_HOST` / `MILVUS_PORT` | Milvus 地址 |

### 4. 启动后端

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python run.py
# 或安装后使用：cs-agent
```

服务默认监听 `http://localhost:8090`，健康检查：`GET /health`。

### 5. 同步 RAG

首次使用前，将 mall 商品与帮助文档同步至 Milvus：

```bash
curl -X POST http://localhost:8090/api/v1/rag/sync -H "Authorization: Bearer <token>"
```

`<token>` 为 mall-portal 登录后获得的 JWT。

### 6. 前端

```bash
cd ../customer-service-agent-frontend
npm install
npm run dev
```

浏览器访问 `http://localhost:5173`，使用 mall 会员账号登录即可开始对话。

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/v1/chat` | 非流式聊天 |
| POST | `/api/v1/chat/stream` | SSE 流式聊天（前端主要使用） |
| GET | `/api/v1/conversations` | 当前用户会话列表 |
| GET | `/api/v1/conversations/{id}/messages` | 会话消息历史 |
| POST | `/api/v1/rag/sync` | 同步商品/帮助文档到 Milvus |

所有 `/api/v1/*` 接口需在请求头携带 `Authorization: Bearer <jwt_token>`（mall-portal 签发）。

## 测试

```bash
pip install -e ".[dev]"
pytest
```

## 更多文档

- [前后端调用链路详解](docs/API_CALL_CHAIN.md) — 含登录流程、SSE 协议、Tool 映射与端到端时序图
