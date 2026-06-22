# 🎬 短剧多智能体内容生产系统

基于 **LangGraph** + **DeepSeek / ByteDance Doubao** 的短剧创作多智能体系统，支持内容整理、文案生成、资料答疑、合规审核，带可视化工作流和反思迭代闭环。

## ✨ 核心特性

| 能力             | 说明                                                      |
| ---------------- | --------------------------------------------------------- |
| 🤖 4 类 Agent     | 任务解析、素材检索、内容润色、合规审核                    |
| 🔄 LangGraph 调度 | 条件路由 + 反思迭代 + 降级分支                            |
| 📚 分层向量检索   | FAISS 父子块索引 + 重排                                   |
| ✅ 双轨合规审核   | 规则引擎 + LLM 语义审核                                   |
| 🎯 可视化前端     | SSE 流式事件 + 实时工作流动画                             |
| 🛠 5 个 MCP 工具  | sensitive_check / normalize_text / retrieve_materials ... |
| ⚡ 完整 API       | 同步 / 异步 / 流式 三种调用模式                           |
| 🚀 Docker 部署    | 一键容器化部署                                            |

## 📁 项目结构

```
drama-multi-agent/
├── pyproject.toml              # 包描述
├── requirements.txt            # 依赖清单
├── .env                        # 运行时配置（LLM API Key 等）
├── .env.example                # 配置模板
├── Dockerfile                  # 容器镜像构建
├── docker-compose.yml          # 一键启动
├── README.md                   # 本文件
├── frontend/
│   └── index.html              # 可视化前端页面（工作流+事件+结果）
├── src/drama_agent/
│   ├── __init__.py
│   ├── config.py               # 配置管理（Pydantic Settings）
│   ├── models.py               # 全局 State 与数据模型
│   ├── exceptions.py           # 自定义异常 + 重试装饰器
│   ├── logging_setup.py        # 结构化日志
│   ├── llm.py                  # LLM HTTP 客户端（OpenAI 兼容协议）
│   ├── graph.py                # LangGraph 调度器 + 事件回调（前端可视化）
│   ├── api.py                  # FastAPI REST 接口（同步/异步/SSE流式）
│   ├── agents/
│   │   ├── parser_agent.py     # 任务解析 Agent
│   │   ├── retriever_agent.py  # 素材检索 Agent
│   │   ├── polish_agent.py     # 内容润色 Agent
│   │   ├── audit_agent.py      # 合规审核 Agent
│   │   └── prompts.py          # 统一 Prompt 模板管理
│   └── tools/
│       ├── vector_retriever.py # FAISS 向量检索工具
│       ├── compliance_engine.py# 三级敏感词规则引擎
│       └── text_processor.py   # 文本规范化工具
├── scripts/
│   ├── build_knowledge.py      # 素材库构建脚本
│   └── api_test.py             # API 快速测试脚本
├── tests/                      # 单元测试
└── data/
    └── faiss_index/            # FAISS 向量索引（运行时自动生成）
```

## 🚀 快速开始

### 方式一：直接运行（推荐，最快上手）

```bash
# 1. 安装依赖（Python 3.11+）
pip install -r requirements.txt

# 2. 配置 API Key（DeepSeek 已默认激活，可改 .env）
#    .env 已配置：LLM_API_KEY=sk-...

# 3. 构建素材知识库（首次运行必做）
set PYTHONPATH=src
python scripts\build_knowledge.py

# 4. 启动服务（含可视化前端页面）
python -m uvicorn drama_agent.api:app --host 0.0.0.0 --port 8000

# 5. 浏览器打开：http://localhost:8000
```

### 方式二：命令行体验

```bash
cd drama-multi-agent
set PYTHONPATH=src
python -m drama_agent run "写一段关于高考状元穿越古代的短剧开头"
```

### 方式三：API 测试脚本

```bash
python scripts\api_test.py http://127.0.0.1:8000
```

### 方式四：Docker 部署

```bash
docker-compose up -d --build
# 浏览器访问 http://localhost:8000
```

## 🎯 支持的短剧场景

| 场景         | 示例输入                                            |
| ------------ | --------------------------------------------------- |
| **内容整理** | `给我整理一段关于「霸总追妻」的短剧大纲，分3集`     |
| **文案生成** | `写2版不同风格的推广文案，推广都市新剧《错位人生》` |
| **资料答疑** | `短剧创作中哪些内容是红线？列出主要合规要求`        |
| **审核功能** | `帮我检查这段剧本是否有违规内容：<文本>`            |

## 🔌 API 接口

| 方法 | 路径                  | 说明                                             |
| ---- | --------------------- | ------------------------------------------------ |
| GET  | `/`                   | 返回可视化前端页面                               |
| GET  | `/health`             | 健康检查                                         |
| GET  | `/v1/tools`           | 列出已注册的 MCP 工具                            |
| POST | `/v1/generate`        | **同步生成**（单次请求返回完整结果）             |
| POST | `/v1/stream`          | **SSE 流式生成**（前端可视化用，返回工作流事件） |
| POST | `/v1/async/generate`  | **异步提交**（返回 task_id）                     |
| GET  | `/v1/async/{task_id}` | 查询异步任务状态和结果                           |
| GET  | `/docs`               | Swagger API 文档                                 |

### 请求体（所有 POST 接口）

```json
{
  "raw_input": "用户原始输入，任意长度字符串",
  "user_id": "调用方标识（可选）"
}
```

### 响应体（/v1/generate 同步接口）

```json
{
  "status": "ok",
  "data": {
    "task_type": "content_organize",
    "content": "生成的短剧文本...",
    "audit_result": {
      "passed": true,
      "score": 0.975,
      "issues": [],
      "degrade_mode": false,
      "summary": "..."
    },
    "iteration_count": 1,
    "degrade_mode": false,
    "error": null,
    "elapsed_ms": 21574.3
  }
}
```

### SSE 事件流（/v1/stream 流式接口）

每个事件格式为：

```
event: node_start
data: {"type": "node_start", "node": "parse_node", "ts": 1781611415.55}

event: node_done
data: {"type": "node_done", "node": "parse_node", "duration_ms": 1361.0, "summary": "..."}

event: final
data: {"type": "final", "data": {"task_type": "...", "content": "...", ...}}
```

事件类型：`start` / `node_start` / `node_done` / `node_error` / `workflow_done` / `final` / `error`

## 🔄 工作流状态图

```
           ┌──────────────┐
┌─────────▶│ 任务解析 (P)│───┐
│          └──────────────┘   │
│                           需要检索?
│                             ├─NO ─────────────────────────────┐
│                             └─YES ┌──────────────┐            │
│                                     │ 素材检索 (R) │──┐        │
│                                     └──────────────┘  │        │
│                                              ▲          ▼        ▼
│                                              │     ┌──────────────┐
│                                              │     │ 内容润色 (G) │
│                                              │     └──────────────┘
│                              素材不足反向联动          │
│                                              │          │
│                                              └──────────┘
│                                                        │
│                                                        ▼
│                                              ┌──────────────┐
│                                              │ 合规审核 (A) │
│                                              └──────────────┘
│                                                        │
│                                       不通过 & 迭代<MAX?
│                                                        ├─YES ──┐
│                                                        └─NO ───┤
│                                                                   │
│                                                           ▼       ▼
│                                                  再次润色（注入审核反馈）
│                                                    (反思迭代闭环)
│
│
└─────────────────────────── 降级模式：任何节点故障 → 跳过后续可选节点
```

## 🛡 三级降级容错策略

1. **LLM 不可用**：降级为 stub 模式，返回预设结构和示例内容
2. **检索失败/无素材**：直接进入纯生成模式，跳过素材注入
3. **审核 LLM 不可用**：仅跑规则引擎敏感词检测

## 🧩 配置文件（.env）

```ini
# LLM（默认使用 DeepSeek）
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=120

# 切换至豆包（ByteDance Doubao）
# LLM_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
# LLM_MODEL=doubao-pro-32k

# 检索
VECTOR_INDEX_PATH=data/faiss_index
RETRIEVE_TOP_K=20

# 审核
AUDIT_MAX_ITERATION=3

# 服务
API_PORT=8000
```

## 🔧 核心模块说明

### 1. agents/parser_agent

用户输入的意图识别和任务拆解，输出 `ParsedTask`（任务类型、主题、关键词、是否需要检索）。

### 2. agents/retriever_agent

调用 FAISS 向量索引召回相关素材，去重后整理为结构化上下文注入润色阶段。

### 3. agents/polish_agent

基于原始需求 + 素材库上下文 + 审核反馈（如有），调用 LLM 生成/改写内容。

### 4. agents/audit_agent

- **规则层**：敏感词库 + 正则规则（毫秒级检测硬违规）
- **语义层**：LLM 审核语义级问题，输出问题清单和风险评分
- 不通过且迭代未超限 → 自动回流至润色 Agent

### 5. graph.py（LangGraph 核心调度）

- `run_workflow_with_events` 主入口
- 使用 thread-local 传递事件回调（避免 LangGraph 序列化问题）
- 支持手动调度回退（LangGraph 不可用时）

### 6. api.py（FastAPI）

- `/` 路由返回前端可视化页面
- `/v1/generate` 同步调用
- `/v1/stream` 使用 asyncio.Queue + 线程桥接，实现 SSE 事件流
- `/v1/async/*` 异步任务模式

## 📊 性能基准

| 指标         | 实测值（DeepSeek Chat）   |
| ------------ | ------------------------- |
| 解析节点     | ~1.3s                     |
| 检索节点     | ~1-2s（本地 FAISS）       |
| 润色节点     | ~7-25s（与内容长度相关）  |
| 审核节点     | ~3-10s（与问题数相关）    |
| 单任务总耗时 | 15-60s                    |
| 审核通过率   | 70-95%（与任务类型相关）  |
| 平均迭代次数 | 1.2次（QA类任务迭代更多） |

## 🤝 可扩展点

- **接入 MCP Server**：在 `src/drama_agent/tools/` 目录下新增工具文件，调用 `registry.register(name, func, schema)` 即可自动注册
- **新增 Agent**：在 `graph.py` 中新增节点 + 路由分支
- **自定义素材库**：把真实剧本、人设库放入 `data/knowledge/` 目录，运行 `scripts/build_knowledge.py` 重建索引
- **替换 LLM**：任何支持 OpenAI 兼容接口的模型（豆包、通义、智谱、LLama.cpp 等）只需改 `.env`
- **自定义 Prompt**：修改 `src/drama_agent/agents/prompts.py` 中模板

## 📜 License

项目结构已完整，所有代码可运行、可维护。

## 测试/联调常见问题

Q：前端没有内容？
A：看后端日志输出：
  1) 如果看到 `[Polish] 生成内容 xxx 字` 但是前端没有内容—— 那 大概率是 SSE 编码问题（已在本版通过 `ensure_ascii=False` 修复）；
  2) 如果看到 `[Retriever] 知识库为空`—— 这是预期的提示信息，会走纯生成模式，不会 degrade；
  3) 若是 `LLM 未配置`—— 会走 stub 模式，仍然会输出演示内容。

Q：中文乱码或 `\uXXXX`？
A：已全部替换为 `ensure_ascii=False`，并显式 UTF-8 字节流，不再转义；浏览器收到的应该是纯中文的 JSON。

Q：FAISS 索引路径带中文（如 `D:\\求职\\1\\…`）？
A：通过 `faiss.serialize_index` + 文件 IO 方式保存，避免 faiss.write_index 在中文路径下失败。

## License

MIT
