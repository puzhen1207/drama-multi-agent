# Drama Multi Agent

短剧多智能体内容生产系统。项目基于 FastAPI、LangGraph、FAISS 和 OpenAI 兼容大模型接口，提供从需求解析、素材检索、内容生成、合规审核到会话/个人记忆沉淀的一体化短剧创作工作流。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-enabled-purple)](https://www.langchain.com/langgraph)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## 项目亮点

- 多 Agent 工作流：任务解析、素材检索、内容润色、合规审核四类节点协同运行。
- LangGraph 调度：使用 StateGraph 组织条件路由、审核反馈和反思迭代。
- 可视化前端：内置单页 Web UI，实时展示 SSE 事件流、节点状态、生成结果和审核报告。
- 个人记忆库：用户先预览生成内容，再手动确认是否保存为个人 Q&A 记忆；相似问题会自动召回参考。
- 会话记忆：支持多轮对话、用户画像、反思日志和会话持久化。
- 知识库检索：支持将本地素材构建为 FAISS 向量索引，并在生成前召回相关上下文。
- 合规审核：规则引擎与 LLM 语义审核并行，审核不通过时可触发重写。
- 多种调用方式：Web UI、REST API、SSE 流式接口、CLI、MCP Server、Docker 均可运行。

## 系统架构

```text
User / Web UI / API / CLI
        |
        v
FastAPI service
        |
        v
LangGraph StateGraph
        |
        +--> Parser Agent      解析任务类型、主题、约束
        +--> Retriever Agent   检索公共素材与个人记忆
        +--> Polish Agent      生成/润色短剧内容
        +--> Audit Agent       规则审核 + LLM 语义审核
        |
        v
Final response + audit report + session memory
```

核心运行路径：

```text
用户输入
  -> 任务解析
  -> 素材/个人记忆检索
  -> 内容生成
  -> 合规审核
  -> 审核未通过且未达迭代上限时回到生成节点
  -> 输出最终内容、审核结果和会话记录
```

## 项目结构

```text
drama-multi-agent/
├── frontend/
│   └── index.html              # 可视化前端页面
├── src/drama_agent/
│   ├── api.py                  # FastAPI 路由
│   ├── graph.py                # LangGraph 工作流
│   ├── llm.py                  # OpenAI 兼容 LLM 客户端
│   ├── memory.py               # 会话记忆与用户画像
│   ├── mcp_server.py           # MCP Server
│   ├── models.py               # Pydantic 数据模型
│   ├── agents/                 # Parser / Retriever / Polish / Audit
│   └── tools/                  # 向量检索、合规、文本处理、个人记忆
├── scripts/
│   ├── build_knowledge.py      # 构建 FAISS 知识库索引
│   └── api_test.py             # API 快速测试脚本
├── tests/                      # pytest 测试
├── data/
│   ├── knowledge/              # 自定义素材
│   ├── faiss_index/            # FAISS 索引
│   ├── user_memory/            # 个人记忆库
│   └── sessions/               # 会话持久化文件
├── pyproject.toml
├── requirements.txt
├── .env.example
├── Dockerfile
└── docker-compose.yml
```

## 环境要求

- Python 3.10+
- Windows、macOS 或 Linux
- 可选：DeepSeek、豆包、OpenAI 或任意 OpenAI 兼容接口的 API Key
- 可选：Docker / Docker Compose

如果没有配置 LLM API Key，系统会进入本地 Stub 模式，仍可跑通完整流程，适合做 UI 和 API 联调。

## 快速开始

### 1. 获取项目

```bash
git clone <your-repo-url>
cd drama-multi-agent
```

如果项目已经在本地，例如：

```powershell
cd /d D:\1求职\1\drama-multi-agent
```

### 2. 创建并激活虚拟环境

PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

CMD：

```cmd
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

可选安装开发依赖：

```bash
pip install -e ".[dev,mcp]"
```

### 4. 配置环境变量

复制配置模板：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

常用配置：

```ini
LLM_API_KEY=sk-your-api-key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=120

EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
EMBEDDING_DIM=512

VECTOR_INDEX_PATH=data/faiss_index
MATERIAL_KNOWLEDGE_PATH=data/knowledge

USER_MEMORY_PATH=data/user_memory
ENABLE_USER_MEMORY=true

API_HOST=127.0.0.1
API_PORT=8000
```

## 运行服务

### 方式一：命令行运行

PowerShell：

```powershell
cd /d D:\1求职\1\drama-multi-agent
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m uvicorn drama_agent.api:app --host 127.0.0.1 --port 8001
```

CMD：

```cmd
cd /d D:\1求职\1\drama-multi-agent
.venv\Scripts\activate
set PYTHONPATH=src
python -m uvicorn drama_agent.api:app --host 127.0.0.1 --port 8001
```

启动后访问：

- Web UI: http://127.0.0.1:8001/
- Swagger API Docs: http://127.0.0.1:8001/docs
- Health Check: http://127.0.0.1:8001/health

开发时可加 `--reload`：

```bash
python -m uvicorn drama_agent.api:app --host 127.0.0.1 --port 8001 --reload
```

### 方式二：项目 CLI

安装为可编辑包后，可以使用 `drama-agent` 命令：

```bash
drama-agent serve --host 127.0.0.1 --port 8001
```

单次运行工作流：

```bash
drama-agent run "写一个高考状元穿越古代的短剧开头"
```

查看注册工具：

```bash
drama-agent tools
```

启动 MCP Server：

```bash
drama-agent mcp
```

### 方式三：PyCharm 运行

1. 使用 PyCharm 打开项目根目录。
2. 解释器选择：

```text
D:\1求职\1\drama-multi-agent\.venv\Scripts\python.exe
```

3. 新建 Python 运行配置：

```text
Module name: uvicorn
Parameters: drama_agent.api:app --host 127.0.0.1 --port 8001
Working directory: D:\1求职\1\drama-multi-agent
Environment variables: PYTHONUNBUFFERED=1;PYTHONPATH=src
```

如果 `8001` 端口被占用，可改成 `8002` 并访问对应端口。

### 方式四：Docker

```bash
docker compose up -d --build
```

默认访问：

```text
http://127.0.0.1:8000/
```

## 构建知识库索引

将剧本、人物设定、文案、人设资料等文本文件放入：

```text
data/knowledge/
```

然后执行：

```bash
python scripts/build_knowledge.py --rebuild
```

如果日志出现类似提示：

```text
FAISS 索引维度 384 与当前 embedding 1024 不一致
```

说明当前 embedding 模型和旧索引维度不同，需要重新构建索引：

```bash
python scripts/build_knowledge.py --rebuild
```

## Web UI 使用说明

访问首页后可完成以下操作：

1. 输入短剧创作需求。
2. 点击实时生成，观察每个 Agent 的执行过程。
3. 在结果区查看完整生成内容与合规审核结果。
4. 预览满意后，点击结果工具栏中的保存到记忆库。
5. 在弹窗中确认完整问答内容，再选择是否保存。
6. 后续相似问题会自动召回个人记忆作为参考。

个人记忆库支持：

- 保存生成问答
- 查看全部记忆
- 编辑记忆
- 删除记忆
- 导出 JSON
- 导入 JSON

## REST API

### 健康检查

```http
GET /health
```

返回示例：

```json
{
  "status": "ok",
  "version": "2.1.0",
  "memory_module": true,
  "user_memory_enabled": true,
  "embedding": {
    "available": true,
    "mode": "sentence_transformers"
  }
}
```

### 同步生成

```http
POST /v1/generate
Content-Type: application/json
```

```json
{
  "raw_input": "写一段都市逆袭短剧开头",
  "user_id": "demo-user",
  "session_id": "optional-session-id"
}
```

### SSE 流式生成

```http
POST /v1/stream
Content-Type: application/json
```

事件类型：

- `start`
- `node_start`
- `node_done`
- `node_error`
- `workflow_done`
- `workflow_complete`
- `final`
- `error`

### 异步任务

```http
POST /v1/async/generate
GET  /v1/async/{task_id}
```

### 会话接口

| Method | Path | Description |
| --- | --- | --- |
| GET | `/v1/sessions` | 查询会话列表 |
| GET | `/v1/sessions/{session_id}` | 查询单个会话 |
| DELETE | `/v1/sessions/{session_id}` | 删除会话 |
| POST | `/v1/sessions/{session_id}/writeback` | 将高分内容回写知识库 |

### 个人记忆库接口

| Method | Path | Description |
| --- | --- | --- |
| GET | `/v1/memory` | 查询个人记忆 |
| POST | `/v1/memory/save` | 保存问答到个人记忆库 |
| GET | `/v1/memory/export` | 导出记忆 JSON |
| POST | `/v1/memory/import` | 导入记忆 JSON |
| GET | `/v1/memory/{memory_id}` | 获取单条记忆 |
| PUT | `/v1/memory/{memory_id}` | 更新单条记忆 |
| DELETE | `/v1/memory/{memory_id}` | 删除单条记忆 |

## MCP 工具

项目内置 MCP 工具注册机制，当前主要工具包括：

- `sensitive_check`
- `normalize_text`
- `truncate_text`
- `split_paragraphs`
- `retrieve_materials`

启动 MCP Server：

```bash
drama-agent mcp
```

可参考 `mcp.json.example` 接入 Cursor、Claude Desktop 或其他支持 MCP 的客户端。

## 测试

运行单元测试：

```bash
set PYTHONPATH=src && pytest tests/ -v
```

PowerShell：

```powershell
$env:PYTHONPATH = "src"
pytest tests/ -v
```

冒烟测试：

```bash
python smoke_test.py
```

API 测试：

```bash
python scripts/api_test.py http://127.0.0.1:8001
```

## 常见问题

### 端口被占用

错误示例：

```text
[WinError 10048] 通常每个套接字地址只允许使用一次
```

解决方法：

- 关闭正在运行的旧服务。
- 或将端口改为 `8002`：

```bash
python -m uvicorn drama_agent.api:app --host 127.0.0.1 --port 8002
```

### PowerShell 无法激活虚拟环境

如果 `Activate.ps1` 被执行策略拦截，可以使用 CMD 激活：

```cmd
.venv\Scripts\activate
```

或临时允许当前 PowerShell 会话执行脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 页面能打开但保存记忆失败

确认服务进程对项目目录有写权限，尤其是：

```text
data/user_memory/
data/sessions/
```

如果在 IDE、沙箱或权限受限终端中运行，建议改用正常 Windows 终端或 PyCharm 本地解释器运行。

### 检索结果不准确或 FAISS 维度不一致

重新构建知识库索引：

```bash
python scripts/build_knowledge.py --rebuild
```

### 未配置 LLM API Key

系统会进入 Stub 模式。流程仍能跑通，但生成内容是本地演示结果。要获得真实生成效果，请在 `.env` 中配置：

```ini
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...
```

## 开发指南

### 新增 Agent

1. 在 `src/drama_agent/agents/` 下新增实现。
2. 在 `graph.py` 中注册节点。
3. 根据需要配置条件边和 state 字段。
4. 为核心行为添加测试。

### 新增工具

1. 在 `src/drama_agent/tools/` 下新增工具文件。
2. 在工具注册表中注册工具。
3. 如需对外暴露，更新 MCP Server 或 API。

### 修改前端

前端是单文件实现：

```text
frontend/index.html
```

FastAPI 根路径 `/` 会直接读取并返回该文件，因此多数前端修改刷新浏览器即可生效。

## License

MIT License. See [LICENSE](LICENSE).
