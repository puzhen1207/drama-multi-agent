# 短剧多智能体内容生产系统（drama-multi-agent1）
**v2.1 · 手动调度 + Pydantic V2 + FAISS + FastAPI

## 为什么新建了 drama-multi-agent1？

原 drama-multi-agent 使用 LangGraph 的 StateGraph 做调度，在某些版本中 dict 型 state 在节点之间的 merge/替换规则不一致，容易导致前端 SSE 收到的 `content` 为空字符串，且 degrade_mode 被不正确触发。为了彻底解决这些问题，这里把核心调度从 LangGraph 改为手写的「手动调度（_manual_fallback）」，并对 SSE 事件编码、前端事件解析做了深度的全面加固。

**核心特性：

- 🔍 **任务解析 Agent**：规则 + 可选 LLM 双通道。
- 📚 **素材检索 Agent**：FAISS 向量索引 + 轻量重排。
- ✨ **内容润色 Agent**：参考素材 + 草稿 + 审核反馈 + 会话上下文驱动。
- ✅ **合规审核 Agent**：规则引擎 + 可选 LLM 语义审核双轨机制。
- 🧠 **会话记忆**：最近几轮对话、用户画像（风格、主题、字数）、反思日志。
- 🌊 **SSE 流式可视化**：在前端实时可视化每个节点的生命周期和内容。

## 快速开始

需要 Python 3.10+。

```bash
cd drama-multi-agent1
python -m venv .venv
# Windows 激活:
.venv\Scripts\activate
# 或 Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

# 可选：把 .env.example 复制为 .env 后填入 LLM 配置
cp .env.example .env
# （未配置 LLM 时系统会自动进入本地 stub 模式，流程依然可跑通
#  便于在 PyCharm/VSCode 里进行联调。)

# 启动服务
uvicorn drama_agent.api:app --host 127.0.0.1 --port 8000 --reload
# 或直接：
python -m drama_agent
```

启动之后，浏览器打开：

- 前台页面：http://127.0.0.1:8000/
- Swagger 文档：http://127.0.0.1:8000/docs

## API 一览

| Method | Path | 用途 |
| ---- | ---- | ---- |
| GET    | /                                  | 前端页面 |
| GET    | /health                            | 健康检查 |
| GET    | /v1/tools                          | 已注册 MCP 工具列表 |
| POST   | /v1/generate                       | 阻塞式生成 |
| POST   | /v1/stream                         | SSE 流式生成（前端可视化）|
| POST   | /v1/async/generate               | 异步提交任务 |
| GET    | /v1/async/{task_id}               | 任务查询 |
| GET    | /v1/sessions                      | 会话列表（可选 user_id）|
| GET    | /v1/sessions/{session_id}          | 获取会话详情 |
| DELETE | /v1/sessions/{session_id}           | 删除会话 |
| POST   | /v1/sessions/{session_id}/writeback | 把高分内容回写知识库 |

## 目录结构

```
drama-multi-agent1/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .env.example
├── frontend/
│   └── index.html                      # 前端：可视化 + SSE 可视化
├── data/
│   ├── faiss_index/                     # FAISS 向量索引（首次自动创建）
│   └── sessions/                        # JSON 会话持久化
└── src/
    └── drama_agent/
        ├── __init__.py
        ├── __main__.py
        ├── config.py                       # pydantic-settings 配置
        ├── logging_setup.py                # loguru 日志
        ├── exceptions.py                   # 异常 & retry 装饰器
        ├── models.py                       # Pydantic 模型 + 共享 state
        ├── graph.py                        # 核心调度 + 事件分发 + 会话生命周期
        ├── api.py                          # FastAPI 路由 + SSE 编码
        ├── memory.py                       # 会话管理、用户画像学习
        ├── llm.py                        # LLM 适配（可选）
        ├── agents/
        │   ├── parser_agent.py
        │   ├── retriever_agent.py
        │   ├── polish_agent.py
        │   ├── audit_agent.py
        │   └── prompts.py                  # 四大 Agent 的 Prompt 模板
        └── tools/
            ├── text_processor.py
            ├── compliance_engine.py           # 规则引擎（敏感词、规则校验）
            ├── vector_retriever.py         # FAISS 向量检索 + 索引保存/加载（兼容 Windows 中文路径）
            └── registry.py               # MCP 风格工具注册表（register）
```

## 主要架构思路（相比原 drama-multi-agent）

1. **从 LangGraph 改为手动调度
   - 原来 StateGraph 的 add_node 负责 state 的字段被序列化时会不同版本的 LangGraph 不一致（有些版本把 dict 视为replace，有些版本视为 merge），导致节点之间传 draft_content 丢失；
   - 现在统一在 graph._manual_fallback(ctx, state) 做严格的节点-by-节点的 dict 合并。

2. **SSE 编码：
   - 用 `json.dumps(..., ensure_ascii=False)` 然后 `.encode("utf-8") 编码；
   - 每个事件严格：`event: xxx\ndata: {...}\n\n`；
   - 返回头 `text/event-stream; charset=utf-8`、`Cache-Control: no-cache, no-transform` + `X-Accel-Buffering: no`。

3. **degrade_mode 不再由“知识库为空”错误触发：
   - 知识库为空或检索空返回空列表；
   - 仅当发生不可恢复的错误才设置 degrade_mode；
   - 无 LLM API Key 时进入 stub 的 polish 会用本地模板输出，保证前端有内容。

4. **前端：
   - 正确解析 SSE 事件（`event:` 头 + `data:` JSON payload）；
   - 空内容兜底：若最终 payload 的 content 为空，显示提示；
   - 中文、【…】风格分段渲染。

## 环境变量

拷贝 `.env.example` 到 `.env`，主要字段：

```bash
LLM_API_KEY=sk-...               # 你的大模型 API Key（可选）
LLM_BASE_URL=https://api.deepseek.com/v1   # 可选：豆包/OpenAI 兼容接口
LLM_MODEL=deepseek-chat            # 可选：模型名称
LLM_TEMPERATURE=0.7

# 向量检索
VECTOR_INDEX_PATH=data/faiss_index
MATERIAL_KNOWLEDGE_PATH=data/knowledge
RETRIEVE_TOP_K=20
RERANK_TOP_K=3

# 审核阈值
AUDIT_MAX_ITERATION=3
AUDIT_PASS_THRESHOLD=0.8

# 服务
API_HOST=127.0.0.1
API_PORT=8000
LOG_LEVEL=INFO
```

## 在 PyCharm 里运行

1. 用 PyCharm 打开项目根目录 `drama-multi-agent1`；
2. `File → Settings → Project → Python Interpreter` 选中你本地的 Python；
3. 在终端（PyCharm 自带 Terminal ）：
   ```bash
   pip install -r requirements.txt
   ```
4. 新建一个 Run Configuration：
   - `Module name` 填入 `uvicorn`
   - `Parameters` 填入 `drama_agent.api:app --host 127.0.0.1 --port 8000 --reload`
   - `Working directory` 设为项目根目录；
   - `Environment variables` 里可以把 `PYTHONPATH` 自动推断即可（src 目录结构已支持）。
5. 启动浏览器访问 http://127.0.0.1:8000/。

也可以直接用终端直接：

```bash
cd drama-multi-agent1
python -m drama_agent
```

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
