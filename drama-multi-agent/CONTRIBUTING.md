# 贡献指南

感谢你对本项目的关注！以下是参与贡献的几条小规则。

## 环境搭建

```bash
# 克隆项目并安装依赖（Python 3.11+）
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -e ".[dev]"
# 首次部署素材库
python scripts/build_knowledge.py
```

## 代码风格

- 使用 [ruff](https://github.com/astral-sh/ruff) 作为主要 linter/formatter
- 命令：`ruff check src tests`、`ruff format --diff src tests`
- 行宽 120 列，import 按字母序分组（标准库 / 第三方 / 内部）
- 模块顶部写 docstring，函数/方法 30 行以上鼓励写 docstring

## 运行测试

```bash
python -m pytest tests/unit -v            # 单元测试（不需要 LLM）
python -m pytest tests/ -v --cov=src/drama_agent --cov-report=term-missing
```

- `tests/unit/`：不依赖外部服务，CI 必须全部通过
- `tests/`：集成测试（test_workflow.py 会访问 LLM），可本地 `pytest tests/` 验证

## 提交 MR / PR 前

- `ruff check src tests` 无 error 级别问题
- 新增功能请写对应单元测试（tests/unit/）
- 保持 README.md 与实际代码同步（API / 配置项变更尤其重要）
- commit 信息建议使用 [Conventional Commits](https://www.conventionalcommits.org/zh-cn/v1.0.0/)

## 目录结构

```
drama-multi-agent/
├── src/drama_agent/        # 主包
│   ├── agents/             # 4 大 Agent（解析/检索/润色/审核）
│   ├── tools/              # MCP 工具注册中心（敏感词/向量检索/文本规范化）
│   ├── api.py              # FastAPI 入口（同步/异步/SSE 流式）
│   ├── graph.py            # LangGraph 调度与事件发射
│   ├── config.py           # Pydantic-Settings 配置
│   ├── llm.py              # 大模型 HTTP 客户端（OpenAI 兼容）
│   └── models.py           # 全局数据模型
├── frontend/               # 可视化前端（单文件 HTML + 原生 JS）
├── scripts/                # 构建知识库、API 测试脚本
├── tests/                  # pytest 测试
├── data/                   # 运行时向量索引（不上传到 git）
├── Dockerfile / docker-compose.yml
└── pyproject.toml          # 项目元数据 + lint/测试配置
```

## 设计原则

1. **Agent 职责不重叠**：新增 Agent 前先判断是否应放到 tools/ 中（工具是无状态的函数，Agent 才会写 state）
2. **不硬编码敏感信息**：密钥/URL 一律走 `.env` 或环境变量，不要写入 git
3. **降级优先**：任何节点故障都不应中断整个工作流，默认走 stub/纯生成模式
4. **可测试**：所有路由/业务判断函数写成纯函数（如 `_route_after_*`），方便单测
