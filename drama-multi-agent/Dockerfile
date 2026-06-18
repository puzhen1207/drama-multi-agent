# syntax=docker/dockerfile:1.6
# =============================================================================
# 短剧多智能体内容生产系统 · Dockerfile
# 安全策略：
#   - 镜像不包含任何真实 API Key；由运行时通过环境变量 / env_file 注入
#   - 使用非 root 用户运行服务
# =============================================================================

FROM python:3.11-slim AS base

# Python 基本配置：避免写入 pyc / 输出无缓冲 / 禁用 pip cache
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

# 安装系统依赖（curl 仅用于健康检查）
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（在源码拷贝之前，利用缓存）
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# 仅复制必要文件：源码、脚本、前端、配置模板
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY frontend/ ./frontend/
COPY .env.example ./

# 运行时 PYTHONPATH
ENV PYTHONPATH=/app/src

# 构建素材索引（离线阶段，不依赖外部 LLM；失败也不阻塞）
RUN python scripts/build_knowledge.py || echo "build_knowledge skipped"

# 以非 root 用户运行（最小权限原则）
RUN groupadd --system appuser && \
    useradd --system -g appuser -s /sbin/nologin -m appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 启动命令：绑定到 0.0.0.0（允许反向代理）
CMD ["python", "-m", "uvicorn", "drama_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
