"""全局配置管理 — 基于 pydantic-settings，从 .env / 环境变量读取。
敏感字段用 SecretStr 包装，避免日志/print 泄露。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


# 项目根目录：优先使用 DRAMA_ROOT 环境变量，其次按项目结构推断
def _resolve_project_root() -> Path:
    env_root = os.environ.get("DRAMA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    try:
        candidate = Path(__file__).resolve().parents[2]
        for marker in ("pyproject.toml", ".env.example", "frontend", "requirements.txt"):
            if (candidate / marker).exists():
                return candidate
    except Exception:
        pass
    return Path.cwd()


PROJECT_ROOT = _resolve_project_root()


class Settings(BaseSettings):
    """应用全局配置。字段大写会映射到环境变量（LLM_API_KEY → llm_api_key）。"""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------- LLM ----------
    llm_api_key: SecretStr = Field(default=SecretStr(""), description="大模型 API Key")
    llm_base_url: str = Field(default="https://api.deepseek.com/v1")
    llm_model: str = Field(default="deepseek-chat")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    llm_timeout: int = Field(default=120, ge=5)
    llm_max_retries: int = Field(default=3, ge=0)

    # ---------- Embedding ----------
    embedding_provider: str = "local"
    embedding_api_key: SecretStr = SecretStr("")
    embedding_base_url: str = ""
    embedding_model: str = "bge-large-zh"
    embedding_dim: int = 384

    # ---------- 向量检索 ----------
    vector_index_path: str = "data/faiss_index"
    material_knowledge_path: str = "data/knowledge"
    retrieve_top_k: int = Field(default=20, ge=1)
    rerank_top_k: int = Field(default=3, ge=1)
    enable_rerank: bool = True

    # ---------- 合规审核 ----------
    sensitive_words_path: Optional[str] = None
    audit_max_iteration: int = Field(default=3, ge=1, le=10)
    audit_pass_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    # ---------- 服务 ----------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"

    # ---------- 运行时属性（基于项目根目录）----------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def absolute_vector_index_path(self) -> Path:
        p = PROJECT_ROOT / self.vector_index_path
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def absolute_knowledge_path(self) -> Path:
        p = PROJECT_ROOT / self.material_knowledge_path
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def absolute_sensitive_words_path(self) -> Optional[Path]:
        if not self.sensitive_words_path:
            return None
        p = PROJECT_ROOT / self.sensitive_words_path
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()


def describe() -> Settings:
    """方便从任意模块调用，避免全局单例。"""
    return settings
