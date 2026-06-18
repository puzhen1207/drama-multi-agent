"""API 单元测试：使用 FastAPI TestClient（不启动真实服务）。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402
from drama_agent.api import app  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "uptime_seconds" in data


def test_tools(client):
    resp = client.get("/v1/tools")
    assert resp.status_code == 200
    data = resp.json()
    assert "tools" in data
    assert isinstance(data["tools"], list)
    # 至少包含敏感词与文本规范化相关工具
    names = [t for t in data["tools"] if isinstance(t, str)]
    assert any("sensitive" in n or "normalize" in n or "retrieve" in n for n in names)


def test_generate_empty_input_is_422(client):
    """空输入应被 Pydantic BaseModel 校验拒绝。"""
    resp = client.post("/v1/generate", json={"raw_input": ""})
    assert resp.status_code == 422


def test_generate_basic_request_ok(client):
    """一个短请求应该能跑通完整工作流（至少不抛异常）。"""
    resp = client.post("/v1/generate", json={
        "raw_input": "给我整理关于「测试项目」的短剧大纲",
        "user_id": "pytest",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    payload = data["data"]
    assert "content" in payload
    assert "iteration_count" in payload
    assert "degrade_mode" in payload
    # content 必须是字符串；长度可能为 0（stub 模式），但不报错即通过
    assert isinstance(payload["content"], str)


def test_async_flow(client):
    """异步提交 → 查询（即便服务已经完成，也应返回 ok / failed 状态）。"""
    resp = client.post("/v1/async/generate", json={
        "raw_input": "生成一句测试文案",
        "user_id": "pytest",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert "task_id" in body
    task_id = body["task_id"]
    # 查询状态（无论 pending / ok 都应返回）
    resp2 = client.get(f"/v1/async/{task_id}")
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2["task_id"] == task_id
    assert body2["status"] in {"pending", "ok", "failed"}


def test_async_unknown_task_is_404(client):
    resp = client.get("/v1/async/nonexistent-12345")
    assert resp.status_code == 404


def test_root_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
