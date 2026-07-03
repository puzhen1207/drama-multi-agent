"""FastAPI 路由测试。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from drama_agent.api import app, _format_sse


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root_returns_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_list_tools():
    resp = client.get("/v1/tools")
    assert resp.status_code == 200
    tools = resp.json()["tools"]
    assert "sensitive_check" in tools
    assert "retrieve_materials" in tools


def test_generate_sync():
    resp = client.post("/v1/generate", json={
        "raw_input": "写一段短剧开头",
        "user_id": "test",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["data"]["content"]


def test_sse_encoding():
    b = _format_sse("final", {"content": "你好，世界", "score": 0.95})
    assert b.startswith(b"event: final\n")
    assert "你好".encode("utf-8") in b
    assert b"\\u4f60" not in b
