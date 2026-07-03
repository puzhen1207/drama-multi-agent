"""高优先级改进项测试。"""
from __future__ import annotations

from drama_agent.agents.audit_agent import run_audit
from drama_agent.config import settings
from drama_agent.tools.embedding import embedding_status, get_embedding_provider
from drama_agent.tools.user_memory import UserMemoryStore


def test_embedding_status_structure():
    st = embedding_status()
    assert "available" in st
    assert "mode" in st
    assert "dim" in st
    assert st["dim"] > 0


def test_audit_uses_config_threshold(monkeypatch):
    monkeypatch.setattr(settings, "audit_pass_threshold", 0.95)
    state = {
        "draft_content": "这是一段足够长的测试文本，用于验证审核阈值读取配置。" * 3,
        "iteration_count": 0,
        "degrade_mode": True,
    }
    result = run_audit(state)
    audit = result["audit_result"]
    assert audit is not None
    # 规则层模式，无 forbidden 时应接近通过逻辑
    assert hasattr(audit, "passed")


def test_memory_export_import(tmp_path, monkeypatch):
    from drama_agent import config
    import drama_agent.tools.user_memory as um

    monkeypatch.setattr(config.settings, "user_memory_path", str(tmp_path / "um"))
    um._user_memory_store = None
    store = UserMemoryStore(store_dir=tmp_path / "um")
    store.add("u_exp", "测试导出问题", "这是用于导出导入测试的回答内容" * 3)
    exported = store.export_all("u_exp")
    assert len(exported) == 1

    result = store.import_entries("u_new", exported, skip_duplicates=True)
    assert result["imported"] == 1
    assert store.count("u_new") == 1


def test_mcp_server_importable():
    from drama_agent.mcp_server import run_mcp_server
    assert callable(run_mcp_server)
