"""个人记忆库测试。"""
from __future__ import annotations

import pytest

from drama_agent.tools.user_memory import UserMemoryStore, search_user_memory


@pytest.fixture
def memory_store(tmp_path, monkeypatch):
    from drama_agent import config

    monkeypatch.setattr(config.settings, "user_memory_path", str(tmp_path / "user_memory"))
    monkeypatch.setattr(config.settings, "enable_user_memory", True)
    monkeypatch.setattr(config.settings, "user_memory_top_k", 2)
    import drama_agent.tools.user_memory as um

    um._user_memory_store = None
    store = UserMemoryStore(store_dir=tmp_path / "user_memory")
    return store


def test_add_and_search(memory_store):
    memory_store.add("user_a", "霸总追妻短剧大纲", "第一集：女主被误会…" * 5)
    memory_store.add("user_a", "重生80年代当首富", "他醒来发现自己回到了1980年…" * 5)
    memory_store.add("user_b", "其他用户问题", "其他用户回答内容…" * 5)

    hits = memory_store.search("user_a", "霸总追妻 大纲", top_k=2)
    assert hits
    assert all(h.source == "user_memory" for h in hits)
    assert any("霸总" in h.content or "追妻" in h.content for h in hits)

    hits_b = memory_store.search("user_b", "霸总追妻", top_k=2)
    assert all("其他用户" in h.content for h in hits_b)


def test_list_and_delete(memory_store):
    mid = memory_store.add("u1", "测试问题", "测试回答内容足够长" * 3)
    assert memory_store.count("u1") == 1
    items = memory_store.list_memories("u1")
    assert len(items) == 1
    assert items[0]["memory_id"] == mid

    assert memory_store.delete(mid, "u1") is True
    assert memory_store.count("u1") == 0


def test_search_user_memory_helper(memory_store, monkeypatch):
    import drama_agent.tools.user_memory as um

    um._user_memory_store = memory_store
    memory_store.add("helper_user", "短剧合规要求", "禁止政治敏感、色情低俗…" * 3)
    results = search_user_memory("helper_user", "合规 红线", top_k=1)
    assert len(results) == 1
    assert results[0]["source"] == "user_memory"


def test_add_validation(memory_store):
    with pytest.raises(ValueError):
        memory_store.add("u", "x", "太短")
    with pytest.raises(ValueError):
        memory_store.add("u", "", "足够长的回答内容" * 3)


def test_get_and_update(memory_store):
    mid = memory_store.add("u1", "原始问题内容", "原始回答内容足够长" * 3, title="标题A")
    got = memory_store.get(mid, "u1")
    assert got is not None
    assert got["question"] == "原始问题内容"
    assert "原始回答" in got["answer"]

    memory_store.update(mid, "u1", question="修改后的问题", answer="修改后的回答内容足够长" * 3, title="新标题")
    updated = memory_store.get(mid, "u1")
    assert updated["question"] == "修改后的问题"
    assert updated["title"] == "新标题"
    assert updated["updated_ts"] >= updated["created_ts"]

    hits = memory_store.search("u1", "修改后的问题", top_k=1)
    assert hits and hits[0].material_id == mid
