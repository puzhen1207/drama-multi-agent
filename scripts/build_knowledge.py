#!/usr/bin/env python3
"""从 data/knowledge/ 目录构建 FAISS 向量索引。

用法：
    python scripts/build_knowledge.py
    python scripts/build_knowledge.py --rebuild
    python scripts/build_knowledge.py --dir data/knowledge --output data/faiss_index
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def _load_documents(knowledge_dir: Path) -> list[dict]:
    """扫描目录，支持 .txt / .md / .json 格式。"""
    docs: list[dict] = []
    if not knowledge_dir.exists():
        print(f"[WARN] 知识库目录不存在：{knowledge_dir}")
        return docs

    for fp in sorted(knowledge_dir.rglob("*")):
        if not fp.is_file():
            continue
        suffix = fp.suffix.lower()
        if suffix not in (".txt", ".md", ".json"):
            continue

        rel = fp.relative_to(knowledge_dir)
        if suffix == ".json":
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[SKIP] JSON 解析失败 {rel}: {e}")
                continue
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("content"):
                        docs.append({
                            "title": item.get("title", fp.stem),
                            "category": item.get("category", "unknown"),
                            "content": item["content"],
                        })
            elif isinstance(data, dict) and data.get("content"):
                docs.append({
                    "title": data.get("title", fp.stem),
                    "category": data.get("category", "unknown"),
                    "content": data["content"],
                })
        else:
            content = fp.read_text(encoding="utf-8").strip()
            if not content:
                continue
            category = fp.parent.name if fp.parent != knowledge_dir else "unknown"
            docs.append({
                "title": fp.stem.replace("_", " "),
                "category": category,
                "content": content,
            })

    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="构建短剧素材 FAISS 向量索引")
    parser.add_argument("--dir", default="data/knowledge", help="知识库源文件目录")
    parser.add_argument("--output", default="data/faiss_index", help="索引输出目录")
    parser.add_argument("--rebuild", action="store_true", help="清空已有索引后重建")
    args = parser.parse_args()

    knowledge_dir = (ROOT / args.dir).resolve()
    output_dir = (ROOT / args.output).resolve()

    from drama_agent.config import settings
    from drama_agent.tools.vector_retriever import HierarchicalVectorStore

    settings.vector_index_path = str(output_dir.relative_to(ROOT))

    docs = _load_documents(knowledge_dir)
    print(f"[INFO] 扫描到 {len(docs)} 份文档（目录：{knowledge_dir}）")

    if not docs:
        print("[ERROR] 没有找到可用文档。请在 data/knowledge/ 下放置 .txt / .md / .json 文件。")
        sys.exit(1)

    if args.rebuild and output_dir.exists():
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"[INFO] 已清空索引目录：{output_dir}")

    vs = HierarchicalVectorStore(index_dir=output_dir)
    vs.add_documents(docs)
    vs.save()

    stats = vs.count()
    print(f"[OK] 索引构建完成：父块={stats['parents']}，子块={stats['sub_blocks']}")
    print(f"[OK] 索引路径：{output_dir}")


if __name__ == "__main__":
    main()
