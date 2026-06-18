"""
构建素材知识库：
- 从 scripts/knowledge_samples.py 内置的语料构建；或把你自己的 JSON 文件放在 data/knowledge/ 下
- 每个文档结构：{
    "title": "...",
    "content": "...",
    "category": "剧本" | "文案" | "人设" | "规则"
}
用法： python scripts/build_knowledge.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 把 src 目录加入 sys.path，便于直接 `python scripts/build_knowledge.py`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drama_agent.config import settings  # noqa: E402
from drama_agent.logging_setup import setup_logging, get_logger  # noqa: E402
from drama_agent.tools.vector_retriever import HierarchicalVectorStore  # noqa: E402

setup_logging()
logger = get_logger("build_knowledge")


# =============================================================================
# 内置语料：覆盖短剧常见场景
# =============================================================================

BUILTIN_DOCS = [
    # ---- 剧本类 ----
    {
        "title": "短剧「爽文」剧本结构模板",
        "category": "剧本",
        "content": """
第一幕：开场冲突。用 300 字交代主角身份、所处困境，制造强烈情绪钩子。
第二幕：反转升级。引入关键配角或外力，让局势反复反转，保持高密度节奏。
第三幕：高潮+钩子。冲突达到顶点，以巨大悬念结尾，吸引读者看下一集。
核心技巧：爽点前置、对话驱动叙事、每 500 字一个情绪钩子。
""".strip(),
    },
    {
        "title": "霸总追妻剧本样例片段",
        "category": "剧本",
        "content": """
【林晚】：(皱眉) 我们已经离婚了，请你不要再来找我。
【顾言】：(冷笑) 离婚协议我没签字。你以为你跑得掉？
【林晚】：顾总，三年前是你亲手把我送进监狱的。
【顾言】：(愣) 那是…… 是误会。
【林晚】：(笑，带着泪) 误会？好，那我也误会你一次。今晚，你滚出我的生活。
——结尾钩子：第二天清晨，顾言在林晚的公司楼下守了整整一夜，手里拿着一份从未公开的遗嘱。
""".strip(),
    },
    {
        "title": "都市悬疑短剧：《消失的证人》大纲",
        "category": "剧本",
        "content": """
第 1 集：案件开场。女警接到一通匿名电话，声称目睹了一起谋杀，随后通话者消失。
第 2 集：调查深入。女警发现自己童年福利院与此案有关，关键证人均在同一福利院长大。
第 3 集：真相与反转。真凶竟是她最信任的导师，为掩盖当年的一桩罪行而灭口。
结尾钩子：导师留下一张小女孩的照片，女警意识到自己也是被选中的证人之一。
""".strip(),
    },
    # ---- 文案类 ----
    {
        "title": "爆款推广文案 标题公式",
        "category": "文案",
        "content": """
公式 1：【强烈反差】她被豪门抛弃三年，归来时身价十亿。
公式 2：【悬念提问】如果你一觉醒来回到高考前一天，你会做什么？
公式 3：【数字冲击】3 天破亿播放，这部短剧凭什么爆？
公式 4：【利益前置】看一集上头，看三集上瘾，这部短剧真的不能错过。
""".strip(),
    },
    {
        "title": "都市爆款文案样例",
        "category": "文案",
        "content": """
标题：他花了三年，只为让前妻在他面前跪下。

正文：
林晚从来没想过，离婚后第一次见顾言，会是在他的订婚宴上。
她端着酒杯，笑着走上前：「祝你们幸福。」
顾言却一把攥住她的手腕，眼神冰冷：「你敢试试？」
下一秒，全场哗然——顾言的手指划过她的无名指，那里戴着他三年前送给她的婚戒。
""".strip(),
    },
    # ---- 人设类 ----
    {
        "title": "短剧「霸总」人设模板",
        "category": "人设",
        "content": """
身份：30 岁左右的企业总裁 / CEO / 家族继承人
外貌：身材挺拔，五官深邃，气场压人
性格：外冷内热，控制欲强，对女主专一，有不可告人的过去
动机：弥补当年对女主的伤害 / 复仇 / 守护家庭
核心张力：霸道的控制 vs 隐藏的深情；强势 vs 脆弱
""".strip(),
    },
    {
        "title": "短剧「逆袭女主」人设模板",
        "category": "人设",
        "content": """
身份：职场新人 / 落魄千金 / 重生归来者
外貌：清秀耐看，后期气质逆袭
性格：外柔内刚，心思缜密，善于隐忍，关键时刻爆发
动机：复仇 / 守护家人 / 证明自我
核心张力：柔弱外壳 vs 强悍内核；被欺 vs 逆袭
""".strip(),
    },
    # ---- 合规规则类 ----
    {
        "title": "短剧行业合规红线（硬违规）",
        "category": "规则",
        "content": """
1. 禁止政治敏感内容、国家领导人姓名与相关符号。
2. 禁止色情低俗、淫秽暗示、床戏赤裸描写。
3. 禁止血腥暴力、砍杀虐杀、自残自杀的详细描写。
4. 禁止毒品、赌博、邪教、恐怖主义相关内容。
5. 禁止歧视性言论（种族、地域、性别、残障等）。
6. 禁止暴露个人信息（身份证、手机号、地址）。
7. 禁止未成年人出现成人化、低俗化剧情。
""".strip(),
    },
    {
        "title": "短剧「警告级」问题清单",
        "category": "规则",
        "content": """
1. 擦边性暗示（过于亲密的动作/称呼）
2. 暴力情节的过度渲染
3. 粗口、脏话（可替换为情绪表达）
4. 惊悚恐怖画面（适合时段、年龄分级提示）
5. 价值观扭曲（宣扬极端利己、报复、唯利是图）
建议：改写为中性或合规表达，保留核心情节但降低刺激度。
""".strip(),
    },
]


def _load_from_dir() -> list:
    """读取 data/knowledge 下的 .json 文件。"""
    kb_dir = settings.absolute_knowledge_path
    docs = []
    if kb_dir.exists():
        for fp in sorted(kb_dir.rglob("*.json")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    docs.extend(data)
                elif isinstance(data, dict) and "content" in data:
                    docs.append(data)
                logger.info(f"已加载 {fp.name}")
            except Exception as e:
                logger.warning(f"加载 {fp.name} 失败: {e}")
    return docs


def main() -> int:
    all_docs = list(BUILTIN_DOCS) + _load_from_dir()
    logger.info(f"开始构建，共 {len(all_docs)} 份文档")
    store = HierarchicalVectorStore()
    store.add_documents(all_docs)
    store.save()
    logger.info(f"构建完成，索引位于 {settings.absolute_vector_index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
