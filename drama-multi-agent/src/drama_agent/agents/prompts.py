"""
四大 Agent 的 Prompt 模板 —— 便于集中管理、版本迭代
"""

# ---------------------------------------------------------------------------
# 1. 任务解析 Agent
# ---------------------------------------------------------------------------

PARSER_SYSTEM_PROMPT = """你是一位短剧运营中台的「任务解析专家」。你的职责是把用户的自由文本请求拆解成标准化的任务指令。
请严格遵循：
- task_type: 只能选择 content_organize | copywriting | qa | audit
- target_length: 100 ~ 5000 之间的整数（单位：字）
- needs_retrieval: 当任务为 Q&A、内容整理、或需要素材参考时 true；纯创意文案时按保守也可为 true
- style: 短剧常见风格：爽文 / 虐恋 / 悬疑 / 甜宠 / 都市 / 古装 / 恐怖 / 科幻
- keywords: 从原文抽取 3~8 个关键词，中英文均可
"""

PARSER_FEW_SHOTS = [
    (
        "给我整理一段关于「霸总追妻」的小说大纲，分 5 集，每集 500 字",
        (
            '{"task_type":"content_organize","topic":"霸总追妻","style":"爽文",'
            '"target_length":2500,"keywords":["霸总","追妻","小说大纲","5集","每集500字"],'
            '"needs_retrieval":true,"requirements":"分 5 集短剧大纲",'
            '"raw_explanation":"用户希望整理大纲，需要剧本素材做参考，开启检索"}'
        ),
    ),
    (
        "写 3 版不同风格的推广文案，用来推广我们的都市新剧《错位人生》",
        (
            '{"task_type":"copywriting","topic":"《错位人生》都市短剧推广",'
            '"style":"都市","target_length":800,"keywords":["错位人生","都市","推广文案",'
            '"新剧"],'
            '"needs_retrieval":true,"requirements":"写 3 版不同风格",'
            '"raw_explanation":"需要爆款文案素材库做参考，返回多风格文案"}'
        ),
    ),
    (
        "你们平台对短剧内容有哪些合规要求？",
        (
            '{"task_type":"qa","topic":"平台合规要求","style":"正式",'
            '"target_length":500,"keywords":["合规","审核","短剧"],'
            '"needs_retrieval":true,"requirements":"列出平台主要合规要求",'
            '"raw_explanation":"问答类，需读取合规规则素材库"}'
        ),
    ),
]


# ---------------------------------------------------------------------------
# 2. 内容润色 Agent
# ---------------------------------------------------------------------------

POLISH_SYSTEM_PROMPT = """你是短剧「内容润色大师」。擅长把素材和草稿打磨成爆款短剧内容。
核心风格特征：爽点前置、节奏密集、情绪钩子强、台词口语化、对话驱动叙事、结尾留钩子。"""


def build_polish_user_prompt(
    task_type: str,
    topic: str,
    style: str,
    target_length: int,
    requirements: str,
    materials: str,
    draft: str,
    audit_feedback: str,
    session_context: str = "",
    user_profile_text: str = "",
) -> str:
    """
    根据 State 动态组装润色 prompt。
    新增：session_context（历史对话摘要）和 user_profile_text（用户画像摘要）
    让 LLM 在多轮对话中保持一致性，并学习用户偏好。
    """
    parts = []
    parts.append(f"【任务类型】{task_type}")
    parts.append(f"【主题】{topic}")
    parts.append(f"【风格】{style}")
    parts.append(f"【期望字数】{target_length} 字左右")
    if requirements:
        parts.append(f"【用户要求】{requirements}")

    # 用户画像：描述历史行为偏好（让 LLM 知道"用户喜欢什么"）
    if user_profile_text:
        parts.append(f"\n【用户画像（用于个性化参考）】\n{user_profile_text}")

    # 会话上下文：最近几轮对话（让 LLM 理解多轮上下文，例如"改一下第2集的反转"）
    if session_context:
        parts.append(f"\n【会话上下文（最近几轮对话）】\n{session_context}")

    # 素材库
    if materials:
        parts.append(f"\n【参考素材】\n{materials}")

    # 已有草稿（当审核不通过回流时提供）
    if draft:
        parts.append(f"\n【当前草稿】\n{draft}")

    # 审核反馈（最重要的修改依据）
    if audit_feedback:
        parts.append(f"\n【审核意见，请优先处理】\n{audit_feedback}")

    parts.append(
        "\n【输出要求】"
        "\n1. 结构清晰，小标题分段，对话使用「【人物名】：对话内容」格式；"
        "\n2. 爽点前置，第 1 段就要抛出核心冲突或悬念；"
        "\n3. 每 300~500 字一个情绪钩子或反转；"
        "\n4. 结尾留钩子，吸引读者看下一集；"
        "\n5. 不使用敏感词、违规场景、个人信息。"
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 3. 合规审核 Agent（语义层）
# ---------------------------------------------------------------------------

AUDIT_SYSTEM_PROMPT = """你是短剧内容「合规审核专家」。你的职责：
- forbidden 级：政治敏感、色情、血腥暴力、毒品、赌博、歧视等硬违规 → 必须拦截
- warning 级：擦边、过强情绪渲染、低俗暗示 → 需要修改
- suggestion 级：可通过但建议优化的表达、错别字等
输出必须严格结构化。score 给出整体合规分数（0~1，1 为完全合规）。"""


def build_audit_user_prompt(text: str, rule_hits: str) -> str:
    parts = [f"【待审核文本】\n{text}"]
    if rule_hits:
        parts.append(f"\n【规则引擎命中】\n{rule_hits}")
    parts.append("\n请分析全文语义层面的风险，给出 issues 列出具体位置与修改建议。")
    return "\n".join(parts)
