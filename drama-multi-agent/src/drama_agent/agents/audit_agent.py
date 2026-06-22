"""合规审核 Agent —— 规则引擎 + 语义审核双轨机制。"""
from __future__ import annotations

from typing import Any, Dict, List

from ..exceptions import with_retry
from ..llm import chat_structured, llm_available
from ..logging_setup import get_logger
from ..models import AuditIssue, AuditResult
from ..tools.compliance_engine import tool_sensitive_check
from .prompts import AUDIT_SYSTEM_PROMPT, build_audit_user_prompt

logger = get_logger("audit_agent")


@with_retry
def run_audit(state: Dict[str, Any]) -> Dict[str, Any]:
    text = state.get("draft_content") or ""
    iteration = int(state.get("iteration_count", 0) or 0)
    degrade_mode = bool(state.get("degrade_mode"))

    logger.info(f"[Audit] 审核文本 {len(text)} 字，iter={iteration}, degrade={degrade_mode}")

    if len(text.strip()) < 10:
        logger.warning("[Audit] 文本过短，直接返回需修改")
        return {"audit_result": AuditResult(
            passed=False,
            score=0.3,
            issues=[AuditIssue(level="warning", category="内容过短",
                                position="全文",
                                suggestion="请提供更有信息量的内容（至少 10 字）。")],
            summary="文本过短，无法进行完整的合规审核",
            degrade_mode=True,
        ), "iteration_count": iteration + 1}

    # 规则层
    rule_result = tool_sensitive_check(text)
    forbidden_hits: List[dict] = rule_result.get("forbidden", [])
    warning_hits: List[dict] = rule_result.get("warning", [])
    suggestion_hits: List[dict] = rule_result.get("suggestion", [])
    rule_engine_hit = bool(forbidden_hits)

    issues: List[AuditIssue] = []
    for h in forbidden_hits:
        issues.append(AuditIssue(
            level="forbidden",
            category=h.get("category", "硬违规"),
            position=f"{h.get('snippet','')}({h.get('keyword','')})",
            suggestion="立即删除或改写本段，替换为合规表达。",
        ))
    for h in warning_hits:
        issues.append(AuditIssue(
            level="warning",
            category=h.get("category", "软违规"),
            position=f"{h.get('snippet','')}({h.get('keyword','')})",
            suggestion="建议改写成中性/合规表述。",
        ))
    for h in suggestion_hits:
        issues.append(AuditIssue(
            level="suggestion",
            category=h.get("category", "优化建议"),
            position=h.get("snippet", ""),
            suggestion="可优化表达使其更合规/更生动。",
        ))

    # 语义层（LLM 可用且未强制降级）
    passed_by_rule = not rule_engine_hit
    if llm_available() and not degrade_mode:
        rule_hits_text = _format_rule_hits(
            forbidden_hits + warning_hits + suggestion_hits
        )
        user_prompt = build_audit_user_prompt(text[:3000], rule_hits_text)
        try:
            semantic = chat_structured(pydantic_cls=AuditResult,
                                       user_prompt=user_prompt,
                                       system_prompt=AUDIT_SYSTEM_PROMPT)
            if isinstance(semantic.issues, list):
                for issue in semantic.issues:
                    try:
                        if isinstance(issue, AuditIssue):
                            issues.append(issue)
                        elif isinstance(issue, dict):
                            issues.append(AuditIssue(**issue))
                    except Exception:
                        pass
            score = (float(semantic.score) + (1.0 if passed_by_rule else 0.0)) / 2.0
            summary_parts: List[str] = []
            if rule_engine_hit:
                summary_parts.append("规则层命中硬违规，必须修改")
            if getattr(semantic, "summary", ""):
                summary_parts.append(str(semantic.summary))
            summary = "；".join(summary_parts) or "双轨审核完成"
            passed = (score >= 0.8) and not rule_engine_hit
            result = AuditResult(
                passed=passed,
                score=round(float(score), 3),
                issues=issues,
                summary=summary,
                rule_engine_hit=rule_engine_hit,
                degrade_mode=False,
            )
        except Exception as e:
            logger.warning(f"[Audit] 语义审核失败，仅保留规则层结果：{e}")
            result = _rule_only_result(issues, passed_by_rule, rule_engine_hit, degrade=False)
    else:
        result = _rule_only_result(issues, passed_by_rule, rule_engine_hit, degrade=True)

    logger.info(
        f"[Audit] 完成: passed={result.passed}, score={result.score}, "
        f"issues={len(result.issues)}, degrade={result.degrade_mode}"
    )
    return {"audit_result": result, "iteration_count": iteration + 1}


def _format_rule_hits(hits: List[dict]) -> str:
    if not hits:
        return "（规则层无命中）"
    lines: List[str] = []
    for h in hits[:10]:
        level = h.get("level", "")
        category = h.get("category", "")
        keyword = h.get("keyword", "")
        snippet = h.get("snippet", "")
        lines.append(f"- [{level}] {category}: {keyword} -> ...{snippet}...")
    return "\n".join(lines)


def _rule_only_result(issues: List[AuditIssue], passed_by_rule: bool,
                      rule_hit: bool, degrade: bool) -> AuditResult:
    if rule_hit:
        score = 0.3
    elif issues:
        score = max(0.5, 1.0 - 0.05 * len(issues))
    else:
        score = 0.95
    summary = (
        f"规则层审核：{'无硬违规' if not rule_hit else '存在硬违规，请修改'}；"
        f"共命中 {len(issues)} 项"
    )
    return AuditResult(
        passed=passed_by_rule and not issues,
        score=round(float(score), 3),
        issues=issues,
        summary=summary,
        rule_engine_hit=rule_hit,
        degrade_mode=degrade,
    )
