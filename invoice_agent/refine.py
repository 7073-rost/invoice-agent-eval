"""Prompt 自迭代：基于 Bad Case 反思新 Prompt，并通过回归测试决定是否采纳。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from invoice_agent.errors import ModelOutputError
from invoice_agent.extractors.base import parse_json_object
from invoice_agent.llm.base import BaseLLMClient
from invoice_agent.prompts import PromptTemplate

REFINE_PROMPT_VERSION = "refine-v1"

REFINE_SYSTEM_PROMPT = """你是票据信息抽取的 Prompt 优化专家。你会收到当前 System Prompt 和一批失败样例。
请分析失败原因，并给出改进后的 System Prompt。

约束：
1. 输出字段必须保持为 invoice_number、issue_date、buyer_name、seller_name、total_amount。
2. 只能输出 JSON，不要输出 Markdown 或解释。
3. new_prompt 必须是完整、可直接替换的 System Prompt。
4. 针对具体失败类型补充规则或 Few-shot，但不要编造与票据无关的内容。

输出格式：
{"analysis": "失败原因分析", "new_prompt": "完整的新 System Prompt"}"""


@dataclass
class PromptProposal:
    analysis: str
    new_prompt: str
    raw: str


@dataclass
class RegressionDecision:
    accepted: bool
    reason: str
    baseline_exact_match: float
    candidate_exact_match: float
    regressed_files: list[str]
    improved_files: list[str]


def propose_prompt(
    client: BaseLLMClient,
    current_prompt: PromptTemplate,
    bad_cases: list[dict[str, Any]],
    *,
    max_examples: int = 10,
) -> PromptProposal:
    examples = []
    for case in bad_cases[:max_examples]:
        raw_output = case.get("raw_output")
        if isinstance(raw_output, str) and len(raw_output) > 600:
            raw_output = raw_output[:600] + "...(截断)"
        examples.append(
            {
                "file": case.get("file"),
                "ground_truth": case.get("ground_truth"),
                "prediction": case.get("prediction"),
                "failure_type": case.get("failure_type"),
                "failure_detail": case.get("failure_detail"),
                "error": case.get("error"),
                "raw_output": raw_output,
            }
        )

    payload = {
        "current_prompt_version": current_prompt.version,
        "current_system_prompt": current_prompt.system,
        "bad_cases": examples,
    }
    response = client.chat_text(
        system=REFINE_SYSTEM_PROMPT,
        user=json.dumps(payload, ensure_ascii=False, indent=2),
    )
    parsed = parse_json_object(response.content)
    new_prompt = str(parsed.get("new_prompt") or "").strip()
    if not new_prompt:
        raise ModelOutputError("反思结果缺少 new_prompt")
    return PromptProposal(
        analysis=str(parsed.get("analysis") or "").strip(),
        new_prompt=new_prompt,
        raw=response.content,
    )


def decide_acceptance(
    baseline_cases: list[dict[str, Any]],
    candidate_cases: list[dict[str, Any]],
) -> RegressionDecision:
    baseline = {
        str(case["file"]): bool(case["exact_match"])
        for case in baseline_cases
        if case.get("exact_match") is not None
    }
    candidate = {
        str(case["file"]): bool(case["exact_match"])
        for case in candidate_cases
        if case.get("exact_match") is not None
    }
    if not baseline:
        return RegressionDecision(
            accepted=False,
            reason="基线没有可比对的样本",
            baseline_exact_match=0.0,
            candidate_exact_match=0.0,
            regressed_files=[],
            improved_files=[],
        )

    baseline_rate = sum(baseline.values()) / len(baseline)
    candidate_rate = sum(candidate.get(name, False) for name in baseline) / len(baseline)
    regressed = [
        name for name, correct in baseline.items() if correct and not candidate.get(name, False)
    ]
    improved = [
        name for name, correct in baseline.items() if not correct and candidate.get(name, False)
    ]

    if regressed:
        accepted = False
        reason = (
            f"出现 {len(regressed)} 个回归样本（原本正确变为错误），拒绝新 Prompt"
        )
    elif candidate_rate <= baseline_rate:
        accepted = False
        reason = "整单完全正确率未提升，拒绝新 Prompt"
    else:
        accepted = True
        reason = "整单完全正确率提升且没有回归，接受新 Prompt"

    return RegressionDecision(
        accepted=accepted,
        reason=reason,
        baseline_exact_match=round(baseline_rate, 4),
        candidate_exact_match=round(candidate_rate, 4),
        regressed_files=regressed,
        improved_files=improved,
    )
