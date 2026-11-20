"""LLM-as-a-Judge：语义一致性判定与人工校准（Kappa）。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from invoice_agent.errors import ModelOutputError
from invoice_agent.extractors.base import parse_json_object
from invoice_agent.llm.base import BaseLLMClient

JUDGE_PROMPT_VERSION = "judge-v1"

JUDGE_SYSTEM_PROMPT = """你是票据字段评测裁判。你的任务是判断 prediction 与 ground_truth 是否表达同一事实。

判定规则：
1. 忽略空白、全半角、大小写、货币符号、千分位、日期分隔符的差异。
2. 公司名称允许空格和括号全半角的差异，但法律实体不同必须判 false。
3. 金额按数值比较，数值相等即一致。
4. 发票号码去除空白后必须逐位相同。
5. 只输出 JSON，不要输出解释或 Markdown。

输出格式：
{"fields": {"<字段名>": {"match": true, "reason": "简要理由"}}}"""


def _stringify(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    return "" if value is None else str(value)


@dataclass
class JudgeVerdict:
    matches: dict[str, bool] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    raw: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class LLMAsJudge:
    """用一个强模型对“严格比对失败但可能语义一致”的字段做二次判定。"""

    def __init__(self, client: BaseLLMClient) -> None:
        self._client = client

    @property
    def model_name(self) -> str:
        return self._client.model_name

    def judge_fields(self, diffs: Mapping[str, tuple[Any, Any]]) -> JudgeVerdict:
        if not diffs:
            return JudgeVerdict()

        payload = {
            "fields": {
                name: {
                    "ground_truth": _stringify(ground_truth),
                    "prediction": _stringify(prediction),
                }
                for name, (ground_truth, prediction) in diffs.items()
            }
        }
        response = self._client.chat_text(
            system=JUDGE_SYSTEM_PROMPT,
            user=json.dumps(payload, ensure_ascii=False),
        )
        parsed = parse_json_object(response.content)
        raw_fields = parsed.get("fields")
        if not isinstance(raw_fields, dict):
            raise ModelOutputError("裁判输出缺少 fields 对象")

        matches: dict[str, bool] = {}
        reasons: dict[str, str] = {}
        for name in diffs:
            item = raw_fields.get(name)
            if not isinstance(item, dict):
                matches[name] = False
                reasons[name] = "裁判未返回该字段的判定"
                continue
            matches[name] = bool(item.get("match", False))
            reasons[name] = str(item.get("reason", ""))[:300]

        return JudgeVerdict(
            matches=matches,
            reasons=reasons,
            raw=response.content,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
        )


@dataclass(frozen=True)
class CalibrationCase:
    id: str
    field: str
    ground_truth: str
    prediction: str
    human_label: bool
    note: str = ""


@dataclass
class CalibrationResult:
    total: int
    valid: int
    errors: int
    accuracy: float
    precision: float
    recall: float
    kappa: float
    tp: int
    fp: int
    fn: int
    tn: int
    rows: list[dict[str, Any]]


def load_calibration(path: str | Path) -> list[CalibrationCase]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if file_path.suffix.lower() == ".jsonl":
        entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        entries = json.loads(text)
    if not isinstance(entries, list):
        raise ValueError("校准集必须是 JSON 数组或 JSONL")

    cases: list[CalibrationCase] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"第 {index} 条校准样本不是对象")
        human_label = entry.get("human_label")
        if not isinstance(human_label, bool):
            raise ValueError(f"第 {index} 条校准样本的 human_label 必须是布尔值")
        cases.append(
            CalibrationCase(
                id=str(entry.get("id", f"case-{index}")),
                field=str(entry["field"]),
                ground_truth=str(entry["ground_truth"]),
                prediction=str(entry["prediction"]),
                human_label=human_label,
                note=str(entry.get("note", "")),
            )
        )
    return cases


def cohen_kappa(human: Sequence[bool], judge: Sequence[bool]) -> float:
    if len(human) != len(judge):
        raise ValueError("human 与 judge 标签数量不一致")
    n = len(human)
    if n == 0:
        return 0.0

    observed = sum(1 for a, b in zip(human, judge) if a == b) / n
    human_pos = sum(human) / n
    judge_pos = sum(judge) / n
    expected = human_pos * judge_pos + (1 - human_pos) * (1 - judge_pos)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def run_calibration(
    judge: LLMAsJudge,
    cases: Sequence[CalibrationCase],
    *,
    limit: int | None = None,
) -> CalibrationResult:
    selected = list(cases[:limit] if limit else cases)
    rows: list[dict[str, Any]] = []
    human_labels: list[bool] = []
    judge_labels: list[bool] = []
    errors = 0

    for case in selected:
        try:
            verdict = judge.judge_fields({case.field: (case.ground_truth, case.prediction)})
            judge_label = bool(verdict.matches.get(case.field, False))
            reason = verdict.reasons.get(case.field, "")
            error = None
        except Exception as exc:  # 单条裁判失败不影响整体校准
            judge_label = None
            reason = ""
            error = str(exc)
            errors += 1

        row = {
            "id": case.id,
            "field": case.field,
            "ground_truth": case.ground_truth,
            "prediction": case.prediction,
            "human_label": case.human_label,
            "judge_label": judge_label,
            "reason": reason,
            "error": error,
            "note": case.note,
        }
        rows.append(row)
        if judge_label is not None:
            human_labels.append(case.human_label)
            judge_labels.append(judge_label)

    tp = sum(1 for h, j in zip(human_labels, judge_labels) if h and j)
    fp = sum(1 for h, j in zip(human_labels, judge_labels) if not h and j)
    fn = sum(1 for h, j in zip(human_labels, judge_labels) if h and not j)
    tn = sum(1 for h, j in zip(human_labels, judge_labels) if not h and not j)

    valid = len(human_labels)
    accuracy = (tp + tn) / valid if valid else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return CalibrationResult(
        total=len(selected),
        valid=valid,
        errors=errors,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        kappa=cohen_kappa(human_labels, judge_labels),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        rows=rows,
    )
