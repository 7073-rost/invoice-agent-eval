"""批处理评测：并发跑批、严格比对、LLM Judge 复核与失败归因。"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from invoice_agent.config import Settings, get_settings
from invoice_agent.judge import LLMAsJudge
from invoice_agent.pricing import estimate_cost
from invoice_agent.prompts import DEFAULT_PROMPT, PromptTemplate
from invoice_agent.router import ClientFactory, SUPPORTED_SUFFIXES, InvoiceRouter
from invoice_agent.schemas import InvoiceData
from invoice_agent.tracing import (
    ExtractionOutcome,
    FailureType,
    TraceLogger,
    classify_exception,
)

INVOICE_FIELDS: tuple[str, ...] = (
    "invoice_number",
    "issue_date",
    "buyer_name",
    "seller_name",
    "total_amount",
)

ProgressCallback = Callable[[int, int], None]


def _stringify(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    return "null" if value is None else str(value)


@dataclass
class CaseEvaluation:
    file: str
    ground_truth: InvoiceData | None
    prediction: InvoiceData | None
    strict_matches: dict[str, bool]
    matches: dict[str, bool]
    exact_match: bool | None
    failure_type: FailureType
    failure_detail: str
    model: str
    prompt_version: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    extraction_cost_cny: float
    judge_used: bool = False
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_cost_cny: float = 0.0
    judge_latency_ms: float = 0.0
    judge_error: str | None = None
    judge_reasons: dict[str, str] = field(default_factory=dict)
    raw_output: str | None = None
    error: str | None = None

    @property
    def total_cost_cny(self) -> float:
        return self.extraction_cost_cny + self.judge_cost_cny

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        payload = {
            "file": self.file,
            "exact_match": self.exact_match,
            "failure_type": self.failure_type.value,
            "failure_detail": self.failure_detail,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "latency_ms": round(self.latency_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "extraction_cost_cny": round(self.extraction_cost_cny, 6),
            "judge_used": self.judge_used,
            "judge_input_tokens": self.judge_input_tokens,
            "judge_output_tokens": self.judge_output_tokens,
            "judge_cost_cny": round(self.judge_cost_cny, 6),
            "judge_latency_ms": round(self.judge_latency_ms, 2),
            "judge_error": self.judge_error,
            "judge_reasons": self.judge_reasons,
            "error": self.error,
            "strict_matches": self.strict_matches,
            "field_matches": self.matches,
            "ground_truth": (
                self.ground_truth.model_dump(mode="json")
                if self.ground_truth is not None
                else None
            ),
            "prediction": (
                self.prediction.model_dump(mode="json") if self.prediction else None
            ),
        }
        if include_raw:
            payload["raw_output"] = self.raw_output
        return payload


@dataclass
class EvaluationRun:
    cases: list[CaseEvaluation]
    prompt: PromptTemplate
    text_model: str
    vision_model: str
    judge_model: str | None
    data_dir: str
    ground_truth_path: str
    skipped_files: list[str]
    started_at: str
    duration_seconds: float
    trace_path: str | None
    workers: int


def discover_invoice_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {root}")
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def load_ground_truth(path: str | Path) -> dict[str, InvoiceData]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("ground_truth 必须是 JSON 对象（文件名 -> 字段）")

    ground_truth: dict[str, InvoiceData] = {}
    for name, entry in raw.items():
        if name.startswith("_"):
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"标准答案 {name!r} 必须是对象")
        ground_truth[name.replace("\\", "/")] = InvoiceData.model_validate(entry)
    return ground_truth


def _lookup_ground_truth(
    path: Path,
    data_dir: Path,
    ground_truth: dict[str, InvoiceData],
) -> InvoiceData | None:
    relative = path.relative_to(data_dir).as_posix()
    return ground_truth.get(relative) or ground_truth.get(path.name)


def _build_case(
    path: Path,
    relative: str,
    ground_truth: InvoiceData | None,
    outcome: ExtractionOutcome,
    judge: LLMAsJudge | None,
) -> CaseEvaluation:
    prediction = outcome.data

    if ground_truth is None:
        # 无标准答案：只保留抽取轨迹，不计算字段匹配与整单准确率
        failed = prediction is None
        failure_type = outcome.failure_type
        if failed and failure_type is FailureType.NONE:
            failure_type = FailureType.TYPE_G_OTHER
        return CaseEvaluation(
            file=relative,
            ground_truth=None,
            prediction=prediction,
            strict_matches={},
            matches={},
            exact_match=None,
            failure_type=failure_type,
            failure_detail=(outcome.error or "抽取失败") if failed else "",
            model=outcome.model,
            prompt_version=outcome.prompt_version,
            latency_ms=outcome.latency_ms,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            extraction_cost_cny=outcome.cost_cny,
            raw_output=outcome.raw_output,
            error=outcome.error,
        )

    if prediction is None:
        matches = {name: False for name in INVOICE_FIELDS}
        failure_type = (
            outcome.failure_type
            if outcome.failure_type is not FailureType.NONE
            else FailureType.TYPE_G_OTHER
        )
        return CaseEvaluation(
            file=relative,
            ground_truth=ground_truth,
            prediction=None,
            strict_matches=dict(matches),
            matches=matches,
            exact_match=False,
            failure_type=failure_type,
            failure_detail=outcome.error or "抽取失败",
            model=outcome.model,
            prompt_version=outcome.prompt_version,
            latency_ms=outcome.latency_ms,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            extraction_cost_cny=outcome.cost_cny,
            raw_output=outcome.raw_output,
            error=outcome.error,
        )

    strict_matches = {
        name: getattr(ground_truth, name) == getattr(prediction, name)
        for name in INVOICE_FIELDS
    }
    matches = dict(strict_matches)

    judge_used = False
    judge_input_tokens = 0
    judge_output_tokens = 0
    judge_cost = 0.0
    judge_latency = 0.0
    judge_error: str | None = None
    judge_reasons: dict[str, str] = {}

    diff_fields = [
        name
        for name in INVOICE_FIELDS
        if not strict_matches[name]
        and getattr(ground_truth, name) is not None
        and getattr(prediction, name) is not None
    ]
    if judge is not None and diff_fields:
        judge_used = True
        try:
            verdict = judge.judge_fields(
                {
                    name: (getattr(ground_truth, name), getattr(prediction, name))
                    for name in diff_fields
                }
            )
            for name in diff_fields:
                matches[name] = bool(verdict.matches.get(name, False))
            judge_reasons = verdict.reasons
            judge_input_tokens = verdict.input_tokens
            judge_output_tokens = verdict.output_tokens
            judge_latency = verdict.latency_ms
            judge_cost = estimate_cost(
                judge.model_name, judge_input_tokens, judge_output_tokens
            )
        except Exception as exc:  # 裁判失败时退回严格比对，不中断整批
            judge_error = str(exc)

    exact_match = all(matches.values())
    if exact_match:
        failure_type = FailureType.NONE
        failure_detail = ""
    else:
        mismatched = [name for name in INVOICE_FIELDS if not matches[name]]
        fabricated = [
            name
            for name in mismatched
            if getattr(ground_truth, name) is None
            and getattr(prediction, name) is not None
        ]
        if fabricated:
            failure_type = FailureType.TYPE_C_NULL_RULE
        elif mismatched == ["issue_date"]:
            failure_type = FailureType.TYPE_B_DATE
        else:
            failure_type = FailureType.TYPE_E_VALUE

        details = [
            f"{name}: 标准={_stringify(getattr(ground_truth, name))} "
            f"预测={_stringify(getattr(prediction, name))}"
            for name in mismatched
        ]
        reasons = [
            f"裁判[{name}]={judge_reasons[name]}"
            for name in mismatched
            if name in judge_reasons
        ]
        if judge_error:
            reasons.append(f"裁判失败: {judge_error}")
        failure_detail = "; ".join(details + reasons)

    return CaseEvaluation(
        file=relative,
        ground_truth=ground_truth,
        prediction=prediction,
        strict_matches=strict_matches,
        matches=matches,
        exact_match=exact_match,
        failure_type=failure_type,
        failure_detail=failure_detail,
        model=outcome.model,
        prompt_version=outcome.prompt_version,
        latency_ms=outcome.latency_ms,
        input_tokens=outcome.input_tokens,
        output_tokens=outcome.output_tokens,
        extraction_cost_cny=outcome.cost_cny,
        judge_used=judge_used,
        judge_input_tokens=judge_input_tokens,
        judge_output_tokens=judge_output_tokens,
        judge_cost_cny=judge_cost,
        judge_latency_ms=judge_latency,
        judge_error=judge_error,
        judge_reasons=judge_reasons,
        raw_output=outcome.raw_output,
        error=outcome.error,
    )


def _evaluate_one(
    router: InvoiceRouter,
    judge: LLMAsJudge | None,
    path: Path,
    relative: str,
    ground_truth: InvoiceData | None,
    prompt: PromptTemplate,
) -> CaseEvaluation:
    try:
        outcome = router.extract_outcome(path)
    except Exception as exc:  # 配置或路由异常也要进入报表
        outcome = ExtractionOutcome(
            file_path=path,
            file_type=path.suffix.lstrip(".").lower() or "unknown",
            prompt_version=prompt.version,
            model="",
            error=str(exc),
            failure_type=classify_exception(exc),
            exception=exc,
        )
    return _build_case(path, relative, ground_truth, outcome, judge)


def run_evaluation(
    *,
    data_dir: str | Path,
    ground_truth_path: str | Path,
    settings: Settings | None = None,
    prompt: PromptTemplate = DEFAULT_PROMPT,
    workers: int = 4,
    judge: LLMAsJudge | None = None,
    trace_log_path: str | Path | None = None,
    limit: int | None = None,
    client_factory: ClientFactory | None = None,
    progress_callback: ProgressCallback | None = None,
    include_unlabeled: bool = False,
) -> EvaluationRun:
    settings = settings or get_settings()
    data_root = Path(data_dir)
    files = discover_invoice_files(data_root)
    ground_truth = load_ground_truth(ground_truth_path)

    evaluated: list[tuple[Path, str, InvoiceData | None]] = []
    skipped_files: list[str] = []
    for path in files:
        relative = path.relative_to(data_root).as_posix()
        truth = _lookup_ground_truth(path, data_root, ground_truth)
        if truth is None and not include_unlabeled:
            skipped_files.append(relative)
            continue
        evaluated.append((path, relative, truth))
    if limit:
        evaluated = evaluated[:limit]

    trace_logger = TraceLogger(trace_log_path) if trace_log_path else None
    router = InvoiceRouter(
        settings=settings,
        client_factory=client_factory,
        prompt=prompt,
        trace_logger=trace_logger,
    )

    def task(item: tuple[Path, str, InvoiceData | None]) -> CaseEvaluation:
        path, relative, truth = item
        return _evaluate_one(router, judge, path, relative, truth, prompt)

    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.perf_counter()
    total = len(evaluated)
    ordered: list[CaseEvaluation | None] = [None] * total
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(task, item): index for index, item in enumerate(evaluated)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            ordered[futures[future]] = future.result()
            if progress_callback is not None:
                progress_callback(done, total)
    cases = [case for case in ordered if case is not None]
    duration = time.perf_counter() - started

    return EvaluationRun(
        cases=cases,
        prompt=prompt,
        text_model=settings.text_model,
        vision_model=settings.vision_model,
        judge_model=judge.model_name if judge is not None else None,
        data_dir=str(data_root),
        ground_truth_path=str(ground_truth_path),
        skipped_files=skipped_files,
        started_at=started_at,
        duration_seconds=duration,
        trace_path=str(trace_log_path) if trace_log_path else None,
        workers=workers,
    )
