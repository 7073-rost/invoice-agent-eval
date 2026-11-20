"""Pandas 评测报表：字段指标、Exact Match、P99 与 Token 成本。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from invoice_agent.evaluation.runner import (
    INVOICE_FIELDS,
    CaseEvaluation,
    EvaluationRun,
)
from invoice_agent.tracing import FAILURE_DESCRIPTIONS, FailureType

SUMMARY_LABELS = {
    "num_cases": "样本数",
    "num_labeled": "有标准答案样本数",
    "num_unlabeled": "无标准答案样本数",
    "num_skipped": "无标准答案跳过数",
    "exact_match_rate": "整单完全正确率",
    "macro_precision": "字段平均精确率",
    "macro_recall": "字段平均召回率",
    "macro_f1": "字段平均 F1",
    "p99_latency_ms": "P99 抽取耗时 (ms)",
    "p99_judge_latency_ms": "P99 裁判耗时 (ms)",
    "total_input_tokens": "输入 Token 总数",
    "total_output_tokens": "输出 Token 总数",
    "extraction_cost_cny": "抽取成本 (元)",
    "judge_cost_cny": "裁判成本 (元)",
    "total_cost_cny": "总成本 (元)",
    "judge_used_cases": "触发裁判的样本数",
    "judge_error_cases": "裁判失败样本数",
}


def build_field_frame(cases: list[CaseEvaluation]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name in INVOICE_FIELDS:
        tp = fp = fn = tn = 0
        for case in cases:
            if case.ground_truth is None:
                continue
            truth = getattr(case.ground_truth, name)
            predicted = getattr(case.prediction, name) if case.prediction else None
            matched = case.matches[name]
            if truth is not None and matched:
                tp += 1
            elif truth is not None:
                fn += 1
            if predicted is not None and not matched:
                fp += 1
            if truth is None and predicted is None:
                tn += 1

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "field": name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "support": tp + fn,
            }
        )
    return pd.DataFrame(rows)


def _p99(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(pd.Series(values, dtype="float64").quantile(0.99))


def compute_summary(run: EvaluationRun) -> dict[str, Any]:
    cases = run.cases
    labeled = [case for case in cases if case.ground_truth is not None]
    fields = build_field_frame(labeled)
    failures = Counter(
        case.failure_type.value
        for case in cases
        if case.failure_type is not FailureType.NONE
    )
    extraction_cost = sum(case.extraction_cost_cny for case in cases)
    judge_cost = sum(case.judge_cost_cny for case in cases)
    macro_precision = (
        round(float(fields["precision"].mean()), 4) if labeled else 0.0
    )
    macro_recall = round(float(fields["recall"].mean()), 4) if labeled else 0.0
    macro_f1 = round(float(fields["f1"].mean()), 4) if labeled else 0.0

    return {
        "num_cases": len(cases),
        "num_labeled": len(labeled),
        "num_unlabeled": len(cases) - len(labeled),
        "num_skipped": len(run.skipped_files),
        "exact_match_rate": round(
            sum(bool(case.exact_match) for case in labeled) / len(labeled), 4
        )
        if labeled
        else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "p99_latency_ms": round(_p99([case.latency_ms for case in cases]), 2),
        "p99_judge_latency_ms": round(
            _p99([case.judge_latency_ms for case in cases if case.judge_used]), 2
        ),
        "total_input_tokens": sum(case.input_tokens for case in cases),
        "total_output_tokens": sum(case.output_tokens for case in cases),
        "extraction_cost_cny": round(extraction_cost, 6),
        "judge_cost_cny": round(judge_cost, 6),
        "total_cost_cny": round(extraction_cost + judge_cost, 6),
        "judge_used_cases": sum(case.judge_used for case in cases),
        "judge_error_cases": sum(case.judge_error is not None for case in cases),
        "failure_counts": dict(failures),
    }


def build_summary_frame(run: EvaluationRun) -> pd.DataFrame:
    summary = compute_summary(run)
    rows: list[tuple[str, str]] = []
    for key, label in SUMMARY_LABELS.items():
        value = summary[key]
        if key.endswith("_rate") or key in {"macro_precision", "macro_recall", "macro_f1"}:
            display = f"{value:.2%}"
        else:
            display = str(value)
        rows.append((label, display))

    total_failures = sum(summary["failure_counts"].values())
    failure_text = "; ".join(
        f"{name}: {count} ({count / total_failures:.1%})"
        for name, count in sorted(summary["failure_counts"].items())
    ) or "无"
    rows.append(("失败归因", failure_text))
    return pd.DataFrame(rows, columns=["指标", "数值"])


def build_failure_frame(cases: list[CaseEvaluation]) -> pd.DataFrame:
    failures = Counter(
        case.failure_type.value
        for case in cases
        if case.failure_type is not FailureType.NONE
    )
    total = sum(failures.values())
    rows = []
    for name, count in failures.most_common():
        failure_type = FailureType(name)
        rows.append(
            {
                "failure_type": name,
                "description": FAILURE_DESCRIPTIONS[failure_type],
                "count": count,
                "ratio_of_failures": round(count / total, 4) if total else 0.0,
            }
        )
    return pd.DataFrame(
        rows, columns=["failure_type", "description", "count", "ratio_of_failures"]
    )


def build_cases_frame(cases: list[CaseEvaluation]) -> pd.DataFrame:
    rows = []
    for case in cases:
        rows.append(
            {
                "file": case.file,
                "exact_match": case.exact_match,
                "failure_type": case.failure_type.value,
                "failure_detail": case.failure_detail,
                "model": case.model,
                "prompt_version": case.prompt_version,
                "latency_ms": round(case.latency_ms, 2),
                "input_tokens": case.input_tokens,
                "output_tokens": case.output_tokens,
                "extraction_cost_cny": round(case.extraction_cost_cny, 6),
                "judge_used": case.judge_used,
                "judge_input_tokens": case.judge_input_tokens,
                "judge_output_tokens": case.judge_output_tokens,
                "judge_cost_cny": round(case.judge_cost_cny, 6),
                "judge_error": case.judge_error,
                "error": case.error,
                "ground_truth": (
                    json.dumps(
                        case.ground_truth.model_dump(mode="json"),
                        ensure_ascii=False,
                    )
                    if case.ground_truth is not None
                    else None
                ),
                "prediction": json.dumps(
                    case.prediction.model_dump(mode="json")
                    if case.prediction
                    else None,
                    ensure_ascii=False,
                ),
                "field_matches": json.dumps(case.matches, ensure_ascii=False),
                "strict_matches": json.dumps(case.strict_matches, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def write_run_report(run: EvaluationRun, run_dir: str | Path) -> Path:
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)

    summary = compute_summary(run)
    fields = build_field_frame(run.cases)
    failures = build_failure_frame(run.cases)
    cases = build_cases_frame(run.cases)
    summary_frame = build_summary_frame(run)

    summary_frame.to_csv(directory / "summary.csv", index=False, encoding="utf-8-sig")
    fields.to_csv(directory / "field_metrics.csv", index=False, encoding="utf-8-sig")
    failures.to_csv(directory / "failures.csv", index=False, encoding="utf-8-sig")
    cases.to_csv(directory / "cases.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(directory / "report.xlsx", engine="openpyxl") as writer:
        summary_frame.to_excel(writer, sheet_name="summary", index=False)
        fields.to_excel(writer, sheet_name="fields", index=False)
        failures.to_excel(writer, sheet_name="failures", index=False)
        cases.to_excel(writer, sheet_name="cases", index=False)

    (directory / "prompt.txt").write_text(run.prompt.system, encoding="utf-8")
    (directory / "cases.json").write_text(
        json.dumps(
            [case.to_dict() for case in run.cases],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (directory / "bad_cases.json").write_text(
        json.dumps(
            [
                case.to_dict(include_raw=True)
                for case in run.cases
                if case.exact_match is False
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "run_dir": str(directory),
                "started_at": run.started_at,
                "duration_seconds": round(run.duration_seconds, 2),
                "data_dir": run.data_dir,
                "ground_truth_path": run.ground_truth_path,
                "prompt_version": run.prompt.version,
                "text_model": run.text_model,
                "vision_model": run.vision_model,
                "judge_model": run.judge_model,
                "judge_enabled": run.judge_model is not None,
                "workers": run.workers,
                "trace_path": run.trace_path,
                "skipped_files": run.skipped_files,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return directory
