"""LLM-as-a-Judge 人工校准脚本。

用法：
    python judge.py --calibration data/eval/judge_calibration.jsonl --judge-model deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from invoice_agent.config import get_settings
from invoice_agent.errors import InvoiceAgentError
from invoice_agent.judge import LLMAsJudge, load_calibration, run_calibration
from invoice_agent.llm.factory import create_llm_client, validate_model_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校准 LLM-as-a-Judge 的判定准确率与 Kappa 系数")
    parser.add_argument(
        "--calibration",
        default="data/eval/judge_calibration.jsonl",
        help="校准集路径（JSONL，需包含 human_label）",
    )
    parser.add_argument("--judge-model", default=None, help="裁判模型，默认读取 JUDGE_MODEL")
    parser.add_argument("--limit", type=int, default=None, help="仅跑前 N 条，便于冒烟测试")
    parser.add_argument(
        "--output-dir",
        default="eval_runs",
        help="校准报告输出目录，会创建带时间戳的子目录",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    judge_model = args.judge_model or settings.judge_model

    try:
        validate_model_config(judge_model, settings)
    except InvoiceAgentError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 1

    try:
        cases = load_calibration(args.calibration)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"校准集读取失败: {exc}", file=sys.stderr)
        return 1

    judge = LLMAsJudge(create_llm_client(judge_model, settings))
    result = run_calibration(judge, cases, limit=args.limit)

    run_dir = Path(args.output_dir) / f"judge_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(result.rows)
    frame.to_csv(run_dir / "calibration_cases.csv", index=False, encoding="utf-8-sig")

    report = {
        "judge_model": judge_model,
        "prompt_version": "judge-v1",
        "calibration_file": str(args.calibration),
        "total": result.total,
        "valid": result.valid,
        "errors": result.errors,
        "accuracy": round(result.accuracy, 4),
        "precision": round(result.precision, 4),
        "recall": round(result.recall, 4),
        "cohen_kappa": round(result.kappa, 4),
        "confusion": {
            "tp": result.tp,
            "fp": result.fp,
            "fn": result.fn,
            "tn": result.tn,
        },
    }
    (run_dir / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = pd.DataFrame(
        [
            ("Judge 模型", judge_model),
            ("校准样本数", result.total),
            ("有效判定数", result.valid),
            ("裁判调用失败", result.errors),
            ("准确率", f"{result.accuracy:.2%}"),
            ("精确率", f"{result.precision:.2%}"),
            ("召回率", f"{result.recall:.2%}"),
            ("Cohen's Kappa", f"{result.kappa:.4f}"),
            ("TP / FP / FN / TN", f"{result.tp} / {result.fp} / {result.fn} / {result.tn}"),
        ],
        columns=["指标", "数值"],
    )
    print(summary.to_string(index=False))
    print(f"\n校准报告: {run_dir / 'calibration_report.json'}")
    print(f"逐条结果: {run_dir / 'calibration_cases.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
