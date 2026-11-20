"""批处理评测脚本：并发跑批 + Pandas 报表。

用法：
    python run_eval.py --data-dir data/invoices --ground-truth data/eval/ground_truth.json --workers 8
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from invoice_agent.config import get_settings
from invoice_agent.errors import InvoiceAgentError
from invoice_agent.evaluation.report import build_summary_frame, write_run_report
from invoice_agent.evaluation.runner import discover_invoice_files, run_evaluation
from invoice_agent.judge import LLMAsJudge
from invoice_agent.llm.factory import create_llm_client, validate_model_config
from invoice_agent.prompts import DEFAULT_PROMPT, load_prompt_file
from invoice_agent.router import IMAGE_SUFFIXES, PDF_SUFFIXES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="票据抽取批处理评测")
    parser.add_argument("--data-dir", default="data/invoices", help="票据文件夹")
    parser.add_argument(
        "--ground-truth",
        default="data/eval/ground_truth.json",
        help="标准答案 JSON（文件名 -> 字段）",
    )
    parser.add_argument("--output-dir", default="eval_runs", help="评测报告根目录")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    parser.add_argument("--limit", type=int, default=None, help="仅评测前 N 条")
    parser.add_argument("--prompt-file", default=None, help="自定义 System Prompt 文件")
    parser.add_argument("--prompt-version", default=None, help="自定义 Prompt 版本号")
    parser.add_argument("--text-model", default=None, help="覆盖 PDF 文本链路模型")
    parser.add_argument("--vision-model", default=None, help="覆盖图片视觉链路模型")
    parser.add_argument("--judge-model", default=None, help="覆盖裁判模型")
    parser.add_argument("--no-judge", action="store_true", help="关闭 LLM-as-a-Judge")
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="缺少标准答案的文件也执行抽取，但不计入准确率",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    overrides = {}
    if args.text_model:
        overrides["text_model"] = args.text_model
    if args.vision_model:
        overrides["vision_model"] = args.vision_model
    if args.judge_model:
        overrides["judge_model"] = args.judge_model
    if overrides:
        settings = settings.model_copy(update=overrides)

    data_dir = Path(args.data_dir)
    ground_truth = Path(args.ground_truth)
    if not data_dir.is_dir():
        print(f"数据目录不存在: {data_dir}", file=sys.stderr)
        return 1
    if not ground_truth.is_file():
        print(f"标准答案文件不存在: {ground_truth}", file=sys.stderr)
        return 1

    try:
        prompt = (
            load_prompt_file(Path(args.prompt_file), version=args.prompt_version)
            if args.prompt_file
            else DEFAULT_PROMPT
        )
    except (OSError, ValueError) as exc:
        print(f"Prompt 读取失败: {exc}", file=sys.stderr)
        return 1

    files = discover_invoice_files(data_dir)
    if not files:
        print(f"未在 {data_dir} 发现 pdf/png/jpg/jpeg 文件", file=sys.stderr)
        return 1

    suffixes = {path.suffix.lower() for path in files}
    try:
        if suffixes & PDF_SUFFIXES:
            validate_model_config(settings.text_model, settings)
        if suffixes & IMAGE_SUFFIXES:
            validate_model_config(settings.vision_model, settings)

        judge = None
        if not args.no_judge:
            validate_model_config(settings.judge_model, settings)
            judge = LLMAsJudge(create_llm_client(settings.judge_model, settings))
    except InvoiceAgentError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 1

    run_dir = Path(args.output_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_{prompt.version}"
    run = run_evaluation(
        data_dir=data_dir,
        ground_truth_path=ground_truth,
        settings=settings,
        prompt=prompt,
        workers=args.workers,
        judge=judge,
        trace_log_path=run_dir / "traces.jsonl",
        limit=args.limit,
        include_unlabeled=args.include_unlabeled,
    )
    write_run_report(run, run_dir)

    print(build_summary_frame(run).to_string(index=False))
    if run.skipped_files:
        print(f"\n跳过 {len(run.skipped_files)} 个缺少标准答案的文件")
    unlabeled = sum(case.ground_truth is None for case in run.cases)
    if unlabeled:
        print(f"无标注样本 {unlabeled} 个：仅抽取，不计入准确率")
    print(f"\n评测报告: {run_dir / 'report.xlsx'}")
    print(f"结构化轨迹: {run_dir / 'traces.jsonl'}")
    print(f"失败样例: {run_dir / 'bad_cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
