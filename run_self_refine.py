"""Prompt 自动迭代与回归验证脚本。

用法：
    python run_self_refine.py --baseline-run eval_runs/20261119_120000_v1 --workers 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from invoice_agent.config import get_settings
from invoice_agent.errors import InvoiceAgentError
from invoice_agent.evaluation.report import build_summary_frame, write_run_report
from invoice_agent.evaluation.runner import discover_invoice_files, run_evaluation
from invoice_agent.judge import LLMAsJudge
from invoice_agent.llm.factory import create_llm_client, validate_model_config
from invoice_agent.prompts import PromptTemplate
from invoice_agent.refine import decide_acceptance, propose_prompt
from invoice_agent.router import IMAGE_SUFFIXES, PDF_SUFFIXES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于 Bad Case 自动迭代 Prompt 并回归验收")
    parser.add_argument("--baseline-run", required=True, help="基线评测目录（run_eval 产物）")
    parser.add_argument("--data-dir", default=None, help="默认沿用基线配置")
    parser.add_argument("--ground-truth", default=None, help="默认沿用基线配置")
    parser.add_argument("--output-dir", default="eval_runs", help="回归评测输出根目录")
    parser.add_argument("--workers", type=int, default=4, help="并发线程数")
    parser.add_argument("--max-examples", type=int, default=10, help="喂给反思模型的 Bad Case 数")
    parser.add_argument("--refiner-model", default=None, help="覆盖反思模型")
    parser.add_argument("--judge-model", default=None, help="覆盖裁判模型")
    parser.add_argument("--no-judge", action="store_true", help="回归阶段关闭 Judge")
    return parser


def _next_version(current: str) -> str:
    match = re.fullmatch(r"v(\d+)", current)
    if match:
        return f"v{int(match.group(1)) + 1}"
    return f"{current}-next"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline_dir = Path(args.baseline_run)

    try:
        meta = json.loads((baseline_dir / "summary.json").read_text(encoding="utf-8"))
        baseline_cases = json.loads(
            (baseline_dir / "cases.json").read_text(encoding="utf-8")
        )
        prompt_text = (baseline_dir / "prompt.txt").read_text(encoding="utf-8").strip()
        bad_cases_path = baseline_dir / "bad_cases.json"
        if bad_cases_path.is_file():
            bad_cases = json.loads(bad_cases_path.read_text(encoding="utf-8"))
        else:
            bad_cases = [case for case in baseline_cases if not case.get("exact_match")]
    except (OSError, json.JSONDecodeError) as exc:
        print(f"基线评测读取失败: {exc}", file=sys.stderr)
        return 1

    if not prompt_text:
        print("基线 prompt.txt 为空", file=sys.stderr)
        return 1

    data_dir = Path(args.data_dir or meta["data_dir"])
    ground_truth = Path(args.ground_truth or meta["ground_truth_path"])
    baseline_prompt = PromptTemplate(
        version=str(meta["prompt_version"]),
        system=prompt_text,
    )

    settings = get_settings()
    overrides = {
        "text_model": meta.get("text_model") or settings.text_model,
        "vision_model": meta.get("vision_model") or settings.vision_model,
    }
    if args.refiner_model:
        overrides["refiner_model"] = args.refiner_model
    if args.judge_model:
        overrides["judge_model"] = args.judge_model
    settings = settings.model_copy(update=overrides)

    judge_enabled = bool(meta.get("judge_enabled")) and not args.no_judge
    files = discover_invoice_files(data_dir)
    suffixes = {path.suffix.lower() for path in files}

    try:
        validate_model_config(settings.refiner_model, settings)
        if suffixes & PDF_SUFFIXES:
            validate_model_config(settings.text_model, settings)
        if suffixes & IMAGE_SUFFIXES:
            validate_model_config(settings.vision_model, settings)
        judge = None
        if judge_enabled:
            validate_model_config(settings.judge_model, settings)
            judge = LLMAsJudge(create_llm_client(settings.judge_model, settings))
    except InvoiceAgentError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 1

    bad_cases = bad_cases if isinstance(bad_cases, list) else []
    if not bad_cases:
        print("基线没有 Bad Case，无需迭代 Prompt")
        return 0

    try:
        proposal = propose_prompt(
            create_llm_client(settings.refiner_model, settings),
            baseline_prompt,
            bad_cases,
            max_examples=args.max_examples,
        )
    except (InvoiceAgentError, ValueError) as exc:
        print(f"Prompt 反思失败: {exc}", file=sys.stderr)
        return 1

    new_version = _next_version(baseline_prompt.version)
    candidate_prompt = PromptTemplate(version=new_version, system=proposal.new_prompt)
    run_dir = Path(args.output_dir) / f"{datetime.now():%Y%m%d_%H%M%S}_refine_{new_version}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "candidate_prompt.txt").write_text(proposal.new_prompt, encoding="utf-8")

    run = run_evaluation(
        data_dir=data_dir,
        ground_truth_path=ground_truth,
        settings=settings,
        prompt=candidate_prompt,
        workers=args.workers,
        judge=judge,
        trace_log_path=run_dir / "traces.jsonl",
    )
    write_run_report(run, run_dir)

    decision = decide_acceptance(
        baseline_cases,
        [case.to_dict() for case in run.cases],
    )
    decision_payload = {
        "baseline_run": str(baseline_dir),
        "candidate_run": str(run_dir),
        "baseline_prompt_version": baseline_prompt.version,
        "candidate_prompt_version": new_version,
        "accepted": decision.accepted,
        "reason": decision.reason,
        "baseline_exact_match": decision.baseline_exact_match,
        "candidate_exact_match": decision.candidate_exact_match,
        "regressed_files": decision.regressed_files,
        "improved_files": decision.improved_files,
        "analysis": proposal.analysis,
        "proposal_raw": proposal.raw,
    }
    (run_dir / "decision.json").write_text(
        json.dumps(decision_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== 反思分析 ===")
    print(proposal.analysis or "(模型未返回分析)")
    print()
    print(build_summary_frame(run).to_string(index=False))
    print()
    print(f"基线 EM: {decision.baseline_exact_match:.2%}")
    print(f"候选 EM: {decision.candidate_exact_match:.2%}")
    print(f"结论: {'接受' if decision.accepted else '拒绝'} - {decision.reason}")

    if decision.accepted:
        accepted_dir = Path("prompts") / "accepted"
        accepted_dir.mkdir(parents=True, exist_ok=True)
        accepted_path = accepted_dir / f"system_prompt_{new_version}.txt"
        accepted_path.write_text(proposal.new_prompt, encoding="utf-8")
        print(f"\n已保存新 Prompt: {accepted_path}")
        print(f"下次评测使用: python run_eval.py --prompt-file {accepted_path}")

    print(f"\n候选 Prompt: {run_dir / 'candidate_prompt.txt'}")
    print(f"回归报告: {run_dir / 'report.xlsx'}")
    print(f"验收结论: {run_dir / 'decision.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
