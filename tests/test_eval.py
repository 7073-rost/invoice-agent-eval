from __future__ import annotations

import json
from pathlib import Path

from invoice_agent.config import Settings
from invoice_agent.evaluation.report import (
    build_field_frame,
    compute_summary,
    write_run_report,
)
from invoice_agent.evaluation.runner import run_evaluation
from invoice_agent.llm.base import BaseLLMClient, LLMResponse
from invoice_agent.tracing import FailureType

CORRECT = {
    "invoice_number": "12345678",
    "issue_date": "2024-03-08",
    "buyer_name": "买方公司",
    "seller_name": "卖方公司",
    "total_amount": "88.00",
}
WRONG_DATE = {**CORRECT, "issue_date": "2024-03-09"}


class MappingVisionClient(BaseLLMClient):
    def __init__(self, mapping: dict[bytes, dict]) -> None:
        self._mapping = mapping

    @property
    def model_name(self) -> str:
        return "qwen3.8-max"

    def chat_text(self, *, system: str, user: str) -> LLMResponse:
        raise NotImplementedError

    def chat_vision(self, *, system: str, user: str, image: bytes, mime_type: str) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(self._mapping[image], ensure_ascii=False),
            model=self.model_name,
            input_tokens=100,
            output_tokens=20,
            latency_ms=10.0,
        )


def test_run_evaluation_computes_metrics_and_reports(tmp_path: Path):
    data_dir = tmp_path / "invoices"
    data_dir.mkdir()
    (data_dir / "a.png").write_bytes(b"A")
    (data_dir / "b.png").write_bytes(b"B")

    ground_truth = {"a.png": CORRECT, "b.png": CORRECT}
    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps(ground_truth, ensure_ascii=False), encoding="utf-8"
    )

    client = MappingVisionClient({b"A": CORRECT, b"B": WRONG_DATE})
    run = run_evaluation(
        data_dir=data_dir,
        ground_truth_path=ground_truth_path,
        settings=Settings(),
        workers=2,
        judge=None,
        trace_log_path=tmp_path / "traces.jsonl",
        client_factory=lambda model_name: client,
    )

    assert len(run.cases) == 2
    assert sum(case.exact_match for case in run.cases) == 1
    wrong = next(case for case in run.cases if not case.exact_match)
    assert wrong.failure_type is FailureType.TYPE_B_DATE
    assert wrong.latency_ms > 0
    assert wrong.extraction_cost_cny > 0

    fields = build_field_frame(run.cases)
    date_row = fields.set_index("field").loc["issue_date"]
    assert date_row["tp"] == 1
    assert date_row["fp"] == 1
    assert date_row["fn"] == 1

    trace_lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(trace_lines) == 2

    run_dir = write_run_report(run, tmp_path / "run")
    assert (run_dir / "report.xlsx").is_file()
    assert (run_dir / "field_metrics.csv").is_file()
    assert (run_dir / "bad_cases.json").is_file()
    assert (run_dir / "summary.json").is_file()


def test_run_evaluation_skips_files_without_ground_truth(tmp_path: Path):
    data_dir = tmp_path / "invoices"
    data_dir.mkdir()
    (data_dir / "a.png").write_bytes(b"A")

    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text("{}", encoding="utf-8")

    run = run_evaluation(
        data_dir=data_dir,
        ground_truth_path=ground_truth_path,
        settings=Settings(),
        workers=1,
        judge=None,
    )

    assert run.cases == []
    assert run.skipped_files == ["a.png"]


def test_run_evaluation_includes_unlabeled_when_enabled(tmp_path: Path):
    data_dir = tmp_path / "invoices"
    data_dir.mkdir()
    (data_dir / "a.png").write_bytes(b"A")
    (data_dir / "b.png").write_bytes(b"B")

    ground_truth_path = tmp_path / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps({"a.png": CORRECT}, ensure_ascii=False), encoding="utf-8"
    )

    client = MappingVisionClient({b"A": CORRECT, b"B": CORRECT})
    run = run_evaluation(
        data_dir=data_dir,
        ground_truth_path=ground_truth_path,
        settings=Settings(),
        workers=2,
        judge=None,
        trace_log_path=tmp_path / "traces.jsonl",
        client_factory=lambda model_name: client,
        include_unlabeled=True,
    )

    assert len(run.cases) == 2
    unlabeled = next(case for case in run.cases if case.file == "b.png")
    assert unlabeled.ground_truth is None
    assert unlabeled.exact_match is None
    assert unlabeled.matches == {}
    assert unlabeled.prediction is not None

    summary = compute_summary(run)
    assert summary["num_cases"] == 2
    assert summary["num_labeled"] == 1
    assert summary["num_unlabeled"] == 1
    assert summary["exact_match_rate"] == 1.0

    fields = build_field_frame(run.cases)
    assert int(fields["support"].sum()) == 5

    run_dir = write_run_report(run, tmp_path / "run")
    assert json.loads((run_dir / "bad_cases.json").read_text(encoding="utf-8")) == []
