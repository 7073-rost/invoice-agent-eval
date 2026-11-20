from __future__ import annotations

import json
from pathlib import Path

import pytest

from invoice_agent.judge import (
    CalibrationCase,
    LLMAsJudge,
    cohen_kappa,
    load_calibration,
    run_calibration,
)
from invoice_agent.llm.base import BaseLLMClient, LLMResponse

CALIBRATION_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "eval" / "judge_calibration.jsonl"
)


class StubJudgeClient(BaseLLMClient):
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    @property
    def model_name(self) -> str:
        return "stub-judge"

    def chat_text(self, *, system: str, user: str) -> LLMResponse:
        return LLMResponse(
            content=json.dumps(self._payload, ensure_ascii=False),
            model=self.model_name,
            input_tokens=200,
            output_tokens=20,
            latency_ms=8.0,
        )

    def chat_vision(self, *, system: str, user: str, image: bytes, mime_type: str) -> LLMResponse:
        raise NotImplementedError


def test_judge_fields_parses_verdict():
    client = StubJudgeClient(
        {"fields": {"buyer_name": {"match": True, "reason": "括号全半角差异"}}}
    )
    verdict = LLMAsJudge(client).judge_fields(
        {"buyer_name": ("阿里巴巴（中国）有限公司", "阿里巴巴(中国)有限公司")}
    )

    assert verdict.matches["buyer_name"] is True
    assert verdict.reasons["buyer_name"] == "括号全半角差异"
    assert verdict.input_tokens == 200


def test_kappa_known_value():
    kappa = cohen_kappa(
        [True, True, False, False, True],
        [True, False, False, False, True],
    )

    assert kappa == pytest.approx(0.6153846, rel=1e-6)


def test_calibration_fixture_has_balanced_labels():
    cases = load_calibration(CALIBRATION_PATH)

    assert len(cases) == 40
    assert sum(case.human_label for case in cases) == 20


def test_run_calibration_confusion_matrix():
    cases = [
        CalibrationCase("p1", "buyer_name", "甲方", "甲方", True),
        CalibrationCase("p2", "buyer_name", "乙方", "乙公司", True),
        CalibrationCase("n1", "buyer_name", "丙方", "丁方", False),
        CalibrationCase("n2", "buyer_name", "戊方", "戊方", False),
    ]
    judge = LLMAsJudge(
        StubJudgeClient(
            {
                "fields": {
                    "buyer_name": {"match": True, "reason": "测试固定为 true"}
                }
            }
        )
    )

    result = run_calibration(judge, cases)

    assert (result.tp, result.fp, result.fn, result.tn) == (2, 2, 0, 0)
    assert result.accuracy == 0.5
    assert result.kappa == 0.0
