from __future__ import annotations

import json

from invoice_agent.llm.base import BaseLLMClient, LLMResponse
from invoice_agent.prompts import PromptTemplate
from invoice_agent.refine import decide_acceptance, propose_prompt


class StubRefineClient(BaseLLMClient):
    @property
    def model_name(self) -> str:
        return "stub-refiner"

    def chat_text(self, *, system: str, user: str) -> LLMResponse:
        payload = {
            "analysis": "日期字段缺少格式约束，且 null 规则不够显眼。",
            "new_prompt": "新版系统提示词：严格输出 JSON，日期统一为 YYYY-MM-DD。",
        }
        return LLMResponse(
            content=json.dumps(payload, ensure_ascii=False),
            model=self.model_name,
            input_tokens=300,
            output_tokens=80,
            latency_ms=20.0,
        )

    def chat_vision(self, *, system: str, user: str, image: bytes, mime_type: str) -> LLMResponse:
        raise NotImplementedError


def test_propose_prompt_parses_new_prompt():
    proposal = propose_prompt(
        StubRefineClient(),
        PromptTemplate(version="v1", system="旧提示词"),
        [
            {
                "file": "a.png",
                "ground_truth": {"issue_date": "2024-03-08"},
                "prediction": {"issue_date": "2024/03/08"},
                "failure_type": "Type_B",
                "failure_detail": "日期格式错误",
                "error": None,
                "raw_output": "{...}",
            }
        ],
    )

    assert "YYYY-MM-DD" in proposal.new_prompt
    assert "日期字段" in proposal.analysis


def test_decide_acceptance_accepts_improvement_without_regression():
    baseline = [
        {"file": "a.png", "exact_match": True},
        {"file": "b.png", "exact_match": False},
    ]
    candidate = [
        {"file": "a.png", "exact_match": True},
        {"file": "b.png", "exact_match": True},
    ]

    decision = decide_acceptance(baseline, candidate)

    assert decision.accepted is True
    assert decision.regressed_files == []
    assert decision.improved_files == ["b.png"]
    assert decision.candidate_exact_match == 1.0


def test_decide_acceptance_rejects_regression():
    baseline = [
        {"file": "a.png", "exact_match": True},
        {"file": "b.png", "exact_match": False},
    ]
    candidate = [
        {"file": "a.png", "exact_match": False},
        {"file": "b.png", "exact_match": True},
    ]

    decision = decide_acceptance(baseline, candidate)

    assert decision.accepted is False
    assert decision.regressed_files == ["a.png"]
    assert "回归" in decision.reason


def test_decide_acceptance_rejects_flat_result():
    baseline = [{"file": "a.png", "exact_match": False}]
    candidate = [{"file": "a.png", "exact_match": False}]

    decision = decide_acceptance(baseline, candidate)

    assert decision.accepted is False
    assert "未提升" in decision.reason
