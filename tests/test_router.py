from __future__ import annotations

import json
from pathlib import Path

import pytest

from invoice_agent.errors import UnsupportedFileError
from invoice_agent.extractors.pdf import PdfInvoiceExtractor
from invoice_agent.llm.base import BaseLLMClient, LLMResponse
from invoice_agent.router import InvoiceRouter
from invoice_agent.tracing import TraceLogger

FAKE_JSON = (
    '{"invoice_number": "12345678", "issue_date": "2024-03-08", '
    '"buyer_name": "买方公司", "seller_name": "卖方公司", "total_amount": "88.00"}'
)


class FakeClient(BaseLLMClient):
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self.calls: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def chat_text(self, *, system: str, user: str) -> LLMResponse:
        self.calls.append("text")
        return LLMResponse(FAKE_JSON, self._model_name, 100, 50, 12.5)

    def chat_vision(self, *, system: str, user: str, image: bytes, mime_type: str) -> LLMResponse:
        self.calls.append("vision")
        return LLMResponse(FAKE_JSON, self._model_name, 100, 50, 12.5)


def make_factory(client: FakeClient):
    return lambda model_name: client


def test_image_routes_to_vision_model(tmp_path: Path):
    image_path = tmp_path / "invoice.png"
    image_path.write_bytes(b"\x89PNG\r\n")
    client = FakeClient("")
    created_models: list[str] = []

    def factory(model_name: str) -> FakeClient:
        created_models.append(model_name)
        return client

    router = InvoiceRouter(client_factory=factory)
    result = router.extract(image_path)

    assert created_models == ["qwen3.8-max"]
    assert client.calls == ["vision"]
    assert result.seller_name == "卖方公司"


def test_pdf_routes_to_text_model(tmp_path: Path, monkeypatch):
    pdf_path = tmp_path / "invoice.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(
        PdfInvoiceExtractor,
        "_read_text",
        staticmethod(lambda path: "购买方：买方公司  销售方：卖方公司"),
    )
    client = FakeClient("")
    created_models: list[str] = []

    def factory(model_name: str) -> FakeClient:
        created_models.append(model_name)
        return client

    router = InvoiceRouter(client_factory=factory)
    result = router.extract(pdf_path)

    assert created_models == ["deepseek-chat"]
    assert client.calls == ["text"]
    assert result.invoice_number == "12345678"


def test_unsupported_suffix_raises(tmp_path: Path):
    file_path = tmp_path / "invoice.txt"
    file_path.write_text("hello")
    router = InvoiceRouter(client_factory=make_factory(FakeClient("")))

    with pytest.raises(UnsupportedFileError):
        router.extract(file_path)


def test_missing_file_raises(tmp_path: Path):
    router = InvoiceRouter(client_factory=make_factory(FakeClient("")))

    with pytest.raises(FileNotFoundError):
        router.extract(tmp_path / "not-exists.png")


def test_trace_logger_records_usage_and_prompt_version(tmp_path: Path):
    image_path = tmp_path / "invoice.png"
    image_path.write_bytes(b"\x89PNG\r\n")
    trace_path = tmp_path / "traces.jsonl"
    router = InvoiceRouter(
        client_factory=make_factory(FakeClient("qwen3.8-max")),
        trace_logger=TraceLogger(trace_path),
    )

    router.extract(image_path)

    record = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert record["prompt_version"] == "v1"
    assert record["file_type"] == "image"
    assert record["raw_output"] == FAKE_JSON
    assert record["input_tokens"] == 100
    assert record["failure_type"] == "none"
    assert record["extracted"]["seller_name"] == "卖方公司"
