"""抽取器基类：统一执行模型调用、结果解析、耗时与 Trace 采集。"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from invoice_agent.errors import EmptyExtractionError, ModelOutputError
from invoice_agent.llm.base import BaseLLMClient, LLMResponse
from invoice_agent.prompts import DEFAULT_PROMPT, PromptTemplate
from invoice_agent.schemas import InvoiceData
from invoice_agent.tracing import ExtractionOutcome, classify_exception


def parse_json_object(raw: str) -> dict[str, Any]:
    """从模型回复中截取 JSON 对象，兼容代码块包裹和前后缀说明文字。"""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ModelOutputError(f"模型输出中未找到 JSON 对象: {raw[:200]!r}")
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"模型输出不是合法 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelOutputError("模型输出的 JSON 顶层必须是对象")
    return payload


class BaseExtractor(ABC):
    """抽取器基类：持有 LLM 客户端与 Prompt，返回带 Trace 信息的结果。"""

    file_type: str = "unknown"

    def __init__(
        self,
        client: BaseLLMClient,
        prompt: PromptTemplate = DEFAULT_PROMPT,
    ) -> None:
        self._client = client
        self._prompt = prompt

    def extract(self, file_path: Path) -> ExtractionOutcome:
        outcome = ExtractionOutcome(
            file_path=file_path,
            file_type=self.file_type,
            prompt_version=self._prompt.version,
            model=self._client.model_name,
        )
        started = time.perf_counter()
        try:
            response = self._call_model(file_path)
            outcome.raw_output = response.content
            outcome.input_tokens = response.input_tokens
            outcome.output_tokens = response.output_tokens
            outcome.llm_latency_ms = response.latency_ms
            outcome.data = self._parse_invoice(response.content)
        except Exception as exc:  # 边界：单条失败不能中断整批评测
            outcome.error = str(exc)
            outcome.failure_type = classify_exception(exc)
            outcome.exception = exc
        outcome.latency_ms = (time.perf_counter() - started) * 1000
        return outcome

    @abstractmethod
    def _call_model(self, file_path: Path) -> LLMResponse:
        """执行具体链路的模型调用。"""

    def _parse_invoice(self, raw: str) -> InvoiceData:
        try:
            invoice = InvoiceData.model_validate(parse_json_object(raw))
        except ValidationError as exc:
            raise ModelOutputError(f"模型输出不符合票据结构: {exc}") from exc
        if all(value is None for value in invoice.model_dump().values()):
            raise EmptyExtractionError("模型未抽取到任何票据字段，请检查文件内容或提示词")
        return invoice
