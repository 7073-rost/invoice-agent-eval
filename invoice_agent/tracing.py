"""结构化 Trace、失败类型与单次抽取结果。"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from invoice_agent.errors import (
    DocumentUnreadableError,
    EmptyExtractionError,
    InvoiceAgentError,
    LLMRequestError,
    ModelOutputError,
)
from invoice_agent.pricing import estimate_cost
from invoice_agent.schemas import InvoiceData


class FailureType(str, Enum):
    NONE = "none"
    TYPE_A_UNREADABLE = "Type_A"
    TYPE_B_DATE = "Type_B"
    TYPE_C_NULL_RULE = "Type_C"
    TYPE_D_JSON = "Type_D"
    TYPE_E_VALUE = "Type_E"
    TYPE_F_LLM = "Type_F"
    TYPE_G_OTHER = "Type_G"


FAILURE_DESCRIPTIONS: dict[FailureType, str] = {
    FailureType.NONE: "无错误",
    FailureType.TYPE_A_UNREADABLE: "文件不可读/图像模糊/未识别出任何字段",
    FailureType.TYPE_B_DATE: "日期格式异常或日期幻觉",
    FailureType.TYPE_C_NULL_RULE: "忽略 Null 规则，编造了不存在的字段",
    FailureType.TYPE_D_JSON: "模型输出非 JSON 或结构校验失败",
    FailureType.TYPE_E_VALUE: "字段值抽取错误",
    FailureType.TYPE_F_LLM: "大模型调用失败",
    FailureType.TYPE_G_OTHER: "其他异常",
}


def classify_exception(exc: BaseException) -> FailureType:
    """把异常映射为可统计的失败类型。"""
    if isinstance(exc, DocumentUnreadableError):
        return FailureType.TYPE_A_UNREADABLE
    if isinstance(exc, EmptyExtractionError):
        return FailureType.TYPE_A_UNREADABLE
    if isinstance(exc, ModelOutputError):
        return FailureType.TYPE_D_JSON
    if isinstance(exc, LLMRequestError):
        return FailureType.TYPE_F_LLM
    if isinstance(exc, InvoiceAgentError):
        return FailureType.TYPE_G_OTHER
    return FailureType.TYPE_G_OTHER


@dataclass
class ExtractionOutcome:
    """单次抽取的完整运行轨迹，不做异常抛出，便于批处理统计。"""

    file_path: Path
    file_type: str
    prompt_version: str
    model: str
    data: InvoiceData | None = None
    raw_output: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    error: str | None = None
    failure_type: FailureType = FailureType.NONE
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    exception: BaseException | None = field(default=None, repr=False)

    @property
    def cost_cny(self) -> float:
        return estimate_cost(self.model, self.input_tokens, self.output_tokens)

    def to_trace_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "file_path": str(self.file_path),
            "file_type": self.file_type,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "raw_output": self.raw_output,
            "latency_ms": round(self.latency_ms, 2),
            "llm_latency_ms": round(self.llm_latency_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_cny": round(self.cost_cny, 6),
            "failure_type": self.failure_type.value,
            "error": self.error,
            "extracted": self.data.model_dump(mode="json") if self.data else None,
        }


class TraceLogger:
    """线程安全的结构化 JSONL 轨迹日志。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def log(self, outcome: ExtractionOutcome) -> None:
        line = json.dumps(outcome.to_trace_dict(), ensure_ascii=False)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
