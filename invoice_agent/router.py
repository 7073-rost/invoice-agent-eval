"""按文件类型路由到不同的抽取链路。"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from invoice_agent.config import Settings, get_settings
from invoice_agent.errors import ExtractionError, UnsupportedFileError
from invoice_agent.extractors.image import IMAGE_MIME_TYPES, ImageInvoiceExtractor
from invoice_agent.extractors.pdf import PdfInvoiceExtractor
from invoice_agent.llm.base import BaseLLMClient
from invoice_agent.llm.factory import create_llm_client
from invoice_agent.prompts import DEFAULT_PROMPT, PromptTemplate
from invoice_agent.schemas import InvoiceData
from invoice_agent.tracing import ExtractionOutcome, TraceLogger

ClientFactory = Callable[[str], BaseLLMClient]

PDF_SUFFIXES = frozenset({".pdf"})
IMAGE_SUFFIXES = frozenset(IMAGE_MIME_TYPES)
SUPPORTED_SUFFIXES = PDF_SUFFIXES | IMAGE_SUFFIXES


class InvoiceRouter:
    """文件类型路由：

    - PDF  -> pdfplumber 提取文本 -> 文本模型
    - 图片 -> 视觉模型
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client_factory: ClientFactory | None = None,
        prompt: PromptTemplate = DEFAULT_PROMPT,
        trace_logger: TraceLogger | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._client_factory = client_factory or self._default_client_factory
        self._prompt = prompt
        self._trace_logger = trace_logger
        self._client_cache: dict[str, BaseLLMClient] = {}
        self._client_lock = threading.Lock()

    def _default_client_factory(self, model_name: str) -> BaseLLMClient:
        return create_llm_client(model_name, settings=self._settings)

    def _get_client(self, model_name: str) -> BaseLLMClient:
        with self._client_lock:
            client = self._client_cache.get(model_name)
            if client is None:
                client = self._client_factory(model_name)
                self._client_cache[model_name] = client
            return client

    def extract(self, file_path: str | Path) -> InvoiceData:
        outcome = self.extract_outcome(file_path)
        if outcome.data is not None:
            return outcome.data
        if outcome.exception is not None:
            raise outcome.exception
        raise ExtractionError(outcome.error or "抽取失败")

    def extract_outcome(self, file_path: str | Path) -> ExtractionOutcome:
        path = Path(file_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在: {path}")

        suffix = path.suffix.lower()
        if suffix in PDF_SUFFIXES:
            extractor = PdfInvoiceExtractor(
                self._get_client(self._settings.text_model),
                prompt=self._prompt,
            )
        elif suffix in IMAGE_SUFFIXES:
            extractor = ImageInvoiceExtractor(
                self._get_client(self._settings.vision_model),
                prompt=self._prompt,
            )
        else:
            supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
            raise UnsupportedFileError(
                f"不支持的文件类型 {suffix or '(无扩展名)'}，支持: {supported}"
            )

        outcome = extractor.extract(path)
        if self._trace_logger is not None:
            self._trace_logger.log(outcome)
        return outcome
