"""PDF 链路：pdfplumber 提取文本 + 文本模型抽取。"""

from __future__ import annotations

from pathlib import Path

import pdfplumber

from invoice_agent.errors import DocumentUnreadableError
from invoice_agent.extractors.base import BaseExtractor
from invoice_agent.llm.base import BaseLLMClient, LLMResponse
from invoice_agent.prompts import DEFAULT_PROMPT, TEXT_USER_PROMPT, PromptTemplate


class PdfInvoiceExtractor(BaseExtractor):
    file_type = "pdf"

    def __init__(
        self,
        client: BaseLLMClient,
        prompt: PromptTemplate = DEFAULT_PROMPT,
        *,
        min_text_chars: int = 10,
    ) -> None:
        super().__init__(client, prompt)
        self._min_text_chars = min_text_chars

    def _call_model(self, file_path: Path) -> LLMResponse:
        text = self._read_text(file_path)
        if len(text) < self._min_text_chars:
            raise DocumentUnreadableError(
                "PDF 未提取到有效文本层，可能为扫描件；请转为图片后走视觉链路。"
            )
        return self._client.chat_text(
            system=self._prompt.system,
            user=TEXT_USER_PROMPT.format(document=text),
        )

    @staticmethod
    def _read_text(file_path: Path) -> str:
        try:
            with pdfplumber.open(file_path) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages]
        except Exception as exc:  # pdfplumber 底层抛出的异常类型不稳定
            raise DocumentUnreadableError(f"解析 PDF 失败: {exc}") from exc
        return "\n".join(pages).strip()
