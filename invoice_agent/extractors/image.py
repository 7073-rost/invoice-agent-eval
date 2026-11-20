"""图片链路：图片直接送视觉模型抽取。"""

from __future__ import annotations

from pathlib import Path

from invoice_agent.errors import DocumentUnreadableError, UnsupportedFileError
from invoice_agent.extractors.base import BaseExtractor
from invoice_agent.llm.base import BaseLLMClient, LLMResponse
from invoice_agent.prompts import DEFAULT_PROMPT, IMAGE_USER_PROMPT, PromptTemplate

IMAGE_MIME_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


class ImageInvoiceExtractor(BaseExtractor):
    file_type = "image"

    def __init__(
        self,
        client: BaseLLMClient,
        prompt: PromptTemplate = DEFAULT_PROMPT,
    ) -> None:
        super().__init__(client, prompt)

    def _call_model(self, file_path: Path) -> LLMResponse:
        mime_type = IMAGE_MIME_TYPES.get(file_path.suffix.lower())
        if mime_type is None:
            raise UnsupportedFileError(f"不支持的图片格式: {file_path.suffix or '(无扩展名)'}")

        try:
            image_bytes = file_path.read_bytes()
        except OSError as exc:
            raise DocumentUnreadableError(f"读取图片失败: {exc}") from exc
        if not image_bytes:
            raise DocumentUnreadableError(f"图片内容为空: {file_path}")

        return self._client.chat_vision(
            system=self._prompt.system,
            user=IMAGE_USER_PROMPT,
            image=image_bytes,
            mime_type=mime_type,
        )
