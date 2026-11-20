"""基于 OpenAI 兼容协议的客户端，可对接 DeepSeek、DashScope、OpenAI 等供应商。"""

from __future__ import annotations

import base64
import time

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from invoice_agent.errors import ExtractionError, LLMRequestError
from invoice_agent.llm.base import BaseLLMClient, LLMResponse


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str,
        timeout: float = 60.0,
        temperature: float = 0.0,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    @property
    def model_name(self) -> str:
        return self._model

    def chat_text(self, *, system: str, user: str) -> LLMResponse:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._complete(messages, json_mode=True)

    def chat_vision(self, *, system: str, user: str, image: bytes, mime_type: str) -> LLMResponse:
        # 部分视觉模型对 system role 支持不一致，这里把指令合并进 user 文本
        data_url = f"data:{mime_type};base64,{base64.b64encode(image).decode('ascii')}"
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": f"{system}\n\n{user}"},
                ],
            }
        ]
        return self._complete(messages)

    def _extra_body(self) -> dict[str, object] | None:
        """qwen3.8-max 默认开启思考模式，抽取任务关闭以稳定输出 JSON。"""
        if self._model.startswith("qwen3.8-max"):
            return {"enable_thinking": False}
        return None

    def _complete(
        self,
        messages: list[ChatCompletionMessageParam],
        *,
        json_mode: bool = False,
    ) -> LLMResponse:
        started = time.perf_counter()
        try:
            if json_mode:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    response_format={"type": "json_object"},
                    extra_body=self._extra_body(),
                )
            else:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    extra_body=self._extra_body(),
                )
        except OpenAIError as exc:
            raise LLMRequestError(f"调用模型 {self._model} 失败: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise ExtractionError(f"模型 {self._model} 返回了空内容")

        usage = response.usage
        return LLMResponse(
            content=content,
            model=self._model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
