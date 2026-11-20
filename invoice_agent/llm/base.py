"""大模型客户端抽象与统一返回结构。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """一次模型调用的完整结果，供 Token 统计、成本核算与 Trace 使用。"""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class BaseLLMClient(ABC):
    """统一的大模型调用接口，替换模型/供应商时只需新增实现类。"""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """当前使用的模型名称。"""

    @abstractmethod
    def chat_text(self, *, system: str, user: str) -> LLMResponse:
        """纯文本对话，返回模型原始输出。"""

    @abstractmethod
    def chat_vision(self, *, system: str, user: str, image: bytes, mime_type: str) -> LLMResponse:
        """图文对话，返回模型原始输出。"""
