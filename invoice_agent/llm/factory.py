"""模型注册表与客户端工厂：切换模型只改这里或 .env。"""

from __future__ import annotations

from dataclasses import dataclass

from invoice_agent.config import Settings, get_settings
from invoice_agent.errors import ConfigurationError
from invoice_agent.llm.base import BaseLLMClient
from invoice_agent.llm.openai_compat import OpenAICompatibleClient


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    base_url: str
    supports_vision: bool = False


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "deepseek-chat": ModelSpec(
        provider="deepseek",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    ),
    "qwen3.8-max": ModelSpec(
        provider="dashscope",
        model="qwen3.8-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        supports_vision=True,
    ),
    "qwen-max": ModelSpec(
        provider="dashscope",
        model="qwen-max",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "gpt-4o": ModelSpec(
        provider="openai",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
        supports_vision=True,
    ),
}


def validate_model_config(model_name: str, settings: Settings | None = None) -> ModelSpec:
    """只校验模型是否注册、Key 是否存在，不创建网络客户端。"""
    settings = settings or get_settings()
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ConfigurationError(f"模型 {model_name!r} 未注册，可用模型: {available}")

    if not settings.api_key_for(spec.provider):
        env_name = f"{spec.provider.upper()}_API_KEY"
        raise ConfigurationError(f"缺少 {spec.provider} 的 API Key，请设置环境变量 {env_name}")
    return spec


def create_llm_client(model_name: str, settings: Settings | None = None) -> BaseLLMClient:
    settings = settings or get_settings()
    spec = validate_model_config(model_name, settings)
    api_key = settings.api_key_for(spec.provider)
    base_url = (
        settings.dashscope_base_url
        if spec.provider == "dashscope"
        else spec.base_url
    )
    return OpenAICompatibleClient(
        model=spec.model,
        api_key=api_key or "",
        base_url=base_url,
        timeout=settings.request_timeout,
        temperature=settings.temperature,
    )
