"""运行时配置：环境变量 > .env > 默认值。"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # 抽取链路
    text_model: str = "deepseek-chat"
    vision_model: str = "qwen3.8-max"

    # 评测裁判与 Prompt 反思模型
    judge_model: str = "deepseek-chat"
    refiner_model: str = "deepseek-chat"

    temperature: float = 0.0
    request_timeout: float = 60.0
    trace_log_path: str = "logs/traces.jsonl"

    deepseek_api_key: SecretStr | None = None
    dashscope_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # 业务空间专属接入域名；默认使用百炼北京公共域名
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    def api_key_for(self, provider: str) -> str | None:
        """按供应商读取 API Key，新增供应商时在此补充映射即可。"""
        key_map = {
            "deepseek": self.deepseek_api_key,
            "dashscope": self.dashscope_api_key,
            "openai": self.openai_api_key,
        }
        secret = key_map.get(provider)
        return secret.get_secret_value() if secret else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
