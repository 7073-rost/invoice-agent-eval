"""模型计价表（示例单价，单位：人民币元 / 百万 Token）。

供应商价格会调整，跑批前请按官网最新价格更新本表。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_cny_per_million: float
    output_cny_per_million: float


MODEL_PRICING: dict[str, ModelPricing] = {
    "deepseek-chat": ModelPricing(input_cny_per_million=2.0, output_cny_per_million=8.0),
    "qwen3-vl-plus": ModelPricing(input_cny_per_million=1.0, output_cny_per_million=10.0),
    "qwen3.8-max": ModelPricing(input_cny_per_million=12.0, output_cny_per_million=36.0),
    "qwen-vl-max": ModelPricing(input_cny_per_million=3.0, output_cny_per_million=9.0),
    "qwen-max": ModelPricing(input_cny_per_million=2.4, output_cny_per_million=9.6),
    "gpt-4o": ModelPricing(input_cny_per_million=18.0, output_cny_per_million=72.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    return (
        input_tokens / 1_000_000 * pricing.input_cny_per_million
        + output_tokens / 1_000_000 * pricing.output_cny_per_million
    )
