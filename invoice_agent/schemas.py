"""票据抽取结果模型。"""

from __future__ import annotations

import re
from datetime import date
from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, ConfigDict, Field, field_validator


class InvoiceData(BaseModel):
    """结构化票据字段。

    字段缺失时保留 None，避免模型臆造数据；同时容忍模型返回的中文日期、
    带货币符号/千分位的金额等常见格式。
    """

    model_config = ConfigDict(extra="ignore")

    invoice_number: str | None = Field(default=None, description="发票号码")
    issue_date: date | None = Field(default=None, description="开票日期")
    buyer_name: str | None = Field(default=None, description="购买方名称")
    seller_name: str | None = Field(default=None, description="销售方名称")
    total_amount: Decimal | None = Field(default=None, description="价税合计金额")

    @field_validator("invoice_number", "buyer_name", "seller_name", mode="before")
    @classmethod
    def _normalize_text(cls, value: object) -> object:
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("issue_date", mode="before")
    @classmethod
    def _normalize_date(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text:
            return None
        match = re.search(r"(\d{4})\s*[-/年.]\s*(\d{1,2})\s*[-/月.]\s*(\d{1,2})", text)
        if match:
            year, month, day = (int(part) for part in match.groups())
            return date(year, month, day)
        if re.fullmatch(r"\d{8}", text):
            return date(int(text[:4]), int(text[4:6]), int(text[6:]))
        return text

    @field_validator("total_amount", mode="before")
    @classmethod
    def _normalize_amount(cls, value: object) -> object:
        if value is None or isinstance(value, Decimal):
            return value
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        if not isinstance(value, str):
            return value
        cleaned = re.sub(r"[^\d.\-]", "", value)
        if not cleaned or cleaned in {"-", ".", "-."}:
            return None
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return value
