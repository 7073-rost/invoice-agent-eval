from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from invoice_agent.schemas import InvoiceData


def test_normalizes_chinese_date_and_currency_amount():
    invoice = InvoiceData.model_validate(
        {
            "invoice_number": " 24312000000012345678 ",
            "issue_date": "2024年03月08日",
            "buyer_name": "北京某某科技有限公司",
            "seller_name": "上海某某商贸有限公司",
            "total_amount": "￥1,234.56",
        }
    )

    assert invoice.invoice_number == "24312000000012345678"
    assert invoice.issue_date == date(2024, 3, 8)
    assert invoice.total_amount == Decimal("1234.56")


def test_missing_fields_default_to_none():
    invoice = InvoiceData.model_validate({})

    assert invoice.invoice_number is None
    assert invoice.issue_date is None
    assert invoice.total_amount is None


def test_invalid_date_raises_validation_error():
    with pytest.raises(ValidationError):
        InvoiceData.model_validate({"issue_date": "不是日期"})
