import pytest

from invoice_agent.errors import ExtractionError
from invoice_agent.extractors.base import parse_json_object


def test_parse_plain_json():
    payload = parse_json_object(
        '{"invoice_number": "123", "issue_date": "2024-03-08"}'
    )

    assert payload["invoice_number"] == "123"


def test_parse_json_wrapped_in_code_fence_and_prose():
    raw = '抽取结果如下：\n```json\n{"invoice_number": "123"}\n```\n请核对。'

    assert parse_json_object(raw) == {"invoice_number": "123"}


def test_parse_without_json_raises():
    with pytest.raises(ExtractionError):
        parse_json_object("模型没有返回结构化结果")
