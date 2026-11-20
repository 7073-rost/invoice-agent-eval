"""抽取提示词与版本管理。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """你是票据信息抽取引擎。只输出一个 JSON 对象，不要输出 Markdown 代码块、解释或多余文字。

需要抽取的字段：
- invoice_number: 发票号码，字符串
- issue_date: 开票日期，格式 YYYY-MM-DD
- buyer_name: 购买方名称全称
- seller_name: 销售方名称全称
- total_amount: 价税合计（小写金额），只保留数字和小数点

输出示例：
{"invoice_number": "24312000000012345678", "issue_date": "2024-03-08", "buyer_name": "北京某某科技有限公司", "seller_name": "上海某某商贸有限公司", "total_amount": "1234.56"}

字段缺失时返回 null，禁止猜测或编造。"""

TEXT_USER_PROMPT = """以下是从 PDF 票据中提取的文本，请抽取字段：
<document>
{document}
</document>"""

IMAGE_USER_PROMPT = "请识别这张票据图片并抽取字段。"


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system: str


DEFAULT_PROMPT = PromptTemplate(version=PROMPT_VERSION, system=SYSTEM_PROMPT)


def infer_prompt_version(path: Path) -> str:
    match = re.search(r"(v\d+(?:\.\d+)*)", path.stem)
    return match.group(1) if match else path.stem


def load_prompt_file(path: Path, version: str | None = None) -> PromptTemplate:
    system = Path(path).read_text(encoding="utf-8").strip()
    if not system:
        raise ValueError(f"Prompt 文件为空: {path}")
    return PromptTemplate(version=version or infer_prompt_version(Path(path)), system=system)
