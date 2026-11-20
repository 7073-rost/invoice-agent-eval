"""命令行入口：python -m invoice_agent <file>"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from invoice_agent.config import get_settings
from invoice_agent.errors import InvoiceAgentError
from invoice_agent.router import InvoiceRouter
from invoice_agent.tracing import TraceLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="invoice-agent", description="票据信息抽取 Agent")
    parser.add_argument("file", help="票据文件路径（支持 pdf/png/jpg/jpeg）")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    router = InvoiceRouter(
        settings=settings,
        trace_logger=TraceLogger(settings.trace_log_path),
    )
    try:
        result = router.extract(args.file)
    except (InvoiceAgentError, FileNotFoundError) as exc:
        print(f"抽取失败: {exc}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
