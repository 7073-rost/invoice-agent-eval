"""Streamlit 前端：多模态票据 Agent 评测工作台。

启动：
    .\\.venv\\Scripts\\streamlit.exe run app.py
"""

from __future__ import annotations

import base64
import html
import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from invoice_agent.config import get_settings
from invoice_agent.errors import InvoiceAgentError
from invoice_agent.evaluation.report import (
    build_failure_frame,
    build_field_frame,
    compute_summary,
    write_run_report,
)
from invoice_agent.evaluation.runner import (
    INVOICE_FIELDS,
    discover_invoice_files,
    run_evaluation,
)
from invoice_agent.judge import LLMAsJudge
from invoice_agent.llm.factory import (
    MODEL_REGISTRY,
    create_llm_client,
    validate_model_config,
)
from invoice_agent.prompts import DEFAULT_PROMPT
from invoice_agent.router import IMAGE_SUFFIXES, PDF_SUFFIXES, InvoiceRouter
from invoice_agent.tracing import FAILURE_DESCRIPTIONS, TraceLogger

FIELD_LABELS = {
    "invoice_number": "发票号码",
    "issue_date": "开票日期",
    "buyer_name": "购买方名称",
    "seller_name": "销售方名称",
    "total_amount": "价税合计",
}

MODEL_DISPLAY_NAMES = {
    "deepseek-chat": "DeepSeek",
    "qwen3.8-max": "Qwen3.8-Max",
    "qwen-max": "Qwen-Max",
    "gpt-4o": "GPT-4o",
}
FAILURE_LABELS = {item.value: text for item, text in FAILURE_DESCRIPTIONS.items()}
TEXT_MODELS = sorted(MODEL_REGISTRY)
VISION_MODELS = sorted(
    name for name, spec in MODEL_REGISTRY.items() if spec.supports_vision
)

st.set_page_config(
    page_title="Agent Eval Dashboard",
    page_icon="📊",
    layout="wide",
)

CUSTOM_CSS = """
<style>
/* 1. 隐藏 Streamlit 默认菜单与页脚 */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stStatusWidget"] { display: none; }

/* 2. 全局背景与纤细标题 */
.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #0E1117;
}
[data-testid="stHeader"] {
    background-color: rgba(14, 17, 23, 0.85);
    backdrop-filter: blur(8px);
}
h1 {
    font-weight: 300 !important;
    letter-spacing: -0.035em;
}

/* 3. 内容区最大宽度 1200px 并居中 */
[data-testid="stMainBlockContainer"],
section.main .block-container {
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 3rem;
    padding-bottom: 4rem;
}

/* 4. 卡片容器：圆角、边框、微背景与淡阴影 */
.st-key-playground-input,
.st-key-playground-output {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25), 0 6px 18px rgba(0, 0, 0, 0.16);
    padding: 2px;
}
.st-key-playground-input .stVerticalBlock,
.st-key-playground-output .stVerticalBlock {
    border: none !important;
    background: transparent !important;
}

/* 5. 主按钮：高级蓝渐变 + 悬停放大 */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #1F6FEB 0%, #388BFD 100%);
    border: 1px solid #1F6FEB;
    border-radius: 8px;
    color: #ffffff;
    font-weight: 600;
    transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    transform: scale(1.02);
    filter: brightness(1.06);
    box-shadow: 0 6px 18px rgba(31, 111, 235, 0.35);
}
.stButton > button[kind="primary"]:active,
button[data-testid="stBaseButton-primary"]:active {
    transform: scale(0.99);
}

/* 6. 指标卡片 */
[data-testid="stMetric"] {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 1rem 1.15rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.25);
    transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
    will-change: transform;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px) scale(1.015);
    border-color: rgba(31, 111, 235, 0.65);
    box-shadow: 0 12px 32px rgba(31, 111, 235, 0.16);
}
[data-testid="stMetricLabel"] p {
    color: #8B949E;
    letter-spacing: 0.03em;
}
[data-testid="stMetricValue"] {
    letter-spacing: -0.01em;
}

/* 7. 右侧等待占位 */
.empty-state {
    border: 1px dashed #30363D;
    border-radius: 8px;
    background: rgba(22, 27, 34, 0.55);
    padding: 3.2rem 1rem;
    text-align: center;
    color: #8B949E;
    margin: 0.5rem 0;
}
.empty-state__title {
    font-size: 1.05rem;
    font-weight: 500;
    color: #C9D1D9;
}
.empty-state__hint {
    font-size: 0.85rem;
    margin-top: 0.35rem;
    color: #8B949E;
}

/* 8. 全局字体间距与行高，提升呼吸感 */
html,
body,
.stApp {
    letter-spacing: 0.012em;
}
p,
li,
label,
[data-testid="stMarkdownContainer"] {
    line-height: 1.75;
    letter-spacing: 0.012em;
}
h1, h2, h3, h4 {
    letter-spacing: -0.01em;
    line-height: 1.3;
}

/* 9. 单次运行指标：缩小字号并禁止省略号截断 */
.st-key-playground-output [data-testid="stMetric"] {
    min-width: 0;
    overflow: visible !important;
}
.st-key-playground-output [data-testid="stMetricValue"] {
    font-size: 1.3rem !important;
    line-height: 1.25 !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
.st-key-playground-output [data-testid="stMetricValue"] > div {
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
}
.st-key-playground-output [data-testid="stMetricLabel"] p {
    font-size: 0.78rem !important;
    white-space: nowrap;
}

/* 10. 隐藏上传器原生文件信息，只保留清除按钮（Streamlit 1.67 使用 stFileChip） */
[data-testid="stFileChip"] > :not([data-testid="stFileChipDeleteBtn"]) {
    display: none !important;
}
[data-testid="stFileChip"] {
    min-width: 0 !important;
    width: fit-content !important;
    padding: 0 !important;
    background: transparent !important;
    gap: 0 !important;
}
[data-testid="stFileChips"] {
    justify-content: flex-end;
}
[data-testid="stFileChips"] button[aria-label="Add files"] {
    display: none !important;
}

/* 11. 发票缩略图：限宽限高、居中、描边 */
.st-key-playground-input [data-testid="stImageContainer"] {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
.st-key-playground-input div:has(> [data-testid="stImageContainer"]) {
    margin-left: auto !important;
    margin-right: auto !important;
}
.st-key-playground-input [data-testid="stImageContainer"] img {
    width: auto !important;
    height: auto !important;
    max-width: 300px !important;
    max-height: 360px !important;
    border: 1px solid #30363D;
    border-radius: 8px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
}

/* 12. 运行状态与键值对结果表 */
.run-status {
    text-align: right;
    font-size: 0.86rem;
    font-weight: 600;
    padding-top: 0.35rem;
    white-space: nowrap;
}
.run-status.ok { color: #3FB950; }
.run-status.fail { color: #F85149; }
.kv-table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.35rem 0 0.75rem;
    font-size: 0.95rem;
}
.kv-table th {
    text-align: left;
    color: #8B949E;
    font-weight: 500;
    padding: 0.45rem 0.75rem;
    border-bottom: 1px solid #30363D;
}
.kv-table td {
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid #21262D;
    vertical-align: top;
}
.kv-table tr:last-child td { border-bottom: none; }
.kv-table td:first-child {
    width: 35%;
    color: #8B949E;
}
.kv-table td:last-child {
    color: #E6EDF3;
    font-weight: 500;
    word-break: break-all;
}
.null-value {
    color: #6E7681;
    font-style: italic;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.title("多模态票据 Agent 评测工作台")

tab1, tab2, tab3 = st.tabs(
    [
        "单次运行 (Playground)",
        "批量评测看板 (Eval Dashboard)",
        "Bad Case 归因 (Error Analysis)",
    ]
)


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _default_index(options: list[str], preferred: str | None) -> int:
    return options.index(preferred) if preferred in options else 0


@st.cache_data(show_spinner=False)
def extract_single(file_bytes: bytes, file_name: str, model: str) -> dict:
    """单张票据抽取；相同文件 + 相同模型命中缓存，避免调试时重复扣费。"""
    settings = get_settings()
    suffix = Path(file_name).suffix.lower()
    override_key = "vision_model" if suffix in IMAGE_SUFFIXES else "text_model"
    runtime_settings = settings.model_copy(update={override_key: model})

    try:
        validate_model_config(model, runtime_settings)
    except InvoiceAgentError as exc:
        return {"ok": False, "config_error": True, "error": str(exc)}

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / (Path(file_name).name or f"upload{suffix}")
        path.write_bytes(file_bytes)
        router = InvoiceRouter(
            settings=runtime_settings,
            trace_logger=TraceLogger(runtime_settings.trace_log_path),
        )
        outcome = router.extract_outcome(path)

    return {
        "ok": outcome.data is not None,
        "config_error": False,
        "data": outcome.data.model_dump(mode="json") if outcome.data else None,
        "raw_output": outcome.raw_output,
        "error": outcome.error,
        "failure_type": outcome.failure_type.value,
        "failure_description": FAILURE_LABELS.get(outcome.failure_type.value, ""),
        "model": outcome.model,
        "prompt_version": outcome.prompt_version,
        "latency_ms": round(outcome.latency_ms, 2),
        "input_tokens": outcome.input_tokens,
        "output_tokens": outcome.output_tokens,
        "cost_cny": round(outcome.cost_cny, 6),
    }


def _escape(value: object) -> str:
    if value is None:
        return '<span class="empty">null</span>'
    return html.escape(str(value))


def _failed_field_labels(case: dict) -> str:
    failed = [
        FIELD_LABELS.get(name, name)
        for name, matched in (case.get("field_matches") or {}).items()
        if not matched
    ]
    return "、".join(failed) if failed else "-"


def _case_cost(case: dict) -> float:
    return float(case.get("extraction_cost_cny") or 0.0) + float(
        case.get("judge_cost_cny") or 0.0
    )


def _format_cost(value: float) -> str:
    """固定四位小数，保证 ¥ 金额在指标卡中完整显示。"""
    return f"¥{value:.4f}"


def _raw_matches_data(raw_output: str | None, data: dict) -> bool:
    """判断原始输出与结构化结果是否完全一致，用于隐藏冗余面板。"""
    if not raw_output or not isinstance(data, dict):
        return False
    try:
        parsed = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return False
    if not isinstance(parsed, dict) or set(parsed) != set(data):
        return False
    return all(
        (None if parsed.get(key) is None else str(parsed.get(key)).strip())
        == (None if value is None else str(value).strip())
        for key, value in data.items()
    )


# ---------------------------------------------------------------------------
# Tab 1：单次运行
# ---------------------------------------------------------------------------


def render_playground() -> None:
    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True, key="playground-input"):
            st.markdown("#### 上传票据")

            uploaded = st.file_uploader(
                "上传票据文件",
                type=["png", "jpg", "jpeg", "pdf"],
                help="支持 PDF 与 PNG/JPG；PDF 走文本链路，图片走视觉链路。",
            )
            suffix = Path(uploaded.name).suffix.lower() if uploaded else ""
            is_image = suffix in IMAGE_SUFFIXES

            st.caption("推荐使用 Qwen3.8-Max 处理图像；PDF 文本推荐 DeepSeek。")
            settings = get_settings()
            if is_image:
                model = st.selectbox(
                    "视觉模型",
                    VISION_MODELS,
                    index=_default_index(VISION_MODELS, settings.vision_model),
                    key="playground_model_image",
                )
            else:
                model = st.selectbox(
                    "文本模型",
                    TEXT_MODELS,
                    index=_default_index(TEXT_MODELS, settings.text_model),
                    key="playground_model_text",
                    help="上传图片后会自动切换为视觉模型列表。",
                )

            # 文件或模型变化时清空旧结果，避免展示过期数据
            run_sig = (
                f"{uploaded.name}:{uploaded.size}:{model}"
                if uploaded is not None
                else None
            )
            if st.session_state.get("playground_run_sig") != run_sig:
                st.session_state["playground_run_sig"] = run_sig
                st.session_state["playground_result"] = None

            spec = MODEL_REGISTRY.get(model)
            if spec is not None and not settings.api_key_for(spec.provider):
                st.warning(
                    f"未检测到 {spec.provider.upper()}_API_KEY，调用会失败；"
                    "请在项目根目录的 .env 中配置。"
                )

            if uploaded is not None:
                if is_image:
                    st.image(uploaded.getvalue(), width=300)
                else:
                    st.caption("PDF 已就绪，抽取时将自动解析文本层。")
            else:
                st.caption("请先上传一张票据，再点击开始抽取。")

            clicked = st.button("开始抽取", type="primary")

    with right:
        with st.container(border=True, key="playground-output"):
            header_left, header_right = st.columns([3, 1])
            header_left.markdown("#### 抽取结果")
            status_slot = header_right.empty()

            if clicked:
                if uploaded is None:
                    status_slot.markdown(
                        '<div class="run-status fail">● 运行失败</div>',
                        unsafe_allow_html=True,
                    )
                    st.error("请先上传票据文件。")
                    return
                else:
                    with st.spinner("AI 正在解析票据..."):
                        try:
                            result = extract_single(
                                uploaded.getvalue(), uploaded.name, model
                            )
                        except Exception as exc:  # 前端兜底，避免整页崩溃
                            result = {
                                "ok": False,
                                "config_error": False,
                                "error": str(exc),
                                "failure_type": "Type_G",
                                "failure_description": FAILURE_LABELS.get(
                                    "Type_G", ""
                                ),
                                "model": model,
                            }
                    st.session_state["playground_result"] = result

            result = st.session_state.get("playground_result")
            if result is None:
                st.markdown(
                    """
                    <div class="empty-state">
                        <div class="empty-state__title">等待输入数据</div>
                        <div class="empty-state__hint">
                            上传票据并点击「开始抽取」，结果会显示在这里
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                return

            if result.get("ok"):
                status_slot.markdown(
                    '<div class="run-status ok">● 运行成功</div>',
                    unsafe_allow_html=True,
                )
            else:
                status_slot.markdown(
                    '<div class="run-status fail">● 运行失败</div>',
                    unsafe_allow_html=True,
                )

            if result.get("config_error"):
                st.error(result.get("error", "配置错误"))
                return

            if result.get("ok"):
                data = result.get("data") or {}
                metric_latency, metric_token, metric_cost = st.columns(
                    3, gap="small"
                )
                metric_latency.metric(
                    "耗时",
                    f"{result.get('latency_ms', 0) / 1000:.2f}s",
                )
                metric_token.metric(
                    "Token 用量",
                    f"{result.get('input_tokens', 0) + result.get('output_tokens', 0):,}",
                    help=(
                        f"输入 {result.get('input_tokens', 0)} / "
                        f"输出 {result.get('output_tokens', 0)}"
                    ),
                )
                metric_cost.metric(
                    "成本",
                    _format_cost(float(result.get("cost_cny") or 0)),
                    help="按输入/输出 Token 与单价估算",
                )

                ordered_keys = [name for name in INVOICE_FIELDS if name in data]
                ordered_keys += [
                    name for name in data if name not in INVOICE_FIELDS
                ]
                rows = []
                for name in ordered_keys:
                    value = data.get(name)
                    label = html.escape(FIELD_LABELS.get(name, name))
                    cell = (
                        '<span class="null-value">null</span>'
                        if value is None
                        else html.escape(str(value))
                    )
                    rows.append(f"<tr><td>{label}</td><td>{cell}</td></tr>")
                st.markdown(
                    '<table class="kv-table">'
                    "<thead><tr><th>字段</th><th>提取值</th></tr></thead>"
                    f"<tbody>{''.join(rows)}</tbody>"
                    "</table>",
                    unsafe_allow_html=True,
                )

                json_payload = {name: data.get(name) for name in ordered_keys}
                json_text = json.dumps(
                    json_payload, ensure_ascii=False, indent=2
                )
                encoded = base64.b64encode(
                    json_text.encode("utf-8")
                ).decode("ascii")
                st.iframe(
                    f"""
                    <html>
                    <head>
                    <style>
                        html, body {{ margin: 0; padding: 0; background: transparent; }}
                        body {{ display: flex; justify-content: flex-end; }}
                        .copy-btn {{
                            background: linear-gradient(135deg, #1F6FEB 0%, #388BFD 100%);
                            border: 1px solid #1F6FEB;
                            color: #ffffff;
                            border-radius: 8px;
                            padding: 6px 14px;
                            font-size: 13px;
                            font-weight: 600;
                            cursor: pointer;
                            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                            transition: transform 160ms ease, filter 160ms ease;
                        }}
                        .copy-btn:hover {{ transform: scale(1.02); filter: brightness(1.06); }}
                        .copy-btn:active {{ transform: scale(0.99); }}
                        .copy-btn.copied {{ background: #238636; border-color: #238636; }}
                    </style>
                    </head>
                    <body>
                        <button class="copy-btn" onclick="copyJson(this)">复制 JSON</button>
                        <script>
                            function copyJson(btn) {{
                                const raw = atob("{encoded}");
                                const bytes = Uint8Array.from(raw, c => c.charCodeAt(0));
                                const text = new TextDecoder().decode(bytes);
                                const done = () => {{
                                    btn.textContent = "已复制";
                                    btn.classList.add("copied");
                                }};
                                if (navigator.clipboard && navigator.clipboard.writeText) {{
                                    navigator.clipboard.writeText(text)
                                        .then(done)
                                        .catch(() => fallbackCopy(text, done));
                                }} else {{
                                    fallbackCopy(text, done);
                                }}
                            }}
                            function fallbackCopy(text, done) {{
                                const ta = document.createElement("textarea");
                                ta.value = text;
                                ta.style.position = "fixed";
                                ta.style.opacity = "0";
                                document.body.appendChild(ta);
                                ta.select();
                                document.execCommand("copy");
                                document.body.removeChild(ta);
                                done();
                            }}
                        </script>
                    </body>
                    </html>
                    """,
                    height=46,
                )

                st.caption(
                    f"Prompt 版本 {result.get('prompt_version') or '—'}"
                )
                raw_output = result.get("raw_output") or ""
                if raw_output and not _raw_matches_data(raw_output, data):
                    with st.expander("模型原始输出"):
                        st.code(raw_output, language="json")
            else:
                st.error(f"抽取失败：{result.get('error', '未知错误')}")
                st.caption(
                    f"失败类型 {result.get('failure_type', '')} · "
                    f"{result.get('failure_description', '')}"
                )
                if result.get("raw_output"):
                    with st.expander("模型原始输出"):
                        st.code(result["raw_output"], language="json")


# ---------------------------------------------------------------------------
# Tab 2：批量评测看板
# ---------------------------------------------------------------------------


def _run_batch(
    *,
    data_dir: str,
    ground_truth_path: str,
    workers: int,
    text_model: str,
    vision_model: str,
    use_judge: bool,
    include_unlabeled: bool = False,
) -> None:
    data_path = Path(data_dir)
    truth_path = Path(ground_truth_path)

    if not data_path.is_dir():
        st.error(f"票据文件夹不存在：{data_path}")
        return
    if not truth_path.is_file():
        st.error(f"标准答案文件不存在：{truth_path}")
        return

    try:
        files = discover_invoice_files(data_path)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    if not files:
        st.error(f"未在 {data_path} 发现 pdf/png/jpg/jpeg 文件")
        return

    settings = get_settings()
    runtime_settings = settings.model_copy(
        update={"text_model": text_model, "vision_model": vision_model}
    )

    try:
        if any(path.suffix.lower() in PDF_SUFFIXES for path in files):
            validate_model_config(text_model, runtime_settings)
        if any(path.suffix.lower() in IMAGE_SUFFIXES for path in files):
            validate_model_config(vision_model, runtime_settings)

        judge = None
        if use_judge:
            validate_model_config(settings.judge_model, runtime_settings)
            judge = LLMAsJudge(create_llm_client(settings.judge_model, runtime_settings))
    except InvoiceAgentError as exc:
        st.error(str(exc))
        return

    st.caption(
        f"发现 {len(files)} 个票据文件 · 并发 {workers} · "
        f"裁判 {'开启' if use_judge else '关闭'} · "
        f"无标注 {'抽取' if include_unlabeled else '跳过'}"
    )
    progress = st.progress(0.0, text="准备并发跑批...")

    def on_progress(done: int, total: int) -> None:
        progress.progress(done / total, text=f"并发跑批中 {done}/{total}")

    run_dir = (
        Path("eval_runs")
        / f"{datetime.now():%Y%m%d_%H%M%S}_{DEFAULT_PROMPT.version}"
    )

    try:
        with st.spinner("正在并发跑批，请稍候..."):
            run = run_evaluation(
                data_dir=data_path,
                ground_truth_path=truth_path,
                settings=runtime_settings,
                prompt=DEFAULT_PROMPT,
                workers=workers,
                judge=judge,
                trace_log_path=run_dir / "traces.jsonl",
                progress_callback=on_progress,
                include_unlabeled=include_unlabeled,
            )
            write_run_report(run, run_dir)
    except Exception as exc:  # 跑批失败时给出明确错误，而不是白屏
        progress.empty()
        st.error(f"跑批失败：{exc}")
        return

    progress.progress(1.0, text="跑批完成")

    summary = compute_summary(run)
    fields = build_field_frame(run.cases)
    failures = build_failure_frame(run.cases)

    payload = {
        "run_dir": str(run_dir),
        "label": f"{run.text_model} + {run.vision_model}",
        "started_at": run.started_at,
        "duration_seconds": round(run.duration_seconds, 2),
        "num_cases": len(run.cases),
        "num_skipped": len(run.skipped_files),
        "skipped_files": run.skipped_files,
        "summary": summary,
        "fields": fields.to_dict(orient="records"),
        "failures": failures.to_dict(orient="records"),
        "cases": [case.to_dict(include_raw=True) for case in run.cases],
        "data_dir": str(data_path.resolve()),
        "ground_truth_path": str(truth_path.resolve()),
        "prompt_version": run.prompt.version,
        "text_model": run.text_model,
        "vision_model": run.vision_model,
        "judge_model": run.judge_model,
        "workers": run.workers,
    }

    history = st.session_state.setdefault("eval_history", [])
    history.append(payload)
    st.session_state["eval_history"] = history[-10:]


def _format_delta(
    current: float,
    previous: float | None,
    *,
    percent: bool = False,
    suffix: str = "",
    precision: int = 4,
) -> str | None:
    if previous is None:
        return None
    diff = current - previous
    if percent:
        return f"{diff:+.1%} 较上一版本"
    return f"{diff:+.{precision}f}{suffix} 较上一版本"


def _style_chart(fig, *, height: int = 360) -> None:
    """统一 Plotly 深色风格：透明背景、无白边、细网格。"""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8),
        height=height,
        font=dict(color="#ededed", size=12),
        hoverlabel=dict(
            bgcolor="#111111",
            bordercolor="#262626",
            font=dict(color="#ededed"),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor="#262626",
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        automargin=True,
    )


def render_dashboard() -> None:
    st.subheader("跑批配置")

    col1, col2, col3 = st.columns([2, 2, 1])
    data_dir = col1.text_input(
        "票据文件夹",
        value="data/invoices",
        key="eval_data_dir",
    )
    ground_truth_path = col2.text_input(
        "标准答案 JSON",
        value="data/eval/ground_truth.json",
        key="eval_ground_truth",
    )
    workers = col3.number_input(
        "并发线程",
        min_value=1,
        max_value=16,
        value=4,
        step=1,
        key="eval_workers",
    )

    col4, col5, col6 = st.columns([1.5, 1.5, 1])
    settings = get_settings()
    text_model = col4.selectbox(
        "PDF 文本模型",
        TEXT_MODELS,
        index=_default_index(TEXT_MODELS, settings.text_model),
        key="eval_text_model",
    )
    vision_model = col5.selectbox(
        "图片视觉模型",
        VISION_MODELS,
        index=_default_index(VISION_MODELS, settings.vision_model),
        key="eval_vision_model",
    )
    use_judge = col6.toggle(
        "启用 LLM-as-a-Judge",
        value=True,
        key="eval_use_judge",
        help="对格式不同但语义一致的字段调用裁判复核。",
    )

    include_unlabeled = st.checkbox(
        "包含无标准答案文件（仅抽取，不计入准确率）",
        value=False,
        key="eval_include_unlabeled",
        help=(
            "勾选后，缺少标准答案的图片也会调用模型抽取，结果写入 cases.json，"
            "但不参与 Exact Match 与字段级指标计算。"
        ),
    )

    if st.button("开始全量跑批", type="primary", key="eval_run_button"):
        _run_batch(
            data_dir=data_dir,
            ground_truth_path=ground_truth_path,
            workers=int(workers),
            text_model=text_model,
            vision_model=vision_model,
            use_judge=use_judge,
            include_unlabeled=include_unlabeled,
        )

    history: list[dict] = st.session_state.get("eval_history", [])
    if not history:
        st.info("暂无跑批结果。确认数据路径与模型后，点击上方按钮开始跑批。")
        return

    current = history[-1]
    previous = history[-2] if len(history) > 1 else None
    summary = current["summary"]
    cases = current["cases"]
    num_cases = current["num_cases"]

    avg_latency_s = (
        sum(case["latency_ms"] for case in cases) / num_cases / 1000
        if num_cases
        else 0.0
    )
    avg_cost = summary["total_cost_cny"] / num_cases if num_cases else 0.0
    p99_latency_s = summary["p99_latency_ms"] / 1000

    prev_avg_latency = None
    if previous and previous["num_cases"]:
        prev_avg_latency = (
            sum(case["latency_ms"] for case in previous["cases"])
            / previous["num_cases"]
            / 1000
        )

    st.divider()
    st.subheader("核心指标")
    total_cost = summary["total_cost_cny"]
    prev_total_cost = previous["summary"]["total_cost_cny"] if previous else None

    metric1, metric2, metric3 = st.columns(3)
    labeled_count = summary.get("num_labeled", num_cases)
    unlabeled_count = summary.get("num_unlabeled", 0)
    has_labels = labeled_count > 0
    prev_has_labels = bool(
        previous and previous["summary"].get("num_labeled", previous["num_cases"]) > 0
    )
    metric1.metric(
        "整体准确率 (Exact Match)",
        f"{summary['exact_match_rate']:.1%}" if has_labels else "—",
        delta=_format_delta(
            summary["exact_match_rate"],
            previous["summary"]["exact_match_rate"] if previous and prev_has_labels else None,
            percent=True,
        )
        if has_labels
        else None,
        help=(
            f"标注样本 {labeled_count} 张 · 无标注 {unlabeled_count} 张 · "
            f"跳过 {current['num_skipped']} 张；"
            f"字段平均 F1 {summary['macro_f1']:.1%}"
        ),
    )
    metric2.metric(
        "平均耗时 / 张",
        f"{avg_latency_s:.2f}s",
        delta=_format_delta(
            avg_latency_s, prev_avg_latency, suffix="s", precision=2
        ),
        delta_color="inverse",
        help=f"P99 {p99_latency_s:.2f}s",
    )
    metric3.metric(
        "总消耗成本",
        f"¥{total_cost:.4f}",
        delta=_format_delta(total_cost, prev_total_cost, suffix=" 元"),
        delta_color="inverse",
        help=(
            f"平均 ¥{avg_cost:.4f}/张 · "
            f"抽取 ¥{summary['extraction_cost_cny']:.4f} · "
            f"裁判 ¥{summary['judge_cost_cny']:.4f}"
        ),
    )

    st.caption(
        f"批次 {current['started_at']} · Prompt {current['prompt_version']} · "
        f"模型 {current['label']} · 并发 {current['workers']} · "
        f"总耗时 {current['duration_seconds']:.1f}s · 产物 {current['run_dir']}"
    )
    if current["num_skipped"]:
        st.warning(
            f"有 {current['num_skipped']} 个文件因缺少标准答案被跳过："
            + "、".join(current["skipped_files"][:10])
        )
    if unlabeled_count:
        st.caption(
            f"其中 {unlabeled_count} 张无标准答案，仅执行抽取，"
            "不参与 Exact Match 与字段级指标。"
        )
    if not has_labels:
        st.warning(
            "本批次没有标注样本，准确率与字段指标不可计算；"
            "请到「Bad Case 归因」查看抽取结果。"
        )
    if num_cases == 0:
        st.warning("本批次没有可评测样本：请检查标准答案文件名是否与票据文件对应。")
        return

    st.divider()
    chart_col, failure_col = st.columns([2, 1])

    with chart_col:
        st.subheader("字段级准确率")
        fields_frame = pd.DataFrame(current["fields"])
        fields_frame["字段"] = fields_frame["field"].map(FIELD_LABELS).fillna(
            fields_frame["field"]
        )
        long_frame = fields_frame.melt(
            id_vars=["字段"],
            value_vars=["precision", "recall", "f1"],
            var_name="指标",
            value_name="数值",
        )
        long_frame["指标"] = long_frame["指标"].map(
            {
                "precision": "精确率",
                "recall": "召回率",
                "f1": "F1",
            }
        )
        field_fig = px.bar(
            long_frame,
            x="字段",
            y="数值",
            color="指标",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text_auto=".0%",
        )
        field_fig.update_traces(
            textfont=dict(size=10, color="#ededed"),
            textangle=0,
            cliponaxis=False,
        )
        field_fig.update_yaxes(tickformat=".0%", range=[0, 1.08])
        field_fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                x=0,
                title=None,
                bgcolor="rgba(0,0,0,0)",
            ),
            bargap=0.25,
            bargroupgap=0.08,
        )
        _style_chart(field_fig, height=360)
        st.plotly_chart(
            field_fig,
            width="stretch",
            config={"displayModeBar": False},
        )

        with st.expander("字段明细表"):
            detail = fields_frame[
                ["字段", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "support"]
            ].copy()
            for column in ["precision", "recall", "f1"]:
                detail[column] = (detail[column] * 100).round(2)
            st.dataframe(detail, hide_index=True, width="stretch")

    with failure_col:
        st.subheader("失败归因分布")
        failures = current["failures"]
        if failures:
            failure_frame = pd.DataFrame(failures)
            failure_frame["标签"] = (
                failure_frame["failure_type"]
                + " · "
                + failure_frame["description"]
            )
            failure_frame["占比"] = (
                failure_frame["ratio_of_failures"] * 100
            ).round(1)
            failure_fig = px.bar(
                failure_frame.sort_values("count"),
                x="count",
                y="failure_type",
                orientation="h",
                color="failure_type",
                color_discrete_sequence=px.colors.qualitative.Pastel,
                text="count",
                hover_data={"description": True, "占比": True, "count": False},
            )
            failure_fig.update_traces(
                textposition="outside",
                textfont=dict(color="#ededed"),
                cliponaxis=False,
            )
            failure_fig.update_layout(
                showlegend=False,
                xaxis_title="失败数量",
                yaxis_title=None,
            )
            _style_chart(failure_fig, height=360)
            st.plotly_chart(
                failure_fig,
                width="stretch",
                config={"displayModeBar": False},
            )
            with st.expander("归因明细"):
                st.dataframe(
                    failure_frame[["failure_type", "count", "占比", "description"]],
                    hide_index=True,
                    width="stretch",
                )
        else:
            st.success("本批次没有错误样本")

    st.divider()
    st.subheader("模型准确率对比")

    rows: list[dict] = []
    for run in history[-10:]:
        for case in run.get("cases", []):
            if case.get("exact_match") is None:
                continue
            raw_model = case.get("model") or "未知"
            rows.append(
                {
                    "模型": MODEL_DISPLAY_NAMES.get(raw_model, raw_model),
                    "正确": 1.0 if case.get("exact_match") else 0.0,
                }
            )

    if not rows:
        st.caption("暂无可对比的样本：历史批次没有完成评测的票据。")
        return

    compare_frame = (
        pd.DataFrame(rows)
        .groupby("模型", as_index=False)
        .agg(准确率=("正确", "mean"), 样本数=("正确", "size"))
        .sort_values("准确率", ascending=False)
    )
    model_fig = px.bar(
        compare_frame,
        x="模型",
        y="准确率",
        color="模型",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        text_auto=".1%",
        hover_data={"样本数": True},
    )
    model_fig.update_traces(
        textposition="outside",
        textfont=dict(color="#ededed"),
        cliponaxis=False,
    )
    model_fig.update_layout(
        showlegend=False,
        xaxis_title="模型",
        yaxis_title="整单准确率",
    )
    model_fig.update_yaxes(tickformat=".0%", range=[0, 1.15])
    _style_chart(model_fig, height=380)
    st.plotly_chart(
        model_fig,
        width="stretch",
        config={"displayModeBar": False},
    )
    st.caption(
        "整单 Exact Match 按模型聚合历史批次全部样本；"
        "柱顶为准确率，悬停可查看样本数。"
    )
    if len(compare_frame) < 2:
        st.caption("当前仅有一个模型的样本，补充另一模型的跑批后会形成对比。")


# ---------------------------------------------------------------------------
# Tab 3：Bad Case 归因
# ---------------------------------------------------------------------------


def _render_file_preview(data_dir: str, relative: str) -> None:
    path = Path(data_dir) / relative
    if not path.is_file():
        st.warning(f"原文件不存在：{path}")
        return

    suffix = path.suffix.lower()
    try:
        if suffix in PDF_SUFFIXES:
            st.pdf(path.read_bytes(), height=520)
        elif suffix in IMAGE_SUFFIXES:
            st.image(path.read_bytes(), caption=relative, width="stretch")
        else:
            st.caption(f"暂不支持预览该类型：{path}")
    except Exception as exc:  # 损坏文件或渲染失败时只提示，不影响归因查看
        st.warning(f"文件预览失败：{exc}")
    st.caption(str(path))


_CASE_DETAIL_CSS = """
<style>
.result-card {
    border-radius: 12px;
    padding: 14px 16px;
    border: 1px solid;
    margin-bottom: 12px;
}
.result-card.truth {
    background: rgba(34, 197, 94, 0.10);
    border-color: rgba(34, 197, 94, 0.45);
}
.result-card.pred-bad {
    background: rgba(239, 68, 68, 0.10);
    border-color: rgba(239, 68, 68, 0.55);
}
.result-card.pred-unlabeled {
    background: rgba(139, 148, 158, 0.08);
    border-color: #30363D;
}
.card-empty {
    color: #8B949E;
    font-size: 0.9rem;
    padding: 0.35rem 0;
}
.result-card.pred-ok {
    background: rgba(34, 197, 94, 0.10);
    border-color: rgba(34, 197, 94, 0.45);
}
.result-card .card-title {
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #a1a1a1;
    margin-bottom: 8px;
}
.kv-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.kv-table td {
    padding: 6px 4px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    vertical-align: top;
    word-break: break-all;
}
.kv-table tr:last-child td { border-bottom: none; }
.kv-table td.k { color: #a1a1a1; width: 34%; white-space: nowrap; }
.kv-table td.pred-wrong { color: #f87171; font-weight: 700; }
.kv-table td.truth-hit { color: #4ade80; font-weight: 700; }
.empty { color: #6b7280; font-style: italic; }
.mini-badge {
    margin-left: 6px;
    padding: 1px 6px;
    border-radius: 8px;
    background: rgba(34, 197, 94, 0.18);
    color: #4ade80;
    font-size: 0.7rem;
    font-weight: 500;
}
</style>
"""


def _render_value_card(case: dict, *, kind: str) -> None:
    """渲染标准答案卡片（绿色）或模型输出卡片（错误字段红色）。"""
    truth = case.get("ground_truth") or {}
    prediction = case.get("prediction") or {}
    strict_matches = case.get("strict_matches") or {}
    field_matches = case.get("field_matches") or {}
    unlabeled = case.get("exact_match") is None

    if kind == "truth" and case.get("ground_truth") is None:
        st.markdown(
            '<div class="result-card pred-unlabeled">'
            '<div class="card-title">正确答案</div>'
            '<div class="card-empty">该文件未提供标准答案</div>'
            "</div>",
            unsafe_allow_html=True,
        )
        return

    rows: list[str] = []
    for field in INVOICE_FIELDS:
        label = html.escape(FIELD_LABELS.get(field, field))
        matched = bool(field_matches.get(field, False))

        if kind == "truth":
            value = _escape(truth.get(field))
            css = ' class="truth-hit"' if not matched else ""
            rows.append(
                f'<tr><td class="k">{label}</td><td{css}>{value}</td></tr>'
            )
        else:
            value = _escape(prediction.get(field))
            css = "" if matched or unlabeled else ' class="pred-wrong"'
            judge_fixed = matched and not strict_matches.get(field, False)
            badge = (
                '<span class="mini-badge">裁判修正</span>' if judge_fixed else ""
            )
            rows.append(
                f'<tr><td class="k">{label}</td>'
                f"<td{css}>{value}{badge}</td></tr>"
            )

    if kind == "truth":
        card_class = "truth"
        title = "正确答案"
    else:
        if case.get("exact_match") is True:
            card_class = "pred-ok"
        elif case.get("exact_match") is None:
            card_class = "pred-unlabeled"
        else:
            card_class = "pred-bad"
        title = f"模型输出 · {case.get('model', '')}"

    st.markdown(
        f'<div class="result-card {card_class}">'
        f'<div class="card-title">{html.escape(title)}</div>'
        f'<table class="kv-table">{"".join(rows)}</table>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_case_detail(run: dict, case: dict) -> None:
    st.divider()
    st.markdown(_CASE_DETAIL_CSS, unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### 原始票据")
        _render_file_preview(run["data_dir"], case["file"])

        st.markdown("#### 模型输出结果")
        _render_value_card(case, kind="prediction")

    with right:
        st.markdown("#### 正确答案")
        _render_value_card(case, kind="truth")

        st.markdown("#### 裁判 Agent 评语")
        reasons = case.get("judge_reasons") or {}
        if reasons:
            for field, reason in reasons.items():
                st.markdown(f"**{FIELD_LABELS.get(field, field)}**")
                quoted = "\n".join(
                    f"> {line}" for line in str(reason).splitlines()
                )
                st.markdown(quoted)
        elif case.get("judge_error"):
            st.warning(f"裁判调用失败：{case['judge_error']}")
        elif case.get("judge_used"):
            st.caption("裁判未返回文字理由。")
        else:
            st.caption("未触发 Judge（字段为空、严格匹配或已关闭 Judge）。")

        if case.get("exact_match") is True:
            st.success("该样本整单正确，无失败归因")
        elif case.get("exact_match") is None:
            if case.get("failure_type") not in (None, "none"):
                st.markdown("#### 归因标签")
                failure_type = case.get("failure_type", "")
                description = FAILURE_LABELS.get(failure_type, "")
                st.error(f"{failure_type} · {description}")
                if case.get("failure_detail"):
                    st.code(case["failure_detail"], language=None)
            else:
                st.info(
                    "该文件未提供标准答案，仅展示抽取结果；"
                    "无法判断整单是否正确，请人工核验。"
                )
        else:
            st.markdown("#### 归因标签")
            failure_type = case.get("failure_type", "")
            description = FAILURE_LABELS.get(failure_type, "")
            st.error(f"{failure_type} · {description}")
            if case.get("failure_detail"):
                st.code(case["failure_detail"], language=None)

    with st.expander("轨迹信息与模型原始输出"):
        st.json(
            {
                "file": case["file"],
                "prediction_model": case["model"],
                "prompt_version": case["prompt_version"],
                "latency_ms": case["latency_ms"],
                "input_tokens": case["input_tokens"],
                "output_tokens": case["output_tokens"],
                "extraction_cost_cny": case["extraction_cost_cny"],
                "judge_used": case["judge_used"],
                "judge_input_tokens": case["judge_input_tokens"],
                "judge_output_tokens": case["judge_output_tokens"],
                "judge_latency_ms": case["judge_latency_ms"],
                "judge_error": case["judge_error"],
                "failure_type": case["failure_type"],
                "error": case["error"],
            }
        )
        st.code(case.get("raw_output") or "", language="json")


def render_error_analysis() -> None:
    history: list[dict] = st.session_state.get("eval_history", [])
    if not history:
        st.info("请先在「批量评测看板」完成一次跑批，再回来做 Bad Case 归因。")
        return

    labels = [
        f"{index + 1}. {run['started_at']} · {run['label']} · "
        f"EM {run['summary']['exact_match_rate']:.1%}"
        if run["summary"].get("num_labeled", run["num_cases"])
        else f"{index + 1}. {run['started_at']} · {run['label']} · EM —"
        for index, run in enumerate(history)
    ]
    selected_index = st.selectbox(
        "评测批次",
        options=list(range(len(history))),
        format_func=lambda index: labels[index],
        index=len(history) - 1,
        key="analysis_run_index",
    )
    run = history[selected_index]
    only_failures = st.toggle(
        "只看错误样本",
        value=True,
        key=f"analysis_only_failures_{selected_index}",
    )

    all_cases = run["cases"]
    if not all_cases:
        st.warning("该批次没有可评测样本（全部缺少标准答案）。")
        return
    visible_cases = [
        case
        for case in all_cases
        if not only_failures
        or case["exact_match"] is False
        or (
            case["exact_match"] is None
            and case.get("failure_type") not in (None, "none")
        )
    ]
    if not visible_cases:
        st.success("当前批次没有可展示的错误样本，已标注样本全部正确。")
        return

    table_rows = []
    for case in visible_cases:
        table_rows.append(
            {
                "文件": case["file"],
                "结果": (
                    "全对"
                    if case["exact_match"] is True
                    else "未标注"
                    if case["exact_match"] is None
                    else "错误"
                ),
                "失败字段": _failed_field_labels(case),
                "归因": case["failure_type"],
                "模型": case["model"],
                "耗时(s)": round(case["latency_ms"] / 1000, 2),
                "成本(¥)": round(_case_cost(case), 6),
            }
        )
    table = pd.DataFrame(table_rows)

    st.caption(
        f"共 {len(visible_cases)} 条 · 点击任意一行，下方展示原始票据、"
        "模型输出、正确答案与裁判评语"
    )
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        key=f"analysis_table_{selected_index}",
    )

    selected_rows = event.selection.rows if event is not None else []
    if not selected_rows:
        st.info("尚未选择样本：点击上表任意一行即可下钻。")
        return

    render_case_detail(run, visible_cases[selected_rows[0]])


with tab1:
    render_playground()

with tab2:
    render_dashboard()

with tab3:
    render_error_analysis()
