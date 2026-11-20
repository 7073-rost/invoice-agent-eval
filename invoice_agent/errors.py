"""领域异常。"""


class InvoiceAgentError(Exception):
    """票据抽取 Agent 的基类异常。"""


class UnsupportedFileError(InvoiceAgentError):
    """文件类型不在支持范围内。"""


class ConfigurationError(InvoiceAgentError):
    """模型或 API Key 配置缺失。"""


class ExtractionError(InvoiceAgentError):
    """抽取流程失败：文档解析、模型输出解析或结果校验。"""


class DocumentUnreadableError(ExtractionError):
    """文件不可读、无文本层或图片内容为空。"""


class ModelOutputError(ExtractionError):
    """模型输出不是合法 JSON 或不符合票据结构。"""


class EmptyExtractionError(ModelOutputError):
    """模型返回了合法结构，但所有字段都为空，通常意味着图像模糊或无法识别。"""


class LLMRequestError(InvoiceAgentError):
    """大模型请求失败。"""
