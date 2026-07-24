"""
听书内容类型分类服务
"""
from app.shared.runtime.logger import step_log, logger

BOOK_CONTENT_TYPES = [
    "有声书信息", "书籍简介", "作者介绍", "听书笔记",
    "推荐运营资料", "用户评论摘要", "常见问答"
]


@step_log("classify_content_type_node")
def classify_content_type_node(text: str, file_title: str) -> str:
    """节点：对导入内容进行听书领域分类"""
    from app.infra.llm.providers import llm_provider
    from app.shared.runtime.load_prompt import load_prompt
    from langchain_core.messages import HumanMessage

    prompt = load_prompt(
        "content_type_classify",
        text=text[:800],
        file_title=file_title,
        types=", ".join(BOOK_CONTENT_TYPES)
    )
    llm = llm_provider.chat()
    result = llm.invoke([HumanMessage(content=prompt)])
    content_type = result.content.strip().replace("\n", "")
    if content_type not in BOOK_CONTENT_TYPES:
        content_type = "书籍简介"
    return content_type
