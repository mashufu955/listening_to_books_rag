import json
"""
听书元数据提取服务：从文档内容中提取书名、作者、类别、时长、内容类型等
"""
from app.shared.runtime.logger import step_log, logger
from app.infra.llm.providers import llm_provider
from app.shared.runtime.load_prompt import load_prompt
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser


BOOK_CONTENT_TYPES = [
    "有声书信息", "书籍简介", "作者介绍", "听书笔记",
    "推荐运营资料", "用户评论摘要", "常见问答"
]

BOOK_CATEGORIES = [
    "小说", "非小说", "儿童", "教育", "历史", "科幻",
    "悬疑", "言情", "经管", "心理", "哲学", "传记"
]


@step_log("classify_content_type")
def classify_content_type(text: str, file_title: str) -> str:
    """根据内容判断属于哪种听书内容类型"""
    prompt = load_prompt(
        "content_type_classify",
        text=text[:500],
        file_title=file_title,
        types=", ".join(BOOK_CONTENT_TYPES)
    )
    llm = llm_provider.chat()
    result = llm.invoke([HumanMessage(content=prompt)])
    content_type = result.content.strip()
    if content_type not in BOOK_CONTENT_TYPES:
        content_type = "书籍简介"
    return content_type


@step_log("extract_book_meta")
def extract_book_meta(text: str, file_title: str, content_type: str) -> dict:
    """提取听书元数据"""
    prompt = load_prompt(
        "book_meta_extract",
        text=text[:2000],
        file_title=file_title,
        content_type=content_type,
        categories=", ".join(BOOK_CATEGORIES)
    )
    llm = llm_provider.chat(json_mode=True)
    parser = JsonOutputParser()
    chain = llm | parser
    try:
        result = chain.invoke([HumanMessage(content=prompt)])
    except Exception:
        result = {}

    # 确保 highlights/faq 为字符串（LLM 可能返回 list/dict）
    def _ensure_str(val):
        if val is None:
            return ""
        if isinstance(val, str):
            return val
        if isinstance(val, (dict, list)):
            return json.dumps(val, ensure_ascii=False)
        return str(val)

    meta = {
        "book_title": result.get("book_title") or file_title,
        "author": result.get("author") or "未知",
        "category": result.get("category") or "小说",
        "content_type": content_type,
        "duration": result.get("duration") or "",
        "highlights": _ensure_str(result.get("highlights")),
        "faq": _ensure_str(result.get("faq")),
        "narrator": result.get("narrator") or "",
    }
    return meta


@step_log("apply_book_meta_to_chunks")
def apply_book_meta_to_chunks(chunks: list, meta: dict):
    """将元数据写入每个chunk"""
    for chunk in chunks:
        chunk["book_title"] = meta.get("book_title", "")
        chunk["author"] = meta.get("author", "")
        chunk["category"] = meta.get("category", "小说")
        chunk["content_type"] = meta.get("content_type", "书籍简介")
        chunk["duration"] = meta.get("duration", "")
        chunk["highlights"] = meta.get("highlights", "")
        chunk["faq"] = meta.get("faq", "")
        chunk["narrator"] = meta.get("narrator", "")
    return chunks


def enrich_import_state_with_meta(state: dict) -> dict:
    """根据文档内容提取听书元数据并更新state"""
    md_content = state.get("md_content", "") or ""
    file_title = state.get("file_title", "") or "未知书名"
    if not md_content:
        return state

    content_type = classify_content_type(md_content, file_title)
    meta = extract_book_meta(md_content, file_title, content_type)

    state["book_title"] = meta.get("book_title", file_title)
    state["author"] = meta.get("author", "未知")
    state["category"] = meta.get("category", "小说")
    state["content_type"] = content_type
    state["duration"] = meta.get("duration", "")
    state["highlights"] = meta.get("highlights", "")
    state["faq"] = meta.get("faq", "")
    state["narrator"] = meta.get("narrator", "")

    chunks = state.get("chunks", [])
    if chunks:
        apply_book_meta_to_chunks(chunks, meta)
        state["chunks"] = chunks

    return state
