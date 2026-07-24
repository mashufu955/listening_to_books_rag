import json
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import CHUNK_MAX_SIZE, CHUNK_SIZE, CHUNK_OVERLAP
from app.shared.runtime.logger import step_log, logger


@step_log("load_markdown_content")
def load_markdown_content(state: ImportGraphState) -> tuple[str, str, Path]:
    md_content = state.get("md_content")
    file_title = state.get("file_title")
    md_path = state.get("md_path")
    if not md_content:
        logger.warning("没有从state读取到md_content内容，我们使用md_path尝试再次读取！")
        if md_path:
            md_content = Path(md_path).read_text(encoding="utf-8")
            state["md_content"] = md_content
        if not md_content:
            raise ValueError("md_content没数据，并且尝试读取md_path依然没有数据，终止执行！！")
    if not file_title:
        file_title = Path(md_path).stem if md_path else "default"
        state["file_title"] = file_title
    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    return md_content, file_title, Path(md_path)


@step_log("split_by_titles")
def split_by_titles(md_content: str, file_title: str) -> list[dict]:
    reg = re.compile(r"^\s*#{1,6}\s.+")
    lines = md_content.split("\n")
    chunks: list[dict] = []
    current_title = None
    current_title_lines: list[str] = []
    is_code_block = False
    chunk_size = 0

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("```") or line.startswith("~~~"):
            is_code_block = not is_code_block
            current_title_lines.append(line)
            continue
        if reg.match(line) and not is_code_block:
            if current_title and len(current_title_lines) > 1:
                chunks.append({
                    "content": "\n".join(current_title_lines),
                    "title": current_title,
                    "file_title": file_title
                })
                chunk_size += 1
            current_title = line
            current_title_lines = [current_title]
        else:
            current_title_lines.append(line)

    if current_title and len(current_title_lines) > 1:
        chunks.append({
            "content": "\n".join(current_title_lines),
            "title": current_title,
            "file_title": file_title
        })
        chunk_size += 1

    if chunk_size == 0:
        chunks.append({
            "content": md_content,
            "title": "default",
            "file_title": file_title
        })
    logger.info(f"完成文档语义切割，共计切出：{chunk_size}块！")
    return chunks


@step_log("split_document")
def split_document(state: ImportGraphState) -> ImportGraphState:
    """
    听书文档切分：在标准切分基础上，保留听书元数据
    """
    md_content, file_title, md_path = load_markdown_content(state)
    chunks = split_by_titles(md_content, file_title)

    # 附加听书元数据
    book_title = state.get("book_title") or file_title
    author = state.get("author") or "未知"
    category = state.get("category") or "小说"
    content_type = state.get("content_type") or "书籍简介"
    duration = state.get("duration") or ""
    highlights = state.get("highlights") or ""
    faq = state.get("faq") or ""
    source_path = state.get("source_path") or str(md_path) if md_path else ""
    narrator = state.get("narrator") or ""

    for chunk in chunks:
        chunk["book_title"] = book_title
        chunk["author"] = author
        chunk["category"] = category
        chunk["content_type"] = content_type
        chunk["duration"] = duration
        chunk["highlights"] = highlights
        chunk["faq"] = faq
        chunk["source_path"] = source_path
        chunk["narrator"] = narrator
        # 修复: parent_title 是 Milvus schema 必填字段，补充默认值避免插入失败
        # 语义上 parent_title 表示章节的父标题/顶层标题，默认使用 file_title
        if "parent_title" not in chunk:
            chunk["parent_title"] = file_title

    state["chunks"] = chunks
    state["md_content"] = md_content
    return state


if __name__ == '__main__':
    from app.shared.runtime.logger import logger
    from app.process.import_.agent.state import create_default_state

    test_state = create_default_state(
        task_id="test_split_001",
        md_content="# 三体\n\n《三体》是刘慈欣创作的科幻小说...",
        file_title="三体简介"
    )
    result = split_document(test_state)
    print(f"生成chunks数量: {len(result.get('chunks', []))}")
