from pydantic import BaseModel
from typing import Optional, List


class BookMetaImport(BaseModel):
    """听书元数据导入"""
    book_title: str
    author: str
    category: str
    content_type: str  # 有声书信息/书籍简介/作者介绍/听书笔记/推荐运营资料/用户评论摘要/常见问答
    duration: Optional[str] = None
    source_file: str
    source_path: Optional[str] = None


class UploadSchema(BaseModel):
    code: int = 200
    message: str
    task_ids: list[str]
    book_meta: Optional[BookMetaImport] = None


class TaskStatusSchema(BaseModel):
    code: int = 200
    task_id: str
    status: str
    done_list: list[str]
    running_list: list[str]
