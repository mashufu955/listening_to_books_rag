from typing import Any, List, Optional

from pydantic import BaseModel


class QueryRequestParam(BaseModel):
    query: str
    session_id: str
    is_stream: bool = False
    query_type: Optional[str] = None  # recommend / detail / search / qa / note / auto


class BookMeta(BaseModel):
    """听书书籍元数据"""
    book_title: str
    author: str
    category: str
    content_types: List[str]
    duration: Optional[str] = None
    source_file: str
    highlights: Optional[str] = None
    faq: Optional[str] = None


class RecommendationItem(BaseModel):
    """推荐书籍项"""
    book_title: str
    author: str
    category: str
    reason: str
    suitable_for: str
    features: str


class QueryStreamResponse(BaseModel):
    message: str
    session_id: str


class QueryNotStreamResponse(BaseModel):
    message: str
    session_id: str
    answer: str
    done_list: list
    image_urls: list
    # 听书结构化数据
    query_type: Optional[str] = None
    recommendations: Optional[List[RecommendationItem]] = None
    book_detail: Optional[BookMeta] = None
    notes: Optional[List[dict]] = None


class HistoryCleanResponse(BaseModel):
    message: str
    deleted_count: int


class HistoryItemResponse(BaseModel):
    id: str
    session_id: str
    role: str
    text: str
    rewritten_query: str
    item_names: list
    image_urls: list | None
    ts: Any


class HistoryResponse(BaseModel):
    session_id: str
    items: list[HistoryItemResponse]
