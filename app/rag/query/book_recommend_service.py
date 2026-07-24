"""
听书书籍推荐服务
"""
from app.shared.runtime.logger import step_log, logger
from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.shared.runtime.load_prompt import load_prompt
from langchain_core.messages import HumanMessage


@step_log("search_recommendations")
def search_recommendations(query: str, limit: int = 5) -> list[dict]:
    """
    根据查询条件搜索推荐书籍
    使用向量检索 + 规则匹配
    """
    # 向量化查询
    embedding_result = llm_provider.embed_documents([query])
    dense_vector = embedding_result['dense'][0]
    sparse_vector = embedding_result['sparse'][0]

    # 混合检索
    ann_reqs = milvus_gateway.create_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        limit=limit * 3
    )
    results = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=ann_reqs,
        ranker_weights=(0.5, 0.5),
        limit=limit * 3,
        output_fields=[
            "chunk_id", "title", "file_title", "item_name",
            "content", "book_title", "author", "category",
            "content_type", "duration", "highlights"
        ]
    )

    # 聚合到书籍级别
    books = {}
    for result_batch in results:
        for item in result_batch:
            entity = item.get("entity", {})
            book_title = entity.get("book_title") or entity.get("file_title", "")
            if not book_title:
                continue
            if book_title not in books:
                books[book_title] = {
                    "book_title": book_title,
                    "author": entity.get("author", "未知"),
                    "category": entity.get("category", "小说"),
                    "duration": entity.get("duration", ""),
                    "highlights": entity.get("highlights", ""),
                    "score": item.get("distance", 0),
                    "count": 1,
                }
            else:
                books[book_title]["count"] += 1
                books[book_title]["score"] = max(books[book_title]["score"], item.get("distance", 0))

    # 排序并取top
    sorted_books = sorted(books.values(), key=lambda x: x["score"] * x["count"], reverse=True)[:limit]
    return sorted_books


@step_log("generate_recommendation_reason")
def generate_recommendation_reason(book: dict, query: str) -> dict:
    """为推荐书籍生成推荐理由和适合人群"""
    prompt = load_prompt(
        "book_recommend_reason",
        book_title=book.get("book_title", ""),
        author=book.get("author", ""),
        category=book.get("category", ""),
        highlights=book.get("highlights", "")[:500],
        query=query
    )
    llm = llm_provider.chat()
    result = llm.invoke([HumanMessage(content=prompt)])
    content = result.content.strip()

    # 简单解析 LLM 返回
    reason = content
    suitable_for = ""
    features = ""
    for line in content.split("\n"):
        line = line.strip()
        if line.startswith("适合人群：") or line.startswith("适合:"):
            suitable_for = line.split("：", 1)[-1].strip()
        elif line.startswith("特色：") or line.startswith("特点："):
            features = line.split("：", 1)[-1].strip()

    book["reason"] = reason
    book["suitable_for"] = suitable_for or "一般听众"
    book["features"] = features or book.get("highlights", "")
    return book


def get_book_recommendations(query: str, history: list = None) -> list[dict]:
    """
    获取书籍推荐列表（带推荐理由）
    """
    books = search_recommendations(query, limit=5)
    recommendations = []
    for book in books:
        book_with_reason = generate_recommendation_reason(book, query)
        recommendations.append({
            "book_title": book_with_reason.get("book_title"),
            "author": book_with_reason.get("author"),
            "category": book_with_reason.get("category"),
            "reason": book_with_reason.get("reason"),
            "suitable_for": book_with_reason.get("suitable_for"),
            "features": book_with_reason.get("features"),
        })
    return recommendations
