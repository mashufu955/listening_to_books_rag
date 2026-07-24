"""
听书笔记服务：检索听书笔记、评论摘要等
"""
from app.shared.runtime.logger import step_log, logger
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.infra.llm.providers import llm_provider


@step_log("search_notes")
def search_notes(query: str, book_title: str = None, limit: int = 5) -> list[dict]:
    """
    搜索听书笔记/评论摘要/常见问答
    """
    embedding_result = llm_provider.embed_documents([query])
    dense_vector = embedding_result['dense'][0]
    sparse_vector = embedding_result['sparse'][0]

    expr = None
    if book_title:
        expr = f"book_title in ['{book_title}']"

    ann_reqs = milvus_gateway.create_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=expr,
        limit=limit * 2
    )
    results = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=ann_reqs,
        ranker_weights=(0.5, 0.5),
        limit=limit * 2,
        output_fields=[
            "chunk_id", "title", "content", "book_title", "author",
            "content_type", "file_title", "part"
        ]
    )

    notes = []
    for result_batch in results:
        for item in result_batch:
            entity = item.get("entity", {})
            content_type = entity.get("content_type", "")
            if content_type in ("听书笔记", "用户评论摘要", "常见问答", "推荐运营资料"):
                notes.append({
                    "book_title": entity.get("book_title"),
                    "author": entity.get("author"),
                    "content_type": content_type,
                    "title": entity.get("title"),
                    "content": entity.get("content"),
                    "source_file": entity.get("file_title"),
                })
    return notes[:limit]
