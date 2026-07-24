"""
听书书籍详情服务
"""
from app.shared.runtime.logger import step_log, logger
from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.infra.llm.providers import llm_provider
from app.shared.runtime.load_prompt import load_prompt
from langchain_core.messages import HumanMessage


@step_log("fetch_book_detail")
def fetch_book_detail(book_title: str) -> dict:
    """
    从知识库中聚合书籍详情
    """
    # 使用 book_title 进行向量检索
    embedding_result = llm_provider.embed_documents([book_title])
    dense_vector = embedding_result['dense'][0]
    sparse_vector = embedding_result['sparse'][0]

    ann_reqs = milvus_gateway.create_requests(
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=f"book_title in ['{book_title}']",
        limit=10
    )
    results = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=ann_reqs,
        ranker_weights=(0.5, 0.5),
        limit=10,
        output_fields=[
            "chunk_id", "title", "content", "book_title", "author",
            "category", "content_type", "duration", "highlights", "faq",
            "source_path", "narrator"
        ]
    )

    chunks = []
    for result_batch in results:
        for item in result_batch:
            entity = item.get("entity", {})
            if entity.get("book_title") == book_title or entity.get("file_title") == book_title:
                chunks.append(entity)

    if not chunks:
        return {
            "book_title": book_title,
            "author": "未知",
            "category": "小说",
            "content_types": [],
            "duration": "",
            "highlights": "",
            "faq": "",
            "source_file": "",
            "source_path": "",
            "narrator": "",
        }

    # 聚合元数据
    first = chunks[0]
    detail = {
        "book_title": first.get("book_title") or book_title,
        "author": first.get("author", "未知"),
        "category": first.get("category", "小说"),
        "content_types": list(set(c.get("content_type", "") for c in chunks if c.get("content_type"))),
        "duration": first.get("duration", ""),
        "highlights": first.get("highlights", ""),
        "faq": first.get("faq", ""),
        "source_file": first.get("file_title", ""),
        "source_path": first.get("source_path", ""),
        "narrator": first.get("narrator", ""),
    }
    return detail


@step_log("generate_book_detail_summary")
def generate_book_detail_summary(detail: dict) -> str:
    """基于详情生成结构化摘要"""
    prompt = load_prompt(
        "book_detail_summary",
        book_title=detail.get("book_title", ""),
        author=detail.get("author", ""),
        category=detail.get("category", ""),
        content_types=", ".join(detail.get("content_types", [])),
        duration=detail.get("duration", ""),
        narrator=detail.get("narrator", ""),
        highlights=detail.get("highlights", "")[:500],
        faq=detail.get("faq", "")[:500],
    )
    llm = llm_provider.chat()
    result = llm.invoke([HumanMessage(content=prompt)])
    return result.content.strip()
