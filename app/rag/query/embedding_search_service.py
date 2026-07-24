from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger,step_log
from app.infra.llm.providers import llm_provider
from app.infra.vectorstore.milvus_gateway import milvus_gateway


def get_data_and_validates(state:QueryGraphState) -> tuple[str,list[str]]:
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names", [])

    if not rewritten_query or len(item_names) == 0:
        logger.error(f"重写问题或者关联的主体为空，无法继续业务！")
        raise ValueError(f"重写问题或者关联的主体为空，无法继续业务！")

    return rewritten_query, item_names


def milvus_search_entity(rewritten_query, item_names):
    embedding_result = llm_provider.embed_documents([rewritten_query])
    dense_vector = embedding_result['dense'][0]
    sparse_vector = embedding_result['sparse'][0]
    ann_reqs = milvus_gateway.create_requests(dense_vector=dense_vector,
                                              sparse_vector=sparse_vector, expr=f"item_name in {item_names}", limit=5*2)
    milvus_result = milvus_gateway.hybrid_search(
        collection_name=milvus_gateway.chunk_collection_name,
        reqs=ann_reqs,
        ranker_weights=(0.6,0.4),
        limit=5,
        norm_score=True,
        output_fields=[
            "chunk_id", "title", "parent_title", "file_title",
            "item_name", "content", "part",
            "book_title", "author", "category", "content_type", "duration"
        ]
    )
    return milvus_result[0] if milvus_result and len(milvus_result) > 0 else []


def normalize_retrieved_chunk(milvus_response: list[dict]) -> list[dict]:
    final_list_dict = []
    for milvus_dict in milvus_response:
        entity = milvus_dict.get("entity",{})
        final_list_dict.append(
            {
                "chunk_id": milvus_dict.get("id") or entity.get("chunk_id"),
                "item_name": entity.get("item_name", ""),
                "title": entity.get("title"),
                "parent_title": entity.get("parent_title"),
                "part": entity.get("part"),
                "file_title": entity.get("file_title"),
                "content": entity.get("content", ""),
                "book_title": entity.get("book_title"),
                "author": entity.get("author"),
                "category": entity.get("category"),
                "content_type": entity.get("content_type"),
                "duration": entity.get("duration"),
                "score": milvus_dict.get("distance", 0.0),
                "type": "milvus",
                "url": None,
            }
        )
    return final_list_dict


def search_by_embedding(state: QueryGraphState):
    rewritten_query, item_names = get_data_and_validates(state)
    milvus_response = milvus_search_entity(rewritten_query,item_names)
    final_list_dict = normalize_retrieved_chunk(milvus_response)
    return final_list_dict
