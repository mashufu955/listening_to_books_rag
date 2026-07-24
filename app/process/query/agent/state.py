from typing_extensions import TypedDict
from typing import List, Optional
import copy

class QueryGraphState(TypedDict):
    session_id: str
    original_query: str
    query_type: Optional[str]  # auto/recommend/detail/search/qa/note

    # 检索过程中的中间数据
    embedding_chunks: list
    hyde_embedding_chunks: list
    web_search_docs: list

    # 排序过程中的数据
    rrf_chunks: list
    reranked_docs: list

    # 生成过程中的数据
    prompt: str
    answer: str

    # 辅助信息
    item_names: List[str]
    rewritten_query: str
    history: list
    is_stream: bool
    image_urls: List[str]

    # 听书结构化结果
    recommendations: list
    book_detail: dict
    notes: list


query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "query_type": "auto",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
    "image_urls": [],
    "recommendations": [],
    "book_detail": None,
    "notes": []
}


def create_query_default_state(**overrides) -> QueryGraphState:
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


def get_query_default_state() -> QueryGraphState:
    return copy.deepcopy(query_graph_default_state)


def copy_query_state(state: QueryGraphState, **overrides) -> QueryGraphState:
    new_state = copy.deepcopy(state)
    new_state.update(overrides)
    return new_state


if __name__ == '__main__':
    state = create_query_default_state(
        session_id="test_001",
        original_query="华为P60怎么样？",
        is_stream=False
    )
    print("初始化状态：", state)
