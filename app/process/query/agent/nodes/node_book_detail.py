import sys

from app.shared.runtime.logger import node_log
from app.rag.query.book_detail_service import fetch_book_detail, generate_book_detail_summary
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_book_detail")
def node_book_detail(state):
    """
    节点功能：查询书籍详情
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    item_names = state.get("item_names", [])
    book_title = item_names[0] if item_names else state.get("original_query", "")
    detail = fetch_book_detail(book_title)
    summary = generate_book_detail_summary(detail)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"book_detail": detail, "answer": summary}


if __name__ == '__main__':
    test_state = {
        "session_id": "test_detail_001",
        "original_query": "《三体》的详细介绍",
        "item_names": ["三体"],
        "history": [],
        "is_stream": False,
    }
    result = node_book_detail(test_state)
    print(f"详情结果: {result.get('book_detail')}")
