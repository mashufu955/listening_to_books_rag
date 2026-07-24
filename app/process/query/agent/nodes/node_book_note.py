import sys

from app.shared.runtime.logger import node_log
from app.rag.query.book_note_service import search_notes
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_book_note")
def node_book_note(state):
    """
    节点功能：搜索听书笔记、评论摘要、常见问答
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    query = state.get("original_query") or state.get("rewritten_query", "")
    item_names = state.get("item_names", [])
    book_title = item_names[0] if item_names else None
    notes = search_notes(query, book_title=book_title, limit=5)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"notes": notes}


if __name__ == '__main__':
    test_state = {
        "session_id": "test_note_001",
        "original_query": "《三体》有哪些精彩书评",
        "item_names": ["三体"],
        "history": [],
        "is_stream": False,
    }
    result = node_book_note(test_state)
    print(f"笔记结果: {result.get('notes')}")
