import sys

from app.shared.runtime.logger import node_log
from app.rag.query.book_recommend_service import get_book_recommendations
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_recommend")
def node_recommend(state):
    """
    节点功能：根据用户问题生成书籍推荐列表
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    query = state.get("original_query") or state.get("rewritten_query", "")
    history = state.get("history", [])
    recommendations = get_book_recommendations(query, history)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"recommendations": recommendations}


if __name__ == '__main__':
    test_state = {
        "session_id": "test_rec_001",
        "original_query": "推荐几本科幻小说",
        "history": [],
        "is_stream": False,
    }
    result = node_recommend(test_state)
    print(f"推荐结果: {result.get('recommendations')}")
