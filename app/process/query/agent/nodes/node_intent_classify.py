import sys

from app.shared.runtime.logger import node_log
from app.rag.query.intent_classify_service import classify_query_intent
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_intent_classify")
def node_intent_classify(state):
    """
    节点功能：识别用户问题意图（推荐/详情/检索/问答/笔记）
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    original_query = state.get("original_query", "")
    history = state.get("history", [])
    intent = classify_query_intent(original_query, history)
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    return {"query_type": intent}


if __name__ == '__main__':
    test_state = {
        "session_id": "test_intent_001",
        "original_query": "推荐几本适合通勤听的科幻小说",
        "history": [],
        "is_stream": False,
    }
    result = node_intent_classify(test_state)
    print(f"意图分类结果: {result.get('query_type')}")
