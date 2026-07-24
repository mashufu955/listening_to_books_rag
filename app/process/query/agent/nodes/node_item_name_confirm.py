import json
import sys
import time

from app.shared.runtime.logger import node_log
from app.rag.query.item_name_confirm_service import confirm_item_name
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.query.agent.state import QueryGraphState

@node_log("node_item_name_confirm")
def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品/书名名称。
    输入：state['original_query']
    输出：更新 state['item_names']
    """
    add_running_task(state["session_id"], "node_item_name_confirm", state["is_stream"])
    time.sleep(0.5)
    state = confirm_item_name(state)
    add_done_task(state["session_id"], "node_item_name_confirm", state["is_stream"])
    return state

if __name__ == '__main__':
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "《三体》有哪些精彩书评？",
        "is_stream": False,
    }
    result_state = node_item_name_confirm(mock_state)
    print(result_state)
