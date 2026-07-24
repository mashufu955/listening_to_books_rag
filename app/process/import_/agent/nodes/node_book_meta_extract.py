from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.book_meta_service import enrich_import_state_with_meta

@node_log("node_book_meta_extract")
def node_book_meta_extract(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 听书元数据提取 (node_book_meta_extract)
    从文档内容中提取书名、作者、类别、时长、内容类型等听书专用元数据。
    """
    add_running_task(state["task_id"], "node_book_meta_extract")
    state = enrich_import_state_with_meta(state)
    add_done_task(state["task_id"], "node_book_meta_extract")
    return state

if __name__ == '__main__':
    from app.shared.runtime.logger import logger
    from app.process.import_.agent.state import create_default_state
    import json

    test_state = create_default_state(
        task_id="test_meta_001",
        md_content="# 三体简介\n\n《三体》是刘慈欣创作的科幻小说...",
        file_title="三体简介"
    )
    result = node_book_meta_extract(test_state)
    print(json.dumps(result, ensure_ascii=False, indent=2))
