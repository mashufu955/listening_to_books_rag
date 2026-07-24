# 定义主图全局的state
import json
from typing import TypedDict
import copy
from app.shared.runtime.logger import logger

class ImportGraphState(TypedDict):
    # 最终任务状态
    task_id : str
    # 文件状态判断
    is_md_read_enabled : bool
    is_pdf_read_enabled : bool
    # 地址路径内容
    local_file_path : str
    local_dir : str
    md_path : str
    pdf_path : str
    file_title : str
    # 听书元数据
    book_title : str
    author : str
    category : str
    content_type : str  # 有声书信息/书籍简介/作者介绍/听书笔记/推荐运营资料/用户评论摘要/常见问答
    duration : str
    source_file : str
    source_path : str
    # 文本和切块内容
    md_content : str
    item_name : str
    chunks : list
    embeddings_content : list

default_state:ImportGraphState = {
    "task_id": "",
    "is_md_read_enabled": False,
    "is_pdf_read_enabled": False,
    "local_file_path": "",
    "local_dir": "",
    "md_path": "",
    "pdf_path": "",
    "file_title": "",
    "book_title": "",
    "author": "",
    "category": "小说",
    "content_type": "书籍简介",
    "duration": "",
    "source_file": "",
    "source_path": "",
    "md_content": "",
    "item_name": "",
    "chunks": [],
    "embeddings_content": []
}

def create_default_state(**overriders) -> ImportGraphState:
    copy_state = copy.deepcopy(default_state)
    copy_state.update(overriders)
    return copy_state

def get_default_state() -> ImportGraphState:
    return copy.deepcopy(default_state)

if __name__ == '__main__':
    state = create_default_state(task_id="007")
    logger.info(f"测试复制方法： \n {json.dumps(state, ensure_ascii=False, indent=4)}")

    state1 = get_default_state()
    logger.info(f"测试复制方法： \n {json.dumps(state1, ensure_ascii=False, indent=4)}")
