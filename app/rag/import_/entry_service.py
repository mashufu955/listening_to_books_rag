from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger,step_log
from pathlib import Path

@step_log("resolve_input_file")
def resolve_input_file(state: ImportGraphState) -> ImportGraphState:
    local_file_path = state.get("local_file_path")
    if not local_file_path:
        logger.error("传入的local_file_path参数为空,没有文件,无法继续业务!")
        raise ValueError("传入的local_file_path参数为空,没有文件,无法继续业务!")
    if local_file_path.endswith(".md"):
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    elif local_file_path.endswith(".pdf"):
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
    else:
        logger.warning(f"传入的文件:{local_file_path}类型无法处理,当前项目只支持 md / pdf类型,直接跳转到END节点!")
        return state
    state["file_title"] = Path(local_file_path).stem
    state["source_file"] = Path(local_file_path).name
    state["source_path"] = str(local_file_path)
    return state
