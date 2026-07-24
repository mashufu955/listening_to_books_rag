from dotenv import load_dotenv
from langgraph.constants import START
from langgraph.graph import StateGraph, END

from app.process.import_.agent.state import ImportGraphState
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_document_split import node_document_split
from app.process.import_.agent.nodes.node_item_name_recognition import node_item_name_recognition
from app.process.import_.agent.nodes.node_book_meta_extract import node_book_meta_extract
from app.process.import_.agent.nodes.node_bge_embedding import node_bge_embedding
from app.process.import_.agent.nodes.node_import_milvus import node_import_milvus
from app.shared.runtime.logger import logger

load_dotenv()

# 1. 定义状态图
main_graph = StateGraph(ImportGraphState)
# 2. 添加图节点
main_graph.add_node(node_entry)
main_graph.add_node(node_pdf_to_md)
main_graph.add_node(node_md_img)
main_graph.add_node(node_document_split)
main_graph.add_node(node_book_meta_extract)
main_graph.add_node(node_item_name_recognition)
main_graph.add_node(node_bge_embedding)
main_graph.add_node(node_import_milvus)

# 3. 添加起始节点 和 条件边
main_graph.set_entry_point("node_entry")

def node_entry_after(state: ImportGraphState) -> str:
    if state["is_md_read_enabled"]:
        logger.info(f"node_entry节点判断的文件{state['local_file_path']}类型 md,跳转到:node_md_img 节点")
        return "node_md_img"
    elif state["is_pdf_read_enabled"]:
        logger.info(f"node_entry节点判断的文件{state['local_file_path']}类型 pdf,跳转到:node_pdf_to_md 节点")
        return "node_pdf_to_md"
    else:
        logger.warning(f"node_entry节点获取的文件: {state['local_file_path']} 无法处理对应的类型,直接跳转到END节点!")
        return END

main_graph.add_conditional_edges("node_entry", node_entry_after,
                                 {
                                     "node_md_img":"node_md_img",
                                     "node_pdf_to_md":"node_pdf_to_md",
                                     END:END
                                  })

# 4. 添加静态边
main_graph.add_edge("node_pdf_to_md", "node_md_img")
main_graph.add_edge("node_md_img", "node_document_split")
# 听书元数据提取在切分之后、主体识别之前
main_graph.add_edge("node_document_split", "node_book_meta_extract")
main_graph.add_edge("node_book_meta_extract", "node_item_name_recognition")
main_graph.add_edge("node_item_name_recognition", "node_bge_embedding")
main_graph.add_edge("node_bge_embedding", "node_import_milvus")

# 5. 编译图对象
kb_import_app = main_graph.compile()

if __name__ == "__main__":
    from app.shared.utils.path_util import PROJECT_ROOT
    import os
    from app.shared.runtime.logger import logger

    logger.info("===== 开始执行听书知识库导入全流程测试 =====")
