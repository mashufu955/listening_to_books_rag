from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from pymilvus import DataType

from app.infra.llm.providers import llm_provider
from app.infra.vectorstore import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_.config import ITEM_NAME_CONTEXT_CHUNK_K, ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import step_log, logger


@step_log("validate_chunks_and_title")
def validate_chunks_and_title(state) -> tuple[list[dict],str]:
    chunks = state.get("chunks")
    file_title = state.get("file_title")
    if not chunks:
        logger.error("chunks内容为空，无法继续业务！")
        raise ValueError("chunks内容为空，无法继续业务！")
    if not file_title:
        file_title = chunks[0].get("file_title") or "default_file_title"
    return chunks, file_title


@step_log("build_document_context")
def build_document_context(chunks) -> str:
    top_chunk = chunks[:ITEM_NAME_CONTEXT_CHUNK_K]
    context = ""
    for index, chunk in enumerate(top_chunk, start=1):
        parent_title = chunk.get('parent_title', '')
        context += f"切片：{index} 标题：{chunk['title']} 父标题：{parent_title} 内容：{chunk['content']} \n"
    final_context = context[:ITEM_NAME_CONTEXT_TOTAL_MAX_CHARS]
    return final_context


@step_log("recognize_item_name")
def recognize_item_name(context:str, file_title:str) -> str:
    chat_model = llm_provider.chat()
    system_prompt_str = load_prompt("product_recognition_system")
    human_prompt_str = load_prompt(
        "item_name_recognition",
        file_title = file_title,
        context = context
    )
    messages = [
        SystemMessage(content = system_prompt_str),
        HumanMessage(content = human_prompt_str)
    ]
    chains = chat_model | StrOutputParser()
    item_name = chains.invoke(messages)
    logger.info(f"调用模型进行item_name识别完毕！ item_name:{item_name}")
    if not item_name:
        item_name = file_title
    return item_name


@step_log("apply_item_name")
def apply_item_name(chunks: list[dict], item_name: str):
    for chunk in chunks:
        chunk['item_name'] = item_name
    logger.info(f"完成chunks的item_name数据补充！{chunks[0]['item_name']}")


@step_log("embed_item_name")
def embed_item_name(item_name: str):
    result = llm_provider.embed_documents([item_name])
    return result['dense'][0], result['sparse'][0]


@step_log("prepare_item_name_collection")
def prepare_item_name_collection():
    milvus_client = milvus_gateway.client
    if milvus_client is None:
        logger.warning("Milvus客户端不可用，跳过主体名称集合创建")
        return False
    if milvus_client.has_collection(collection_name=milvus_gateway.item_collection_name):
        logger.info(f"{milvus_gateway.item_collection_name}对应的集合存在，无需创建！")
        return True
    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="book_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
    index_params = milvus_client.prepare_index_params()
    index_params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE",
                           params={"M": 64, "efConstruction": 100})
    index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                           params={"inverted_index_algo": "DAAT_MAXSCORE"})
    milvus_client.create_collection(
        collection_name=milvus_gateway.item_collection_name,
        schema=schema, index_params=index_params
    )
    logger.info(f"{milvus_gateway.item_collection_name}第一次完成初始化！")
    return True


@step_log("upsert_item_name")
def upsert_item_name(item_name: str, file_title: str, book_title: str, dense_vector: list[float], sparse_vector: dict[int, float]):
    milvus_client = milvus_gateway.client
    if milvus_client is None:
        logger.warning("Milvus客户端不可用，跳过主体名称写入")
        return
    
    # 调试日志：打印实际值
    logger.info(f"[DEBUG] upsert_item_name 参数: item_name={item_name!r}, file_title={file_title!r}, book_title={book_title!r}, dense_vector长度={len(dense_vector)}, sparse_vector类型={type(sparse_vector).__name__}")
    
    safe_title = file_title.replace("'", "\\'")
    milvus_client.delete(
        collection_name=milvus_gateway.item_collection_name,
        filter=f"file_title == '{safe_title}'"
    )
    
    # 确保所有字段值都是非空字符串（Milvus不接受None）
    safe_item_name = item_name or "未知商品"
    safe_file_title = file_title or "未知文件"
    safe_book_title = book_title or safe_file_title
    
    data = {
        "file_title": safe_file_title,
        "item_name": safe_item_name,
        "book_title": safe_book_title,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector,
    }
    
    logger.info(f"[DEBUG] 插入数据: file_title={safe_file_title!r}, item_name={safe_item_name!r}, book_title={safe_book_title!r}")
    
    result = milvus_client.insert(
        collection_name=milvus_gateway.item_collection_name,
        data=[data]
    )
    logger.info(f"主体名称写入完成！item_name:{item_name}")


@step_log("recognize_and_index_item_name")
def recognize_and_index_item_name(state: ImportGraphState) -> ImportGraphState:
    chunks, file_title = validate_chunks_and_title(state)
    context = build_document_context(chunks)
    item_name = recognize_item_name(context, file_title)
    book_title = state.get("book_title") or file_title
    apply_item_name(chunks, item_name)
    dense_vector, sparse_vector = embed_item_name(item_name)
    prepare_item_name_collection()
    upsert_item_name(item_name, file_title, book_title, dense_vector, sparse_vector)
    state["item_name"] = item_name
    return state


if __name__ == '__main__':
    from app.shared.runtime.logger import logger
    test_state = ImportGraphState({
        "task_id": "test_task_123456",
        "file_title": "三体简介",
        "book_title": "三体",
        "chunks": [
            {
                "parent_title": "三体",
                "title": "作品简介",
                "content": "《三体》是刘慈欣创作的科幻小说，讲述了地球文明与三体文明之间的故事。"
            }
        ]
    })
    result_state = recognize_and_index_item_name(test_state)
    logger.info(f"最终识别item_name：{result_state.get('item_name')}")
