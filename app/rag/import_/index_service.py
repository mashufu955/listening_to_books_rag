import datetime
import json

from pymilvus import DataType

from app.infra.vectorstore.milvus_gateway import milvus_gateway
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import step_log, logger


@step_log("require_chunks")
def require_chunks(state: dict) -> list[dict]:
    chunks = state.get("chunks", [])
    if not chunks:
        logger.error("chunks为空，无法继续业务！")
        raise ValueError("chunks为空，无法继续业务！")
    return chunks


@step_log("prepare_chunks_collection")
def prepare_chunks_collection() -> None:
    milvus_client = milvus_gateway.client
    if milvus_client is None:
        raise RuntimeError(
            "Milvus 连接失败，无法准备集合。请检查：\n"
            "1. Docker 容器是否运行（docker ps | grep milvus）\n"
            "2. 端口 19530 是否已映射（docker port <milvus_container>）\n"
            "3. 环境变量 MILVUS_URL 是否正确（当前默认：http://localhost:19530）\n"
            "4. 防火墙是否允许本地 19530 端口入站"
        )
    collection_name = milvus_gateway.chunk_collection_name

    if milvus_client.has_collection(collection_name=collection_name):
        logger.info(f"{collection_name}对应的集合已经存在，无需创建，直接使用即可！")
        return

    schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="book_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="part", datatype=DataType.INT8)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=128)
    schema.add_field(field_name="duration", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="highlights", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="faq", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="source_path", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="narrator", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = milvus_client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        index_name="dense_vector_index",
        metric_type="COSINE",
        params={"M": 64, "efConstruction": 100}
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        index_name="sparse_vector_index",
        metric_type="IP",
        params={"inverted_index_algo": "DAAT_MAXSCORE"}
    )
    milvus_client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.info(f"{collection_name}完成对应的集合创建！")


@step_log("remove_old_chunks")
def remove_old_chunks(file_title: str) -> None:
    milvus_gateway.client.delete(
        collection_name=milvus_gateway.chunk_collection_name,
        filter=f"file_title=='{file_title}'"
    )


@step_log("insert_chunks")
def insert_chunks(chunks: list[dict]) -> None:
    # 防御性修复: 确保每个 chunk 都包含 Milvus schema 必填字段
    # 避免上游节点未设置必填字段导致插入失败
    for chunk in chunks:
        # 1. 字段类型安全: 确保 highlights/faq 为字符串
        # LLM 可能返回 dict/list，需序列化为 JSON 字符串
        for field in ("highlights", "faq"):
            val = chunk.get(field)
            if val is None:
                chunk[field] = ""
            elif isinstance(val, (dict, list)):
                chunk[field] = json.dumps(val, ensure_ascii=False)
            elif not isinstance(val, str):
                chunk[field] = str(val)

        # 2. 必填字段默认值（确保不会因缺失字段而插入失败）
        defaults = {
            "file_title": chunk.get("file_title", ""),
            "item_name": chunk.get("item_name") or chunk.get("file_title", ""),
            "book_title": chunk.get("book_title") or chunk.get("file_title", ""),
            "author": chunk.get("author", "未知"),
            "title": chunk.get("title", ""),
            "parent_title": chunk.get("parent_title") or chunk.get("file_title", ""),
            "part": chunk.get("part", 1),
            "content": chunk.get("content", ""),
            "category": chunk.get("category", "小说"),
            "content_type": chunk.get("content_type", "书籍简介"),
            "duration": chunk.get("duration", ""),
            "highlights": chunk.get("highlights", ""),
            "faq": chunk.get("faq", ""),
            "source_path": chunk.get("source_path", ""),
            "narrator": chunk.get("narrator", ""),
        }
        for field, default_val in defaults.items():
            if field not in chunk or chunk.get(field) is None:
                chunk[field] = default_val

    result = milvus_gateway.client.insert(
        collection_name=milvus_gateway.chunk_collection_name,
        data=chunks
    )
    logger.info(f"插入数据成功！总条数：{result.get('insert_count', 0)}")


@step_log("index_chunks")
def index_chunks(state: ImportGraphState) -> ImportGraphState:
    chunks = require_chunks(state)
    prepare_chunks_collection()
    remove_old_chunks(state['file_title'])
    insert_chunks(chunks)
    logger.info(f"{datetime.datetime.now().strftime('%Y%m%d')}完成{state['task_id']}导入文件数据入库操作！")
    return state
