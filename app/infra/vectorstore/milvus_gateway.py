"""
工具模块，负责提供 milvus 相关的辅助能力。
"""
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.shared.config.milvus_config import milvus_config
from app.shared.runtime.logger import logger

# 全局Milvus客户端实例，实现单例复用
_milvus_client: MilvusClient | None = None


def get_milvus_client() -> MilvusClient | None:
    """
    Milvus客户端单例获取方法
    """
    try:
        global _milvus_client
        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            if not milvus_uri:
                logger.error("Milvus客户端连接失败：缺少MILVUS_URL环境变量配置")
                return None
            _milvus_client = MilvusClient(uri=milvus_uri)
            logger.info("Milvus客户端连接成功")
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus客户端连接异常：{str(e)}", exc_info=True)
        return None


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    from pymilvus import RRFRanker
    ranker = RRFRanker(k=limit)
    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=ranker,
        limit=limit,
        output_fields=output_fields or ["*"]
    )
    return results


class MilvusGateway:
    def __init__(self):
        from app.shared.config.milvus_config import milvus_config
        self.milvus_url = milvus_config.milvus_url
        self.chunk_collection_name = milvus_config.chunks_collection
        self.item_collection_name = milvus_config.item_name_collection
        self.book_meta_collection_name = getattr(milvus_config, 'book_meta_collection', 'tingbook_meta')

    @property
    def client(self):
        return get_milvus_client()

    def create_requests(
            self,
            dense_vector: list[float],
            sparse_vector: list[int, float],
            *,
            expr: str = None,
            limit: int = 5
    ):
        return create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=expr,
            limit=limit
        )

    def hybrid_search(
            self,
            *,
            collection_name: str,
            reqs: list[AnnSearchRequest],
            ranker_weights: tuple[float, float] = (0.5, 0.5),
            norm_score: bool = False,
            limit: int = 5,
            output_fields: list[str] | None = None,
            search_params: dict | None = None
    ):
        return hybrid_search(
            client=self.client,
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

    def ensure_book_meta_collection(self):
        """确保听书元数据集合存在"""
        client = self.client
        if client is None:
            return False
        if client.has_collection(collection_name=self.book_meta_collection_name):
            return True
        from pymilvus import DataType
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="book_title", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="content_type", datatype=DataType.VARCHAR, max_length=128)
        schema.add_field(field_name="duration", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="highlights", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="faq", datatype=DataType.VARCHAR, max_length=2048)
        schema.add_field(field_name="source_path", datatype=DataType.VARCHAR, max_length=1024)
        schema.add_field(field_name="narrator", datatype=DataType.VARCHAR, max_length=256)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE",
                               params={"M": 64, "efConstruction": 100})
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
                               params={"inverted_index_algo": "DAAT_MAXSCORE"})
        client.create_collection(
            collection_name=self.book_meta_collection_name,
            schema=schema, index_params=index_params
        )
        logger.info(f"{self.book_meta_collection_name} 集合初始化完成")
        return True

    def upsert_book_meta(self, meta: dict, dense_vector: list[float], sparse_vector: dict):
        """写入听书元数据"""
        client = self.client
        if client is None:
            return
        self.ensure_book_meta_collection()
        safe_book_title = meta.get("book_title", "").replace("'", "\\'")
        client.delete(
            collection_name=self.book_meta_collection_name,
            filter=f"book_title == '{safe_book_title}'"
        )
        data = {
            **meta,
            "dense_vector": dense_vector,
            "sparse_vector": sparse_vector,
        }
        client.insert(collection_name=self.book_meta_collection_name, data=[data])
        logger.info(f"听书元数据已写入: {meta.get('book_title')}")


milvus_gateway = MilvusGateway()
