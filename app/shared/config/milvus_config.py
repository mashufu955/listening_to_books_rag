"""
Milvus 向量数据库配置模块
"""
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MilvusConfig:
    milvus_url: str = os.getenv("MILVUS_URL", "http://127.0.0.1:19530")
    chunks_collection: str = os.getenv("CHUNKS_COLLECTION", "tingbook_chunks")
    item_name_collection: str = os.getenv("ITEM_NAME_COLLECTION", "tingbook_item_names")
    book_meta_collection: str = os.getenv("BOOK_META_COLLECTION", "tingbook_meta")


milvus_config = MilvusConfig()
