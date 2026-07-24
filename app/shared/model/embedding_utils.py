"""
工具模块，负责提供 embedding 相关的辅助能力。
"""
import os
import threading

# 强制离线模式：本地模型场景下跳过 HuggingFace Hub 验证和网络请求
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from typing import Any
import numpy as np

from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from FlagEmbedding import BGEM3FlagModel

from app.shared.config.embedding_config import embedding_config
from app.shared.runtime.logger import logger

# 用于临时禁用 HF repo_id 验证的线程锁
_hf_validation_lock = threading.Lock()


def _load_local_tokenizer(model_path: str):
    """
    从本地路径加载 tokenizer，临时禁用 repo_id 验证。

    BGEM3FlagModel 内部调用 AutoTokenizer.from_pretrained(model_path) 时
    会触发 validate_repo_id，拒绝非 HF repo_id 格式的本地路径。
    此处临时屏蔽验证以加载本地 tokenizer。
    """
    from huggingface_hub.utils._validators import validate_repo_id
    from transformers import AutoTokenizer

    # 转换 WSL 路径为 Windows 原生路径
    native_path = _resolve_native_path(model_path)

    with _hf_validation_lock:
        original = validate_repo_id
        validate_repo_id = lambda *a, **kw: None  # noqa: F841
        try:
            tokenizer = AutoTokenizer.from_pretrained(native_path)
        finally:
            validate_repo_id = original
    return tokenizer


_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
_DEFAULT_EMBEDDING_DEVICE = "cpu"
_bge_m3_ef: BGEM3EmbeddingFunction | None = None


# ---- 稀疏向量容器：兼容现有 CSR 矩阵解析逻辑 ----

class _SimpleCSR:
    """最小化 CSR 稀疏矩阵容器，用于兼容现有稀疏向量解析代码。"""

    def __init__(self, indices: list[int], indptr: list[int], data: list[float]):
        self._indices = np.array(indices, dtype=np.int64)
        self._indptr = np.array(indptr, dtype=np.int64)
        self._data = np.array(data, dtype=np.float32)

    @property
    def indices(self):
        return self._indices

    @property
    def indptr(self):
        return self._indptr

    @property
    def data(self):
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._indptr) - 1, int(self._indices.max()) + 1 if len(self._indices) > 0 else 0)


# ---- 本地模型专用 BGEM3EmbeddingFunction ----

class _LocalBGEM3(BGEM3EmbeddingFunction):
    """
    从本地路径加载 BGE-M3 模型的 BGEM3EmbeddingFunction 子类。

    BGEM3EmbeddingFunction 会将 model_name 原样传给 transformers，
    而 transformers 不接受纯本地文件路径作为 repo_id。
    此类绕过该限制，直接加载本地模型文件。
    """

    def __init__(self, local_path: str, **kwargs):
        # WSL 路径转换为 Windows 原生路径
        self._local_path = _resolve_native_path(local_path)
        self._device = kwargs.get("device", _DEFAULT_EMBEDDING_DEVICE)
        self._use_fp16 = kwargs.get("use_fp16", False)
        self._normalize = kwargs.get("normalize_embeddings", True)
        self._ef = None

        # ---- 核心属性（兼容 BGEM3EmbeddingFunction 接口）----
        self.model_name_or_path = local_path
        self.normalize_embeddings = self._normalize
        self.use_fp16 = self._use_fp16
        self.use_bf16 = kwargs.get("use_bf16", False)
        self.query_instruction_for_retrieval = kwargs.get("query_instruction_for_retrieval")
        self.query_instruction_format = kwargs.get("query_instruction_format", "{}{}")
        self.batch_size = kwargs.get("batch_size", 256)
        self.query_max_length = kwargs.get("query_max_length", 512)
        self.passage_max_length = kwargs.get("passage_max_length", 512)
        self.return_dense = kwargs.get("return_dense", True)
        self.return_sparse = kwargs.get("return_sparse", True)
        self.return_colbert_vecs = kwargs.get("return_colbert_vecs", False)
        self.truncate_dim = kwargs.get("truncate_dim", None)
        self.convert_to_numpy = kwargs.get("convert_to_numpy", True)
        self.pooling_method = kwargs.get("pooling_method", "cls")
        self.pool = None  # 多进程池（单设备模式下不使用）

        # ---- 直接加载组件（绕过父类中对本地路径的 HF repo_id 验证）----
        # 加载 tokenizer（使用已转换的 Windows 原生路径）
        self.tokenizer = _load_local_tokenizer(self._local_path)

        # 加载模型权重和线性层
        import torch
        from FlagEmbedding.inference.embedder.encoder_only.m3 import (
            EncoderOnlyEmbedderM3Runner,
            EncoderOnlyEmbedderM3ModelForInference,
        )

        runner_result = EncoderOnlyEmbedderM3Runner.get_model(
            self._local_path,
            trust_remote_code=kwargs.get("trust_remote_code", False),
            colbert_dim=kwargs.get("colbert_dim", -1),
            cache_dir=kwargs.get("cache_dir"),
            torch_dtype=torch.float16 if self._use_fp16 else torch.float32,
        )

        # 创建推理模型（get_model 返回 {model, colbert_linear, sparse_linear}）
        self.model = EncoderOnlyEmbedderM3ModelForInference(
            runner_result,
            tokenizer=self.tokenizer,
            sentence_pooling_method=self.pooling_method,
            normalize_embeddings=self._normalize,
        )

        # 记录 ColBERT 维度（如有）
        if hasattr(self.model, 'colbert_linear') and self.model.colbert_linear is not None:
            self.colbert_dim = self.model.colbert_linear.out_features

        # 初始化 pooling（延迟初始化，避免在无 GPU 时提前创建）
        self._pool = None

    @property
    def target_devices(self):
        if not hasattr(self, "_target_devices"):
            self._target_devices = self._resolve_target_devices()
        return self._target_devices

    def _resolve_target_devices(self):
        """根据 device 参数解析目标设备列表"""
        device = self._device
        if device == "cpu":
            return ["cpu"]
        if isinstance(device, str) and device.startswith("cuda"):
            return [device]
        return ["cpu"]

    def _encode_single(
        self,
        sentences: list[str],
        max_length: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert_vecs: bool,
    ) -> dict:
        """单设备编码核心逻辑，复用于文档和查询编码"""
        import torch
        from tqdm import tqdm
        from collections import defaultdict

        device = self.target_devices[0]
        if device == "cpu":
            self.model.float()
        self.model.to(device)
        self.model.eval()

        # 分词（无 padding）
        all_inputs = []
        for start_index in range(0, len(sentences), self.batch_size):
            batch = sentences[start_index:start_index + self.batch_size]
            inputs = self.tokenizer(
                batch, truncation=True, max_length=max_length
            )
            inputs_batch = [{k: inputs[k][i] for k in inputs} for i in range(len(batch))]
            all_inputs.extend(inputs_batch)

        # 按长度排序以减少 padding
        length_sorted_idx = sorted(range(len(all_inputs)), key=lambda i: -len(all_inputs[i]['input_ids']))
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]

        # 自动调整 batch size（OOM 时减小）
        batch_size = self.batch_size
        while True:
            try:
                test_batch = self.tokenizer.pad(
                    all_inputs_sorted[:batch_size],
                    padding=True, return_tensors='pt'
                ).to(device)
                self.model(test_batch, return_dense=True, return_sparse=False)
                break
            except RuntimeError:
                batch_size = max(1, batch_size * 3 // 4)

        # 批量编码
        all_dense, all_lexical, all_colbert = [], [], []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Encoding", disable=len(sentences) < batch_size):
            batch_inputs = all_inputs_sorted[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer.pad(
                batch_inputs, padding=True, return_tensors='pt'
            ).to(device)
            outputs = self.model(
                inputs_batch,
                return_dense=return_dense,
                return_sparse=return_sparse,
                return_colbert_vecs=return_colbert_vecs,
                truncate_dim=self.truncate_dim,
            )

            if return_dense:
                dense_np = outputs['dense_vecs'].detach().cpu().numpy()
                all_dense.append(dense_np)

            if return_sparse:
                token_weights = outputs['sparse_vecs'].squeeze(-1).detach().cpu().numpy()
                input_ids_np = inputs_batch['input_ids'].detach().cpu().numpy()
                for tw, ids in zip(token_weights, input_ids_np):
                    lexical = defaultdict(int)
                    unused = set()
                    for tok in ['cls_token', 'eos_token', 'pad_token', 'unk_token']:
                        if tok in self.tokenizer.special_tokens_map:
                            tid = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map[tok])
                            unused.add(tid)
                    for w, idx in zip(tw, ids):
                        if idx not in unused and w > 0:
                            idx_str = str(idx)
                            if w > lexical[idx_str]:
                                lexical[idx_str] = float(w)
                    all_lexical.append(lexical)

            if return_colbert_vecs:
                colbert_np = outputs['colbert_vecs'].detach().cpu().numpy()
                attention = inputs_batch['attention_mask'].detach().cpu().numpy()
                for cv, attn in zip(colbert_np, attention):
                    tokens_num = int(attn.sum())
                    all_colbert.append(cv[:tokens_num - 1])

        # 恢复原始顺序
        reverse_idx = {v: i for i, v in enumerate(length_sorted_idx)}
        if return_dense:
            all_dense = np.concatenate(all_dense, axis=0)
            all_dense = all_dense[sorted(range(len(reverse_idx)), key=lambda i: reverse_idx[i])]
        if return_sparse:
            all_lexical = [all_lexical[reverse_idx[i]] for i in range(len(all_lexical))]
        if return_colbert_vecs:
            all_colbert = [all_colbert[reverse_idx[i]] for i in range(len(all_colbert))]

        return {
            "dense_vecs": all_dense,
            "lexical_weights": all_lexical,
            "colbert_vecs": all_colbert,
        }

    def encode_documents(self, texts: list[str]) -> dict[str, Any]:
        """编码文档，返回 {dense: list[list[float]], sparse: _SimpleCSR}"""
        result = self._encode_single(
            sentences=texts,
            max_length=self.passage_max_length,
            return_dense=self.return_dense,
            return_sparse=self.return_sparse,
            return_colbert_vecs=self.return_colbert_vecs,
        )
        return self._format_output(result)

    def encode_queries(self, texts: list[str]) -> dict[str, Any]:
        """编码查询，返回格式与 encode_documents 一致"""
        result = self._encode_single(
            sentences=texts,
            max_length=self.query_max_length,
            return_dense=self.return_dense,
            return_sparse=self.return_sparse,
            return_colbert_vecs=self.return_colbert_vecs,
        )
        return self._format_output(result)

    def _format_output(self, result: dict) -> dict[str, Any]:
        """将模型输出转换为 BGEM3EmbeddingFunction 兼容格式"""
        dense_vecs = result.get("dense_vecs")
        if dense_vecs is None:
            dense_list = []
        else:
            arr = np.asarray(dense_vecs)
            dense_list = arr.tolist() if arr.ndim > 1 else [arr.tolist()]

        all_indices: list[int] = []
        all_indptr: list[int] = [0]
        all_data: list[float] = []
        for lw in (result.get("lexical_weights") or []):
            for idx, weight in sorted(lw.items()):
                all_indices.append(int(idx))
                all_data.append(float(weight))
            all_indptr.append(len(all_indices))

        return {
            "dense": dense_list,
            "sparse": _SimpleCSR(all_indices, all_indptr, all_data),
        }


def _resolve_model_name(bge_m3_path: str, bge_m3: str) -> str:
    """
    解析最终用于 BGEM3EmbeddingFunction 的模型名。

    优先级：
    1. bge_m3_path 是有效的 HuggingFace repo ID（如 "BAAI/bge-m3"）→ 直接使用
    2. bge_m3_path 是本地路径且路径存在 → 回退到 bge_m3，让 HuggingFace 从缓存加载
    3. 均不可用 → 使用内置默认值

    原因：pymilvus-model 会原样将 model_name 传给 transformers，
    而 transformers 不接受纯本地文件路径（需 HF repo ID 或 HF_HOME 缓存）。
    """
    if bge_m3_path:
        # 先排除本地路径（Linux/WSL 绝对路径、Windows 盘符路径、相对路径）
        is_local_path = (
            bge_m3_path.startswith("/")
            or bge_m3_path.startswith("./")
            or bge_m3_path.startswith("../")
            or len(bge_m3_path) > 1 and bge_m3_path[1] == ":"
        )
        if is_local_path:
            if bge_m3:
                return bge_m3
            # 没有 fallback，继续走默认值逻辑
        elif "/" in bge_m3_path:
            # 非本地路径但含 /，视为有效的 HuggingFace repo ID（如 "BAAI/bge-m3"）
            return bge_m3_path
    if bge_m3:
        return bge_m3
    return _DEFAULT_EMBEDDING_MODEL


def _is_local_path(path: str) -> bool:
    """判断是否为本地文件路径"""
    if not path:
        return False
    return (
        path.startswith("/")
        or path.startswith("./")
        or path.startswith("../")
        or (len(path) > 1 and path[1] == ":")
    )


def _resolve_native_path(path: str) -> str:
    """将 WSL 路径转换为 Windows 原生路径（如需）"""
    if path.startswith("/mnt/") and len(path) > 6:
        parts = path.split("/mnt/", 1)
        if len(parts) == 2 and len(parts[1]) > 1:
            drive = parts[1][0]
            rest = parts[1][2:]
            candidate = f"{drive.upper()}:/{rest}"
            if os.path.isdir(candidate):
                return candidate
    return path


def _local_path_exists(path: str) -> bool:
    """检查本地路径是否存在且包含模型文件"""
    if not _is_local_path(path):
        return False
    # 优先使用原始路径检查（WSL 下 /mnt/ 路径可直接访问）
    if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
        return True
    # Fallback: 如果路径被转换为 Windows 格式，也检查转换后的路径
    native_path = _resolve_native_path(path)
    if native_path != path:
        return os.path.isdir(native_path) and os.path.exists(os.path.join(native_path, "config.json"))
    return False


def get_bge_m3_ef() -> BGEM3EmbeddingFunction:
    """
    获取BGE-M3模型单例对象，自动加载环境变量配置
    :return: 初始化完成的BGEM3EmbeddingFunction实例
    """
    global _bge_m3_ef
    # 单例模式：已初始化则直接返回，避免重复加载模型
    if _bge_m3_ef is not None:
        logger.debug("BGE-M3模型单例已存在，直接返回实例")
        return _bge_m3_ef

    device = embedding_config.bge_device or _DEFAULT_EMBEDDING_DEVICE
    use_fp16 = embedding_config.bge_fp16
    bge_m3_path = embedding_config.bge_m3_path

    # 优先使用本地路径加载（ModelScope 等离线场景）
    if _local_path_exists(bge_m3_path):
        native_path = _resolve_native_path(bge_m3_path)
        logger.info(
            "使用本地BGE-M3模型加载（离线模式）",
            extra={"local_path": native_path, "device": device},
        )
        _bge_m3_ef = _LocalBGEM3(
            local_path=native_path,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True,
        )
        logger.success("BGE-M3本地模型加载成功，已开启原生L2归一化")
        return _bge_m3_ef

    # 远程模型：通过 HF repo ID 加载
    model_name = _resolve_model_name(bge_m3_path, embedding_config.bge_m3)

    # 打印模型初始化配置，便于问题排查
    logger.info(
        "开始初始化BGE-M3模型",
        extra={
            "model_name": model_name,
            "device": device,
            "use_fp16": use_fp16,
            "normalize_embeddings": True,
        },
    )

    try:
        # 初始化 BGE-M3 模型，开启原生 L2 归一化（适配 Milvus IP 内积检索）
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True,  # 模型原生对稠密+稀疏向量做L2归一化
        )
        logger.success("BGE-M3模型初始化成功，已开启原生L2归一化")
        return _bge_m3_ef
    except Exception as e:
        logger.error(f"BGE-M3模型初始化失败：{str(e)}", exc_info=True)
        raise  # 向上抛出异常，由调用方处理


def generate_embeddings(texts: list[str]) -> dict[str, list]:
    """
    为文本列表生成稠密+稀疏混合向量嵌入（模型原生L2归一化）
    :param texts: 要生成嵌入的文本列表，单文本也需封装为列表
    :return: 字典格式的向量结果，key为dense/sparse，对应嵌套列表/字典列表
    :raise: 向量生成过程中的异常，由调用方捕获处理
    """
    # 入参合法性校验
    if not isinstance(texts, list) or len(texts) == 0:
        logger.warning("生成向量入参不合法，texts必须为非空列表")
        raise ValueError("参数texts必须是包含文本的非空列表")
    if any(not isinstance(text, str) for text in texts):
        logger.warning("生成向量入参不合法，texts中存在非字符串内容")
        raise ValueError("参数texts必须是字符串列表")

    logger.info(f"开始为{len(texts)}条文本生成混合向量嵌入")
    try:
        # 加载BGE-M3模型单例
        model = get_bge_m3_ef()
        # 模型编码生成向量，返回dense（稠密向量）+sparse（CSR格式稀疏向量）
        embeddings = model.encode_documents(texts)
        logger.debug(f"模型编码完成，开始解析稀疏向量格式，共{len(texts)}条")

        # 初始化稀疏向量处理结果，解析为字典格式（适配序列化/存储）
        processed_sparse = []
        # # 把模型输出的 CSR 稀疏矩阵 ，按“每条文本一行”拆成 {特征索引: 权重} 字典
        # # - indices ：非零元素的“列号（特征ID）”
        # # - data ：对应列号的权重值
        # # - indptr ：每一行在 indices/data 里的起止位置指针
        # # 数据示例:
        # # indices = [3, 8, 20, 1, 9]
        # # data    = [0.7, 0.2, 0.1, 0.6, 0.4]  -> milvus -> 稠密向量 [1024] 稀疏向量 : {index:值,index:值}
        # # indptr  = [0, 3, 5]
        # # 获取对应的数据
        # # - 第0条文本用 0:3 => indices=[3,8,20] , data=[0.7,0.2,0.1]
        # # - 第1条文本用 3:5 => indices=[1,9] , data=[0.6,0.4]
        for i in range(len(texts)):
            # 提取第i个文本的稀疏向量索引：np.int64 → Python int（满足字典key可哈希要求）
            sparse_indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ]
            # 兼容 numpy array 和普通 list
            if hasattr(sparse_indices, "tolist"):
                sparse_indices = sparse_indices.tolist()
            else:
                sparse_indices = list(sparse_indices)
            # 提取第i个文本的稀疏向量权重：np.float32 → Python float（适配JSON序列化/接口返回）
            sparse_data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]:embeddings["sparse"].indptr[i + 1]
            ]
            # 兼容 numpy array 和普通 list
            if hasattr(sparse_data, "tolist"):
                sparse_data = sparse_data.tolist()
            else:
                sparse_data = list(sparse_data)
            # 构造{特征索引: 归一化权重}的稀疏向量字典
            sparse_dict = {k: v for k, v in zip(sparse_indices, sparse_data)}
            processed_sparse.append(sparse_dict)

        # 构造最终返回结果，稠密向量转列表（兼容 numpy 数组和纯 list）
        dense_output = []
        for emb in embeddings["dense"]:
            if hasattr(emb, "tolist"):
                dense_output.append(emb.tolist())
            else:
                dense_output.append(list(emb))
        result = {
            "dense": dense_output,
            "sparse": processed_sparse,
        }
        logger.success(f"{len(texts)}条文本向量生成完成，格式已适配工业级使用")
        return result

    except Exception as e:
        logger.error(f"文本向量生成失败：{str(e)}", exc_info=True)
        raise  # 不吞异常，向上传递让调用方做重试/降级处理


