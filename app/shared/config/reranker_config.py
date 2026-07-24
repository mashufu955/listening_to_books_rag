"""
Reranker 配置模块，负责读取重排模型相关环境变量。
"""
import os
from dataclasses import dataclass

from app.shared.config.common import env_bool, env_str


def _normalize_model_path(path: str) -> str:
    """
    标准化模型路径，自动处理 WSL/Windows 跨平台路径转换。
    在 Windows 环境中将 `/mnt/<drive>/...` 转换为 `<drive>:/...`。
    """
    if not path:
        return path
    # Windows 环境：将 /mnt/X/ 转换为 X:/
    if os.name == 'nt' and path.startswith('/mnt/'):
        # /mnt/d/xxx -> D:/xxx
        drive_letter = path[5].upper()
        rest = path[7:]  # 跳过 /mnt/d/
        return f"{drive_letter}:/{rest}"
    return path


@dataclass
class RerankerConfig:
    bge_reranker_large: str
    bge_reranker_device: str
    bge_reranker_fp16: bool

    def __post_init__(self):
        """配置加载后自动标准化路径"""
        self.bge_reranker_large = _normalize_model_path(self.bge_reranker_large)


reranker_config = RerankerConfig(
    bge_reranker_large=env_str("BGE_RERANKER_LARGE"),
    bge_reranker_device=env_str("BGE_RERANKER_DEVICE"),
    bge_reranker_fp16=env_bool("BGE_RERANKER_FP16"),
)
