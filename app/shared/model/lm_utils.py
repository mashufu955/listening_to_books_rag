"""
工具模块，负责提供 lm 相关的辅助能力。
"""
from langchain_core.exceptions import LangChainException
from langchain_openai import ChatOpenAI
import httpx

from app.shared.config.lm_config import lm_config
from app.shared.runtime.logger import logger

_DEFAULT_LLM_MODEL = "qwen3-32b"
_DEFAULT_TEMPERATURE = 0.1
_llm_client_cache: dict[tuple[str, bool], ChatOpenAI] = {}


def get_llm_client(model: str | None = None, json_mode: bool = False) -> ChatOpenAI:
    """
    获取带全局缓存的LangChain ChatOpenAI客户端实例
    适配OpenAI/千问/即梦AI等**OpenAI兼容API**，支持自定义模型和JSON标准化输出
    核心特性：缓存机制+配置统一加载+异常精准捕获+代理绕过（强制直接连接）

    :param model: 模型名称，优先级：传入参数 > 配置文件 lm_config.llm_model > 内置默认模型
    :param json_mode: 是否开启JSON输出模式，开启后返回标准json_object格式（适配结构化数据解析）
    :return: 初始化完成的ChatOpenAI实例（优先从全局缓存获取，未命中则新建并缓存）
    :raise ValueError: 缺失API密钥/基础地址等核心配置
    :raise Exception: 模型初始化失败（LangChain封装层异常）
    """
    # 1. 确定目标模型（优先级递减，保证模型名非空）
    target_model = model or lm_config.llm_model or _DEFAULT_LLM_MODEL
    # 缓存键：模型名+JSON模式，唯一标识不同配置的客户端
    cache_key = (target_model, json_mode)

    # 2. 缓存命中：直接返回已初始化的实例，避免重复创建
    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM客户端] 缓存命中，直接返回实例：模型={target_model}，JSON模式={json_mode}")
        return _llm_client_cache[cache_key]

    # 3. 核心配置校验：拦截缺失的API关键配置，提前抛出明确异常
    if not lm_config.api_key:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_API_KEY（大模型API密钥）")
    if not lm_config.base_url:
        raise ValueError("[LLM客户端] 配置缺失：请在.env中配置OPENAI_BASE_URL（API接口基础地址）")
    logger.info(f"[LLM客户端] 开始初始化新实例：模型={target_model}，JSON模式={json_mode}")

    # 4. 配置参数组装：区分「国产模型私有参数」和「OpenAI通用参数」
    # extra_body：千问/即梦等国产模型专属私有参数（LangChain透传至API）
    # 注意：enable_thinking 仅 DashScope（阿里云百炼）支持，SiliconFlow 等其它提供商会报 400 或挂死
    extra_body = None
    if lm_config.base_url and "dashscope" in lm_config.base_url.lower():
        extra_body = {"enable_thinking": False}  # 千问专属：关闭思考链输出，减少冗余内容
    # model_kwargs：OpenAI通用参数，所有兼容API均支持
    model_kwargs = {}
    if json_mode:
        # 开启JSON标准输出模式，强制模型返回可解析的json_object
        model_kwargs["response_format"] = {"type": "json_object"}
        logger.debug(f"[LLM客户端] 已开启JSON输出模式，模型将返回标准JSON结构")

    # 5. 客户端初始化：捕获LangChain封装层异常，抛出更友好的提示
    try:
        # ═══════════════════════════════════════════════════════════════════════════
        # 代理修复核心逻辑
        # ═══════════════════════════════════════════════════════════════════════════
        # 问题根因：环境变量中存在 HTTP_PROXY/HTTPS_PROXY（Docker Desktop 代理 127.0.0.1:7895），
        # LangChain 的 socket_options 注入机制检测到代理环境变量后会绕过自定义 http_client，
        # 导致请求通过 127.0.0.1:7895 代理连接，但该代理对 dashscope 域名无效。
        #
        # 修复方案（三层保障）：
        #   (1) openai_proxy='' — 明确告知 LangChain 不使用任何代理
        #   (2) http_client=httpx.Client(trust_env=False) — 强制 httpx 忽略环境变量代理
        #   (3) http_socket_options=() — 禁用 LangChain 注入的 socket_options，
        #       避免其内部逻辑检测到代理环境变量后重新启用系统代理
        # ═══════════════════════════════════════════════════════════════════════════
        http_client = httpx.Client(trust_env=False)

        llm_client = ChatOpenAI(
            model=target_model,  # 目标模型名
            temperature=lm_config.llm_temperature or _DEFAULT_TEMPERATURE,  # 低温度保证输出确定性（0~1）
            api_key=lm_config.api_key,  # API密钥
            base_url=lm_config.base_url,  # API基础地址（适配国产模型代理地址）
            openai_proxy="",  # 关键修复：明确禁用代理（空字符串表示无代理）
            http_client=http_client,  # 强制使用无代理的 httpx 客户端
            http_socket_options=(),  # 关键修复：禁用 LangChain 的 socket_options 注入，防止其内部代理检测逻辑绕回系统代理
            request_timeout=120,  # API调用超时（秒），防止SiliconFlow等慢响应导致无限挂起
            max_retries=1,  # 失败重试次数，避免长时间重试阻塞流水线
            extra_body=extra_body,  # 国产模型私有参数透传
            model_kwargs=model_kwargs,  # OpenAI通用参数
        )
    except LangChainException as e:
        raise Exception(f"[LLM客户端] 模型【{target_model}】初始化失败（LangChain层）：{str(e)}") from e

    # 6. 新实例存入全局缓存，供后续调用复用
    _llm_client_cache[cache_key] = llm_client
    logger.info(f"[LLM客户端] 实例初始化成功并缓存：模型={target_model}，JSON模式={json_mode}")

    return llm_client
