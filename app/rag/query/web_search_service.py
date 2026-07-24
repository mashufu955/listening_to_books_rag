import json

from agents.mcp import MCPServerStreamableHttp

from app.process.query.agent.state import QueryGraphState
from app.shared.runtime.logger import logger, step_log
from app.infra.config.providers import infra_config
import asyncio


WEB_SEARCH_TIMEOUT = 60  # 联网搜索整体超时（秒），防止MCP连接挂起


@step_log("get_rewritten_query_and_validate")
def get_rewritten_query_and_validate(state) -> str:

    # 1. 获取数据
    rewritten_query = state.get("rewritten_query")
    # 2. 校验
    if not rewritten_query:
        logger.error("rewritten_query没有内容,业务无法继续进行!")
        raise ValueError("rewritten_query没有内容,业务无法继续进行!")
    # 3. 返回
    return rewritten_query


async def web_search_docs(rewritten_query: str):
    # -> CallToolResult | None
    """
    网络调用
    :param rewritten_query:
    :return: CallToolResult 或 None（失败时）
    """
    logger.info(f"[web_search_docs] 步骤开始, query={rewritten_query}")
    # 1. 初始化mcp_server
    mcp_server = MCPServerStreamableHttp(
        name="web_search_mcp",
        client_session_timeout_seconds=WEB_SEARCH_TIMEOUT,
        params={
            "url": infra_config.mcp.mcp_base_url,
            "headers": {"Authorization": f"Bearer {infra_config.mcp.api_key}"},
            "timeout": WEB_SEARCH_TIMEOUT
        },
        cache_tools_list=True,
        max_retry_attempts=1  # 减少重试次数，加速失败
    )
    try:
        # 2. 创建链接
        await mcp_server.connect()
        # 3. 调用网络工具
        tool_list = await mcp_server.list_tools()
        logger.info(f"本次链接服务对应的工具列表：{tool_list}")
        mcp_result = await mcp_server.call_tool(
            tool_name="bailian_web_search",
            arguments={"query": rewritten_query, "count": 5}
        )
        logger.info(f"[web_search_docs] 步骤完成")
        return mcp_result
    except asyncio.TimeoutError:
        logger.error(f"联网搜索超时({WEB_SEARCH_TIMEOUT}s), query={rewritten_query}")
        return None
    except Exception as e:
        logger.exception(f"调用工具出现问题,本次参数:{rewritten_query},错误原因:{str(e)}")
        return None
    finally:
        # 4. 断开链接
        await mcp_server.cleanup()


@step_log("search_by_web")
def search_by_web(state: QueryGraphState) -> list:
    """
    网络搜索服务：
    1. 通过 MCP 协议异步调用百炼联网搜索接口
    2. 将用户的查询转化为实时的、结构化的网络搜索结果
    3. 包含标题、链接和摘要
    4. 回写 web_search_docs
    """
    # 1. 获取和校验参数
    rewritten_query = get_rewritten_query_and_validate(state)
    # 2. 调用业务的网络搜索工具（带超时保护，防止挂起）
    try:
        mcp_result = asyncio.run(
            asyncio.wait_for(web_search_docs(rewritten_query), timeout=WEB_SEARCH_TIMEOUT + 10)
        )
    except asyncio.TimeoutError:
        logger.error(f"search_by_web 整体超时({WEB_SEARCH_TIMEOUT + 10}s), query={rewritten_query}")
        return []
    except Exception as e:
        logger.exception(f"search_by_web 异常, query={rewritten_query}")
        return []

    # 3. 检查返回结果
    if mcp_result is None:
        logger.warning(f"联网搜索无返回结果, query={rewritten_query}")
        return []

    try:
        search_text = mcp_result.content[0].text
        web_search_docs_list = json.loads(search_text).get("pages", [])
        logger.info(f"{rewritten_query}问题对应联网查询的结果数：{len(web_search_docs_list)}")
        return web_search_docs_list
    except (AttributeError, IndexError, json.JSONDecodeError, KeyError) as e:
        logger.error(f"联网搜索结果解析失败: {e}, raw={mcp_result}")
        return []