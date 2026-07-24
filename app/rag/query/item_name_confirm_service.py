from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from app.process.query.agent.state import QueryGraphState
from app.infra.persistence.history_repository import history_repository
from app.shared.runtime.logger import logger, step_log
from app.infra.llm.providers import llm_provider
from app.shared.runtime.load_prompt import load_prompt
from app.infra.vectorstore.milvus_gateway import milvus_gateway


@step_log("get_data_and_validates")
def get_data_and_validates(state: QueryGraphState) -> tuple[str,str]:
    original_query = state.get("original_query")
    session_id = state.get("session_id")

    if not original_query or not session_id:
        logger.error(f"业务核心参数original_query或者session_id为空，业务无法继续进行！")
        raise ValueError(f"业务核心参数original_query或者session_id为空，业务无法继续进行！")

    return original_query, session_id

@step_log("get_history_messages")
def get_history_messages(session_id:str, limit:int = 10) -> list[dict]:
    history_message_list = history_repository.list_recent(session_id=session_id,limit=limit)
    logger.info(f"查询聊天记录数量：{len(history_message_list)}")
    final_message_list = [item for item in history_message_list if item.get("item_names") and len(item.get('item_names')) > 0]
    logger.info(f"校验后历史记录数量：{len(final_message_list)}")
    return final_message_list

@step_log("build_history_context_text")
def build_history_context_text(history_message_list) -> str:
    history_text = ""
    for index, item in enumerate(history_message_list,start=1):
        history_text += (f"序号：{index},类型：{'提问' if item['role'] == 'user' else '回答'},"
                         f"内容：{item['rewritten_query'] if item['role'] == 'user' else item['text']},"
                         f"关联主体：{','.join(item['item_names'])}\n"
                         )
    logger.info(f"最终拼接历史记录上下文：{history_text}")
    return history_text

@step_log("call_llm_deal_data")
def call_llm_deal_data(history_text, original_query)-> dict:
    json_llm_client = llm_provider.chat(json_mode=True)
    prompt_text = load_prompt("rewritten_query_and_item_names",history_text=history_text,query=original_query)
    messages = [
        HumanMessage(
            content=prompt_text
        )
    ]

    chain = json_llm_client | JsonOutputParser()
    result_dict = chain.invoke(messages)

    if "item_names" not in result_dict:
        result_dict['item_names'] = []
    if "rewritten_query" not in result_dict:
        result_dict['rewritten_query'] = original_query
    return result_dict

@step_log("query_item_name_milvus")
def query_item_name_milvus(item_names: list[str]) -> dict[str, list[dict]]:
    milvus_result_dict = {}
    for item_name in item_names:
        embedding_result = llm_provider.embed_documents([item_name])
        dense_vector = embedding_result['dense'][0]
        sparse_vector = embedding_result['sparse'][0]
        ann_request_list = milvus_gateway.create_requests(dense_vector, sparse_vector, limit=10)
        milvus_search_result = milvus_gateway.hybrid_search(
            collection_name=milvus_gateway.item_collection_name,
            reqs=ann_request_list,
            ranker_weights=(0.4,0.6),
            norm_score=True,
            limit=5
        )
        if milvus_search_result is None or len(milvus_search_result) == 0:
            logger.warning(f"模型提供的：{item_name} 没有检索到对应数据库数据！跳过本次！")
            continue
        real_result = milvus_search_result[0]

        current_item_name_list = [{"item_name":item_dict.get('entity',{}).get('item_name'),
                                   "score":item_dict.get('distance',0)} for item_dict in real_result]
        milvus_result_dict[item_name] = current_item_name_list

    return milvus_result_dict


@step_log("confirm_item_name")
def confirm_item_name(state: QueryGraphState) -> QueryGraphState:
    original_query, session_id = get_data_and_validates(state)
    history_message_list = get_history_messages(session_id)
    history_text = build_history_context_text(history_message_list)
    result_dict = call_llm_deal_data(history_text, original_query)

    item_names = result_dict.get('item_names', [])
    rewritten_query = result_dict.get('rewritten_query', original_query)

    if not item_names:
        state['answer'] = f"请问您想查询哪本书籍？您可以告诉我书名或作者。"
        state['rewritten_query'] = rewritten_query
        return state

    milvus_result_dict = query_item_name_milvus(item_names)

    final_item_names = []
    for model_item_name in item_names:
        if model_item_name in milvus_result_dict:
            best_match = milvus_result_dict[model_item_name][0]
            final_item_names.append(best_match.get('item_name'))
        else:
            final_item_names.append(model_item_name)

    state['item_names'] = final_item_names
    state['rewritten_query'] = rewritten_query
    return state


if __name__ == '__main__':
    from app.shared.runtime.logger import logger
    mock_state = QueryGraphState({
        "session_id": "test_session_001",
        "original_query": "《三体》有哪些精彩书评？",
        "is_stream": False,
    })
    result_state = confirm_item_name(mock_state)
    print(result_state)
