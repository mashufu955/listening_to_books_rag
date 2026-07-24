"""
听书意图分类服务：识别用户问题是"推荐/详情/检索/问答/笔记"中的哪种
"""
from app.shared.runtime.logger import step_log, logger
from app.infra.llm.providers import llm_provider
from app.shared.runtime.load_prompt import load_prompt
from langchain_core.messages import HumanMessage


INTENT_TYPES = [
    ("recommend", "书籍推荐"),
    ("detail", "书籍详情"),
    ("search", "内容检索"),
    ("qa", "知识问答"),
    ("note", "听书笔记"),
]


@step_log("classify_query_intent")
def classify_query_intent(query: str, history: list = None) -> str:
    """
    根据用户问题判断意图类型
    返回: recommend / detail / search / qa / note / auto
    """
    history_text = ""
    if history:
        for idx, item in enumerate(history[-5:], 1):
            history_text += f"{idx}. {item.get('role')}: {item.get('text')}\n"

    intent_descriptions = "\n".join([
        f"- {k}: {v}" for k, v in INTENT_TYPES
    ])

    prompt = load_prompt(
        "intent_classify",
        query=query,
        history=history_text,
        intents=intent_descriptions
    )
    llm = llm_provider.chat()
    result = llm.invoke([HumanMessage(content=prompt)])
    intent = result.content.strip().lower().replace(" ", "")

    valid_intents = [k for k, v in INTENT_TYPES]
    if intent not in valid_intents:
        intent = "auto"

    logger.info(f"意图分类结果: {intent} (query: {query[:50]})")
    return intent
