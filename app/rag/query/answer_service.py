import re
import json

from app.infra.llm.providers import llm_provider
from app.infra.persistence.history_repository import history_repository
from app.process.query.agent.state import QueryGraphState
from app.rag.query.item_name_confirm_service import build_history_context_text
from app.shared.runtime.load_prompt import load_prompt
from app.shared.utils.task_utils import add_done_task, add_running_task, push_to_session, set_task_result
from app.shared.utils.sse_utils import SSEEvent
from app.shared.runtime.logger import logger, step_log
import time
import sys


@step_log("try_return_existing_answer")
def try_return_existing_answer(state: dict) -> bool:
    answer = state.get("answer")
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id")

    if not answer:
        return False
    if is_stream:
        for ch in answer:
            push_to_session(session_id, SSEEvent.DELTA, {"delta": ch})
            time.sleep(0.1)

    set_task_result(session_id, "answer", answer)
    return True


@step_log("validate_generation_inputs")
def validate_generation_inputs(state: dict) -> tuple[list[dict], list[str], str, list[dict]]:
    history = state.get("history", [])
    reranked_docs = state.get("reranked_docs")
    item_names = state.get("item_names", [])
    rewritten_query = state.get("rewritten_query") or state.get("original_query")

    if not reranked_docs or not rewritten_query:
        raise ValueError("生成答案需要 reranked_docs 和 rewritten_query / original_query")
    return reranked_docs, item_names, rewritten_query, history


@step_log("build_answer_prompt")
def build_answer_prompt(
        reranked_docs: list[dict],
        rewritten_query: str,
        item_names: list[str],
        history: list[dict],
        query_type: str = "qa",
        recommendations: list = None,
        book_detail: dict = None,
        notes: list = None,
) -> str:
    context_chunk_list = []
    for number, chunk in enumerate(reranked_docs, start=1):
        context_chunk_list.append(
            f"第{number}块：标题：{chunk['title']} 匹配度得分：{chunk['score']} 来源：{'网络搜索' if chunk['type'] == 'web' else '向量查询'}\n内容：{chunk['text']}"
        )

    context_chunk_str = "\n\n".join(context_chunk_list)
    history_text = build_history_context_text(history)
    item_name_str = "本次关联主体：" + ",".join(item_names) if item_names else "没有关联主体"

    extra_context = ""
    if query_type == "recommend" and recommendations:
        rec_text = "\n".join([
            f"- 《{r.get('book_title')}》 by {r.get('author')}: {r.get('reason')}"
            for r in recommendations
        ])
        extra_context = f"\n【推荐书籍】\n{rec_text}"
    elif query_type == "detail" and book_detail:
        extra_context = f"\n【书籍详情】\n书名：{book_detail.get('book_title')}\n作者：{book_detail.get('author')}\n类别：{book_detail.get('category')}\n时长：{book_detail.get('duration')}\n亮点：{book_detail.get('highlights')}\n常见问题：{book_detail.get('faq')}"
    elif query_type == "note" and notes:
        note_text = "\n".join([
            f"- [{n.get('content_type')}] {n.get('title')}: {n.get('content')[:200]}"
            for n in notes
        ])
        extra_context = f"\n【听书笔记/评论】\n{note_text}"

    return load_prompt(
        "tingbook_answer_out",
        context=context_chunk_str,
        history=history_text,
        item_names=item_name_str,
        question=rewritten_query,
        extra_context=extra_context,
        query_type=query_type,
    )


@step_log("final_answer")
def final_answer(state: dict, prompt: str) -> str:
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id")
    lm_client = llm_provider.chat()
    final_result = ""

    if is_stream:
        for chunk in lm_client.stream(prompt):
            delta_content = chunk.content
            final_result += delta_content
            push_to_session(session_id, SSEEvent.DELTA, {"delta": delta_content})
    else:
        response = lm_client.invoke(prompt)
        final_result = response.content

    set_task_result(session_id, "answer", final_result)
    state["answer"] = final_result
    return final_result


@step_log("extract_image_urls")
def extract_image_urls(reranked_docs: list[dict]) -> list[str]:
    image_urls: list[str] = []
    reg = re.compile(r"\!\[.*?\]\((.*?)\)")

    for doc in reranked_docs:
        url = doc.get("url")
        if url:
            image_urls.append(url)
        text = doc.get("text", "")
        matches = reg.findall(text)
        image_urls.extend(matches)

    return list(dict.fromkeys(image_urls))


@step_log("generate_answer")
def generate_answer(state: QueryGraphState) -> QueryGraphState:
    """
    听书知识库答案生成入口：
    - 若已有结构化结果（recommendations / book_detail / notes），直接组装返回
    - 否则走通用RAG生成
    """
    query_type = state.get("query_type", "qa")
    recommendations = state.get("recommendations", [])
    book_detail = state.get("book_detail", {})
    notes = state.get("notes", [])

    # 如果已有结构化结果且不需要额外检索，直接生成答案
    if query_type == "recommend" and recommendations:
        rec_text = "\n".join([
            f"《{r.get('book_title')}》 by {r.get('author')}\n推荐理由：{r.get('reason')}\n适合人群：{r.get('suitable_for')}\n特色：{r.get('features')}\n"
            for r in recommendations
        ])
        state["answer"] = f"根据您的需求，为您推荐以下有声书：\n\n{rec_text}"
        set_task_result(state.get("session_id"), "answer", state["answer"])
        return state

    if query_type == "detail" and book_detail:
        detail_text = f"《{book_detail.get('book_title')}》\n"
        detail_text += f"作者：{book_detail.get('author')}\n"
        detail_text += f"类别：{book_detail.get('category')}\n"
        detail_text += f"时长：{book_detail.get('duration') or '未知'}\n"
        detail_text += f"内容类型：{', '.join(book_detail.get('content_types', []))}\n"
        if book_detail.get('highlights'):
            detail_text += f"听书亮点：{book_detail.get('highlights')}\n"
        if book_detail.get('faq'):
            detail_text += f"常见问题：{book_detail.get('faq')}\n"
        state["answer"] = detail_text
        set_task_result(state.get("session_id"), "answer", state["answer"])
        return state

    if query_type == "note" and notes:
        note_text = "为您找到以下听书笔记/评论：\n\n"
        for n in notes:
            note_text += f"- [{n.get('content_type')}] 《{n.get('book_title')}》 {n.get('title')}\n{n.get('content')[:300]}\n来源：{n.get('source_file')}\n\n"
        state["answer"] = note_text
        set_task_result(state.get("session_id"), "answer", state["answer"])
        return state

    # 通用RAG生成
    try:
        reranked_docs, item_names, rewritten_query, history = validate_generation_inputs(state)
        prompt = build_answer_prompt(
            reranked_docs, rewritten_query, item_names, history,
            query_type=query_type,
            recommendations=recommendations,
            book_detail=book_detail,
            notes=notes,
        )
        state["prompt"] = prompt
        answer = final_answer(state, prompt)
        state["answer"] = answer
        image_urls = extract_image_urls(reranked_docs)
        state["image_urls"] = image_urls
    except Exception as e:
        logger.error(f"生成答案失败: {e}", exc_info=True)
        state["answer"] = f"抱歉，生成答案时出现错误：{str(e)}"
        set_task_result(state.get("session_id"), "answer", state["answer"])

    return state
