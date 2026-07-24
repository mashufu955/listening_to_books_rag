from mimetypes import guess_type
from pathlib import Path
import sys
import uuid

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schema.query_schema import (
    QueryRequestParam, QueryStreamResponse, QueryNotStreamResponse,
    HistoryCleanResponse, HistoryResponse, HistoryItemResponse,
    BookMeta, RecommendationItem
)
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.infra.config.providers import settings
from app.process.query.agent.main_graph import query_graph_app
from app.process.query.agent.state import create_query_default_state
from app.shared.utils.sse_utils import SSEEvent, create_sse_queue, push_to_session, sse_generator
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, TASK_STATUS_PROCESSING,
    clear_task, get_done_task_list, update_task_status
)
from app.infra.persistence.history_repository import history_repository

app = FastAPI(
    title=settings.app_name,
    description="听书知识库 - 智能问答与推荐服务",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)


@app.get("/html")
def chat_html():
    chat_html_path_obj = PROJECT_ROOT / "app" / "resources" / "html" / "chat.html"
    return FileResponse(
        path=chat_html_path_obj,
        media_type=guess_type(chat_html_path_obj.name)[0]
    )


@app.get("/health")
def health():
    return {"code": 200, "message": "可以访问！！"}


@app.get("/stream/{session_id}")
def stream(session_id, request: Request):
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream"
    )


def invoke_query_graph(session_id: str, query: str, is_stream: bool = False, query_type: str = None):
    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream,
        query_type=query_type or "auto",
    )
    clear_task(session_id)
    try:
        update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
        logger.info(f"开始执行，执行参数为：{state}")
        result_state = query_graph_app.invoke(state)
        logger.info(f"执行结束，执行结果为：{result_state}")
        update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)

        image_urls = result_state.get("image_urls", [])
        push_to_session(
            session_id,
            SSEEvent.FINAL,
            {
                "answer": result_state.get("answer", ""),
                "status": "completed",
                "image_urls": image_urls,
                "query_type": result_state.get("query_type"),
                "recommendations": result_state.get("recommendations", []),
                "book_detail": result_state.get("book_detail", {}),
                "notes": result_state.get("notes", []),
            }
        )
        return result_state
    except Exception as e:
        update_task_status(session_id, TASK_STATUS_FAILED, is_stream)
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
        logger.exception(f"{session_id}执行出现了异常!!")
        return {
            "answer": f"查询失败: {str(e)}",
            "error": str(e),
            "image_urls": [],
            "query_type": query_type,
            "recommendations": [],
            "book_detail": {},
            "notes": [],
        }


@app.post("/query")
def query(backgroundtasks: BackgroundTasks, request: QueryRequestParam):
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream
    query = request.query
    query_type = request.query_type or "auto"

    if is_stream:
        create_sse_queue(session_id)
        backgroundtasks.add_task(
            invoke_query_graph,
            session_id=session_id,
            query=query,
            is_stream=is_stream,
            query_type=query_type,
        )
        return QueryStreamResponse(
            message=f"开启：{session_id}异步任务执行!",
            session_id=session_id
        )
    else:
        final_state = invoke_query_graph(
            session_id=session_id,
            query=query,
            is_stream=is_stream,
            query_type=query_type,
        )
        return QueryNotStreamResponse(
            message=f"{session_id}对应的任务已经处理完毕！！",
            session_id=session_id,
            answer=final_state.get("answer", ""),
            done_list=get_done_task_list(session_id),
            image_urls=final_state.get("image_urls", []),
            query_type=final_state.get("query_type"),
            recommendations=final_state.get("recommendations", []),
            book_detail=final_state.get("book_detail"),  # 使用 None 而不是 {}
            notes=final_state.get("notes", []),
        )


# 听书专用接口：书籍推荐
@app.post("/recommend")
def recommend_books(request: QueryRequestParam):
    session_id = request.session_id or str(uuid.uuid4())
    query = request.query
    from app.rag.query.book_recommend_service import get_book_recommendations
    recommendations = get_book_recommendations(query)
    return {
        "code": 200,
        "session_id": session_id,
        "recommendations": recommendations,
    }


# 听书专用接口：书籍详情
@app.get("/book/{book_title}")
def get_book_detail(book_title: str):
    from app.rag.query.book_detail_service import fetch_book_detail
    detail = fetch_book_detail(book_title)
    return {"code": 200, "data": detail}


# 听书专用接口：听书笔记
@app.post("/notes")
def get_book_notes(request: QueryRequestParam):
    session_id = request.session_id or str(uuid.uuid4())
    query = request.query
    from app.rag.query.book_note_service import search_notes
    notes = search_notes(query)
    return {
        "code": 200,
        "session_id": session_id,
        "notes": notes,
    }


@app.delete("/history/{session_id}")
def remove_history(session_id: str):
    delete_count = history_repository.clear_session(session_id=session_id)
    logger.info(f"清空：{session_id}对应的历史记录！清空数量：{delete_count}")
    return HistoryCleanResponse(
        message=f"清空：{session_id}对应的历史记录！清空数量：{delete_count}",
        deleted_count=delete_count
    )


@app.get("/history/{session_id}")
def get_history(session_id: str, limit: int = 10):
    message_list = history_repository.list_recent(session_id=session_id, limit=limit)
    logger.info(f"完成：{session_id}对应的历史记录查询！查询的数据数量：{len(message_list)}")
    return HistoryResponse(
        session_id=session_id,
        items=[
            HistoryItemResponse(
                id=str(message.get("_id")),
                session_id=message.get("session_id"),
                role=message.get("role"),
                text=message.get("text"),
                rewritten_query=message.get("rewritten_query"),
                item_names=message.get("item_names"),
                image_urls=message.get("image_urls"),
                ts=message.get("ts")
            )
            for message in message_list
        ]
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)
