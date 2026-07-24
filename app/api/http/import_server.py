"""
导入服务 HTTP 入口模块
"""
import shutil
import sys
import uuid
import threading
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.schema.import_schema import TaskStatusSchema, UploadSchema, BookMetaImport
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.process.import_.agent.main_graph import kb_import_app
from app.process.import_.agent.state import get_default_state, ImportGraphState, create_default_state
from app.infra.config.providers import settings
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, TASK_STATUS_PROCESSING,
    get_done_task_list, get_running_task_list, get_task_status,
    update_task_status, add_running_task, add_done_task
)

app = FastAPI(
    title=settings.app_name,
    description="听书知识库 - 文档导入服务",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
def health():
    return {"code": 200, "message": "Import Service is running"}


@app.get("/html")
def html():
    html_path_obj = PROJECT_ROOT / "app" / "resources" / "html" / "import.html"
    return FileResponse(
        path=html_path_obj,
        media_type="application/octet-stream"
    )


@app.get("/status/{task_id}")
def task_status(task_id:str):
    logger.info(f"获取任务状态接口被调用，task_id：{task_id}")
    return TaskStatusSchema(
        code=200,
        task_id=task_id,
        status=get_task_status(task_id),
        done_list=get_done_task_list(task_id),
        running_list=get_running_task_list(task_id)
    )


def invoke_graph(task_id:str, local_file_path:Path, local_dir:Path, book_meta: BookMetaImport = None):
    state = create_default_state(task_id=task_id, local_file_path=str(local_file_path), local_dir=str(local_dir))
    if book_meta:
        state["book_title"] = book_meta.book_title
        state["author"] = book_meta.author
        state["category"] = book_meta.category
        state["content_type"] = book_meta.content_type
        state["duration"] = book_meta.duration or ""
        state["source_file"] = book_meta.source_file
        state["source_path"] = book_meta.source_path or ""

    try:
        logger.info(f"{task_id}对应的文件解析任务开始执行！参数state:{state}")
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        final_state = kb_import_app.invoke(state)
        logger.info(f"{task_id}对应的文件解析任务完成！最终结果为：{final_state}")
        update_task_status(task_id, TASK_STATUS_COMPLETED)
    except Exception as e:
        update_task_status(task_id, TASK_STATUS_FAILED)
        logger.exception(f"===== 全流程测试运行失败 =====")


@app.post("/upload")
def upload_and_invoke_graph(files:list[UploadFile], book_meta: BookMetaImport = None):

    task_id = str(uuid.uuid4())
    add_running_task(task_id, "upload_file")
    local_dir_path_obj = PROJECT_ROOT / "output" / datetime.now().strftime("%Y%m%d") / task_id
    local_dir_path_obj.mkdir(parents=True, exist_ok=True)

    current_file = files[0]
    local_file_path_obj = local_dir_path_obj / current_file.filename

    with local_file_path_obj.open("wb") as file_buffer:
        shutil.copyfileobj(current_file.file, file_buffer)

    add_done_task(task_id, "upload_file")

    thread = threading.Thread(
        target=invoke_graph,
        args=(task_id, local_file_path_obj, local_dir_path_obj),
        kwargs={"book_meta": book_meta},
        daemon=True
    )
    thread.start()

    return UploadSchema(
        code=200,
        message="文件上传成功！",
        task_ids=[task_id]
    )


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.import_app_port)
