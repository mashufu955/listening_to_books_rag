"""
控制台服务：统一的管理入口，负责服务启停、状态监控、前端托管。
运行在 8080 端口，可通过 dashboard 前端一键管理导入服务和查询服务。
"""
import subprocess
import sys
import os
import signal
import time
from pathlib import Path
from mimetypes import guess_type
from typing import Dict, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载 .env 环境变量，确保子进程能继承
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

import os
debug_msg = f"[DEBUG] control_server.py loaded .env, BGE_M3_PATH={os.getenv('BGE_M3_PATH', 'NOT SET')}"
debug_log_path = PROJECT_ROOT / "logs" / "control_server_debug.log"
debug_log_path.parent.mkdir(parents=True, exist_ok=True)
with open(str(debug_log_path), 'w') as f:
    f.write(debug_msg + '\n')
print(debug_msg, flush=True)

_DOT_ENV_PATH = PROJECT_ROOT / ".env"

def _read_env(key: str, default: str) -> str:
    if _DOT_ENV_PATH.exists():
        for line in _DOT_ENV_PATH.read_text("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() == key:
                return v.strip().strip("\"'")
    return default

CTRL_PORT = int(_read_env("CTRL_PORT", "8080"))
IMPORT_PORT = int(_read_env("IMPORT_APP_PORT", "8001"))
QUERY_PORT = int(_read_env("QUERY_APP_PORT", "8002"))
APP_HOST = _read_env("APP_HOST", "0.0.0.0")

app = FastAPI(
    title="听书知识库 Dashboard",
    description="听书知识库系统 – 服务启停 / 健康监控 / 前端托管",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_processes: Dict[str, subprocess.Popen] = {}

SERVICES = {
    "import": {
        "name": "导入服务",
        "port": IMPORT_PORT,
        "module": "app.api.http.import_server:app",
        "host": APP_HOST,
    },
    "query": {
        "name": "查询服务",
        "port": QUERY_PORT,
        "module": "app.api.http.query_server:app",
        "host": APP_HOST,
    },
}


def _build_cmd(service: str) -> list:
    info = SERVICES[service]
    # 优先级：优先使用有完整依赖的虚拟环境
    # 在 WSL 中访问 Windows 项目时，Windows venv 通常已有完整依赖
    # 1. Windows venv (.venv/Scripts/python.exe) - 通常已有完整依赖
    venv_win = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_win.exists():
        python_exe = str(venv_win)
    else:
        # 2. Linux venv (.venv/bin/python)
        venv_linux = PROJECT_ROOT / ".venv" / "bin" / "python"
        if venv_linux.exists():
            python_exe = str(venv_linux)
        else:
            # 3. 回退到系统 Python
            python_exe = sys.executable
    return [
        python_exe, "-m", "uvicorn",
        info["module"],
        "--host", info["host"],
        "--port", str(info["port"]),
    ]


@app.get("/")
def dashboard():
    html_path = PROJECT_ROOT / "app" / "resources" / "html" / "dashboard.html"
    if not html_path.exists():
        return JSONResponse({"error": "dashboard.html not found"}, status_code=500)
    return FileResponse(
        path=html_path,
        media_type=guess_type(str(html_path))[0] or "text/html",
    )


@app.post("/api/services/start/{service_name}")
def start_service(service_name: str):
    if service_name not in SERVICES:
        return JSONResponse(
            {"success": False, "message": f"未知服务: {service_name}"},
            status_code=400,
        )
    info = SERVICES[service_name]
    port = info["port"]

    proc = _processes.get(service_name)
    if proc is not None and proc.poll() is None:
        return {"success": True, "message": f"{info['name']} 已在运行"}

    try:
        import urllib.request as _ur
        req = _ur.Request(f"http://127.0.0.1:{port}/health", method="GET")
        _ur.urlopen(req, timeout=2)
        _processes.pop(service_name, None)
        return {"success": True, "message": f"{info['name']} 已在运行"}
    except Exception:
        pass

    cmd = _build_cmd(service_name)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            if sys.platform == "win32" else 0,
        )
        import urllib.request as _ur
        started = False
        for attempt in range(15):
            time.sleep(2.0)
            retcode = proc.poll()
            if retcode is not None:
                stderr_output = b""
                try:
                    stderr_output = proc.stderr.read(4096) if proc.stderr else b""
                except Exception:
                    pass
                err_msg = stderr_output.decode("utf-8", errors="replace") if stderr_output else "未知错误"
                _processes.pop(service_name, None)
                return JSONResponse(
                    {"success": False, "message": f"{info['name']} 启动失败 (exit={retcode}): {err_msg[:500]}"},
                    status_code=500,
                )
            try:
                req = _ur.Request(f"http://127.0.0.1:{port}/health", method="GET")
                _ur.urlopen(req, timeout=2)
                started = True
                break
            except Exception:
                continue

        if not started:
            _processes[service_name] = proc
            return {"success": True, "message": f"{info['name']} 已启动（等待中）"}

        _processes[service_name] = proc
        return {"success": True, "message": f"{info['name']} 已启动"}
    except Exception as e:
        _processes.pop(service_name, None)
        return JSONResponse(
            {"success": False, "message": f"启动失败: {e}"},
            status_code=500,
        )


@app.post("/api/services/stop/{service_name}")
def stop_service(service_name: str):
    if service_name not in SERVICES:
        return JSONResponse(
            {"success": False, "message": f"未知服务: {service_name}"},
            status_code=400,
        )
    info = SERVICES[service_name]
    port = info["port"]

    proc = _processes.get(service_name)
    if proc is not None and proc.poll() is None:
        try:
            if sys.platform == "win32":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
            _processes.pop(service_name, None)
            return {"success": True, "message": f"{info['name']} 已停止"}
        except Exception as e:
            return JSONResponse(
                {"success": False, "message": f"停止失败: {e}"},
                status_code=500,
            )

    _processes.pop(service_name, None)
    try:
        import subprocess as _sp
        result = _sp.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5
        )
        pids = set()
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.strip().split()
                if parts:
                    pid = parts[-1]
                    if pid.isdigit():
                        pids.add(int(pid))
        if not pids:
            return {"success": True, "message": f"{info['name']} 未运行"}

        for pid in pids:
            _sp.run(["taskkill", "/PID", str(pid), "/F"],
                    capture_output=True, timeout=5)
        return {"success": True, "message": f"{info['name']} 已停止"}
    except Exception as e:
        return {"success": True, "message": f"{info['name']} 未运行"}


@app.get("/api/services/status")
def services_status():
    import urllib.request
    import json as _json

    def _health_check(port: int) -> bool:
        try:
            url = f"http://127.0.0.1:{port}/health"
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=2)
            return resp.status == 200
        except Exception:
            return False

    result = {}
    for name, info in SERVICES.items():
        proc = _processes.get(name)
        subproc_alive = proc is not None and proc.poll() is None
        online = _health_check(info["port"])
        is_running = online or subproc_alive

        result[name] = {
            "name": info["name"],
            "port": info["port"],
            "running": is_running,
            "online": online,
        }
    return result


if __name__ == "__main__":
    import uvicorn
    print(f"  Dashboard → http://127.0.0.1:{CTRL_PORT}")
    print(f"  Import     → http://127.0.0.1:{IMPORT_PORT}")
    print(f"  Query      → http://127.0.0.1:{QUERY_PORT}")
    uvicorn.run(app, host=APP_HOST, port=CTRL_PORT)
