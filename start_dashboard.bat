@echo off
chcp 65001 >nul
title 听书知识库管理面板
echo ====================================================
echo   听书知识库管理面板启动中...
echo ====================================================
echo.
echo   项目路径: %~dp0
echo.
echo   Dashboard ^> http://127.0.0.1:8080
echo.
echo   在管理面板中可一键启动/停止：
echo     - 导入服务 (port 8001)
echo     - 查询服务 (port 8002)
echo.
echo   按 Ctrl+C 停止管理面板
echo ====================================================
echo.

cd /d "%~dp0"
python -m uvicorn app.api.http.control_server:app --host 0.0.0.0 --port 8080

if %errorlevel% neq 0 (
    echo.
    echo 启动失败，请确认 Python 环境已激活。
    pause
)
