#!/bin/bash
# 听书知识库管理面板启动脚本

echo "===================================================="
echo "  听书知识库管理面板启动中..."
echo "===================================================="
echo ""
echo "  项目路径: $(pwd)"
echo ""
echo "  Dashboard -> http://127.0.0.1:8080"
echo ""
echo "  在管理面板中可一键启动/停止："
echo "    - 导入服务 (port 8001)"
echo "    - 查询服务 (port 8002)"
echo ""
echo "  按 Ctrl+C 停止管理面板"
echo "===================================================="
echo ""

cd "$(dirname "$0")"
python3 -m uvicorn app.api.http.control_server:app --host 0.0.0.0 --port 8080
