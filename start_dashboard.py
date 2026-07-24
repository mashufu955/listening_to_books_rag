"""
一键启动 RAG 管理面板（控制台）。
启动后访问 http://127.0.0.1:8080 即可使用统一管理界面。

使用方法：
    python start_dashboard.py
"""
import subprocess
import sys
import os
from pathlib import Path

# Windows 控制台 UTF-8 编码修复
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(str(PROJECT_ROOT))

# 确保在项目根目录下运行
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 54)
print("  RAG 管理面板启动中...")
print("=" * 54)
print()
print(f"  项目路径: {PROJECT_ROOT}")
print()
print("  Dashboard → http://127.0.0.1:8080")
print()
print("  在管理面板中可一键启动/停止：")
print("    - 导入服务 (port 8001)")
print("    - 查询服务 (port 8002)")
print()
print("  按 Ctrl+C 停止管理面板")
print("=" * 54)
print()

cmd = [
    sys.executable, "-m", "uvicorn",
    "app.api.http.control_server:app",
    "--host", "0.0.0.0",
    "--port", "8080",
]

try:
    subprocess.run(cmd, cwd=str(PROJECT_ROOT))
except KeyboardInterrupt:
    print("\n管理面板已停止")
except Exception as e:
    print(f"\n启动失败: {e}")
    sys.exit(1)
