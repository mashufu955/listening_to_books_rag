#!/usr/bin/env python3
"""
听书样例数据批量导入脚本
用法：python scripts/import_sample_data.py
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests


SAMPLE_DATA_DIR = PROJECT_ROOT / "output" / "sample_data"
IMPORT_API = os.getenv("IMPORT_API", "http://127.0.0.1:8001/upload")


def import_file(file_path: Path):
    """导入单个文件"""
    print(f"导入中：{file_path.name}")
    with open(file_path, "rb") as f:
        files = {"files": (file_path.name, f, "text/markdown")}
        resp = requests.post(IMPORT_API, files=files)
    if resp.status_code == 200:
        data = resp.json()
        task_id = data.get("task_ids", [""])[0]
        print(f"  -> 上传成功，task_id: {task_id}")
        return task_id
    else:
        print(f"  -> 上传失败：{resp.status_code} {resp.text}")
        return None


def main():
    if not SAMPLE_DATA_DIR.exists():
        print(f"样例数据目录不存在：{SAMPLE_DATA_DIR}")
        return

    md_files = sorted(SAMPLE_DATA_DIR.glob("*.md"))
    if not md_files:
        print("未找到 Markdown 样例文件")
        return

    print(f"找到 {len(md_files)} 个样例文件，开始导入...")
    print("=" * 50)

    for file_path in md_files:
        import_file(file_path)
        print()

    print("=" * 50)
    print("导入请求已发送，请访问管理面板查看任务进度：")
    print("http://127.0.0.1:8080")


if __name__ == "__main__":
    main()
