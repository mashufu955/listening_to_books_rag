#!/bin/bash
# 听书知识库 - 服务停止脚本
cd "$(dirname "$0")/.."
python scripts/stop_services.py "$@"
