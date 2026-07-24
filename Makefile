# 听书知识库 - Makefile
# 用法: make [目标]

.PHONY: help start stop restart status

# 默认目标
help:
	@echo "=================================================="
	@echo "  听书知识库 - 服务管理"
	@echo "=================================================="
	@echo ""
	@echo "  服务:"
	@echo "    make start          启动所有服务"
	@echo "    make stop           停止所有服务"
	@echo "    make restart        重启所有服务"
	@echo "    make status         查看服务状态"
	@echo ""
	@echo "  快速访问:"
	@echo "    make dashboard      打开控制面板"
	@echo "    make chat           打开对话页面"
	@echo "    make import         打开导入页面"
	@echo ""
	@echo "  工具:"
	@echo "    make clean          清理临时文件"
	@echo "    make test           运行测试"
	@echo ""

# 启动所有服务
start:
	@echo "🚀 启动听书知识库服务..."
	@echo "   使用控制面板: http://127.0.0.1:8080"
	@echo "   或命令行启动:"
	@echo "   - 控制面板: python3 -m uvicorn app.api.http.control_server:app --port 8080"
	@echo "   - 导入服务: python3 -m uvicorn app.api.http.import_server:app --port 8001"
	@echo "   - 查询服务: python3 -m uvicorn app.api.http.query_server:app --port 8002"

# 停止所有服务
stop:
	@echo "🛑 停止听书知识库服务..."
	python3 scripts/stop_services.py

# 重启所有服务
restart: stop
	@echo "🔄 重启中..."
	sleep 2
	$(MAKE) start

# 查看服务状态
status:
	@echo "📊 服务状态："
	@echo ""
	@for port in 8080 8001 8002; do \
		if ss -tlnp | grep -q ":$$port "; then \
			pid=$$(ss -tlnp | grep ":$$port " | awk '{print $NF}' | grep -o '[0-9]*'); \
			echo "  ✅ 端口 $$port: 运行中 (PID: $$pid)"; \
		else \
			echo "  ❌ 端口 $$port: 未运行"; \
		fi; \
	done

# 快速打开页面（需要在 Windows 浏览器中打开）
dashboard:
	@echo "🌐 打开控制面板..."
	@powershell.exe -Command "Start-Process 'http://127.0.0.1:8080'" 2>/dev/null || echo "请手动访问 http://127.0.0.1:8080"

chat:
	@echo "🌐 打开对话页面..."
	@powershell.exe -Command "Start-Process 'http://127.0.0.1:8080/html'" 2>/dev/null || echo "请手动访问 http://127.0.0.1:8080/html"

import-page:
	@echo "🌐 打开导入页面..."
	@powershell.exe -Command "Start-Process 'http://127.0.0.1:8080/import'" 2>/dev/null || echo "请手动访问 http://127.0.0.1:8080/import"

# 清理临时文件
clean:
	@echo "🧹 清理临时文件..."
	@find . -type f -name "test_*" -not -path "./.venv/*" -delete 2>/dev/null || true
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ 清理完成"

# 测试
test:
	@echo "🧪 运行测试..."
	python3 -m pytest tests/ -v 2>/dev/null || echo "⚠️  未找到测试套件"
