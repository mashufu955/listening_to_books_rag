# 服务管理工具

本目录包含听书知识库服务的启动、停止和管理脚本。

## 📋 可用脚本

### 1. 停止服务 (`stop_services.py`)

停止指定的服务进程，支持按端口或服务名选择性停止。

**用法：**

```bash
# 停止所有服务（默认）
python3 scripts/stop_services.py

# 仅停止导入服务
python3 scripts/stop_services.py --import

# 仅停止查询服务
python3 scripts/stop_services.py --query

# 仅停止控制面板
python3 scripts/stop_services.py --dashboard

# 显示帮助
python3 scripts/stop_services.py --help
```

**跨平台调用：**

```bash
# Linux / WSL
python3 scripts/stop_services.py

# Windows（命令行）
python scripts\stop_services.py

# 使用 Makefile（推荐）
make stop
```

**功能特性：**

- ✅ 自动检测运行环境（WSL / Windows）
- ✅ 通过端口号精准定位进程
- ✅ 优雅终止（SIGTERM）→ 等待 → 强制终止（SIGKILL）
- ✅ 彩色输出，状态清晰
- ✅ 支持选择性停止单个服务

### 2. Makefile

提供便捷的服务管理命令：

```bash
make help          # 显示帮助
make stop          # 停止所有服务
make start         # 显示启动说明
make status        # 查看服务状态
make dashboard     # 打开控制面板（Windows）
make clean         # 清理临时文件
```

### 3. 启动脚本

项目提供以下启动方式：

**方式一：控制面板（推荐）**

```bash
python3 -m uvicorn app.api.http.control_server:app --host 0.0.0.0 --port 8080
```

然后访问 http://127.0.0.1:8080 在面板中一键启动/停止服务。

**方式二：命令行启动**

```bash
# 终端 1：控制面板
python3 -m uvicorn app.api.http.control_server:app --host 0.0.0.0 --port 8080

# 终端 2：导入服务
python3 -m uvicorn app.api.http.import_server:app --host 0.0.0.0 --port 8001

# 终端 3：查询服务
python3 -m uvicorn app.api.http.query_server:app --host 0.0.0.0 --port 8002
```

**方式三：启动脚本**

```bash
# Linux / WSL
./start_dashboard.sh

# Windows
start_dashboard.bat
```

## 📊 服务端口

| 服务 | 端口 | 说明 |
|------|------|------|
| 控制面板 | 8080 | 服务管理、状态监控、前端页面 |
| 导入服务 | 8001 | 文档上传、解析、向量化 |
| 查询服务 | 8002 | 智能问答、推荐、检索 |

## 🔍 状态检查

**使用 Makefile：**

```bash
make status
```

**手动检查：**

```bash
# WSL / Linux
ss -tlnp | grep -E '8001|8002|8080'

# Windows PowerShell
Get-NetTCPConnection -LocalPort 8001,8002,8080
```

## 🧹 清理

```bash
# 清理临时测试文件
make clean

# 手动清理
find . -name "test_*" -delete
find . -name "__pycache__" -exec rm -rf {} +
```

## 📝 注意事项

1. **停止顺序**：建议先停止子服务（导入/查询），再停止控制面板
2. **强制终止**：脚本会先尝试优雅终止（5秒），未响应则强制终止
3. **权限问题**：如遇权限不足，请使用 `sudo`（WSL/Linux）
4. **Windows 兼容性**：Windows 批处理文件（`.bat`）需在 CMD 或 PowerShell 中运行

## 🐛 故障排除

### 端口被占用

```bash
# 查找占用端口的进程
sudo ss -tlnp | grep :8001

# 手动终止
kill -9 <PID>
```

### 服务无法停止

```bash
# 强制终止所有 Python 服务（谨慎使用）
pkill -9 -f "uvicorn"
```

### WSL 路径问题

确保在 WSL 中使用 Linux 路径，Windows 可执行文件需通过 `/mnt/c/` 访问。
