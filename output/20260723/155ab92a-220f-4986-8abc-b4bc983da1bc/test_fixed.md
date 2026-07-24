# 听书知识库

基于 RAG（检索增强生成）技术构建的智能听书知识库系统，支持书籍推荐、详情查询、听书笔记检索和知识问答。

## 功能特性

- 文档导入：支持 PDF / Markdown 格式的听书资料导入
- 智能切分：基于 Markdown 标题的语义切分
- 主体识别：自动识别书籍主体名称
- 听书元数据提取：自动提取书名、作者、类别、时长、亮点等
- 混合检索：Dense + Sparse 向量检索 + RRF 融合
- 意图识别：自动判断用户问题是推荐/详情/笔记/问答
- 听书推荐：基于用户需求推荐有声书
- 详情查询：聚合书籍元数据生成结构化详情
- 笔记检索：检索听书笔记、评论摘要、常见问答
- 前端面板：可视化服务控制和对话界面

## 技术栈

- 后端：FastAPI + LangGraph + LangChain
- 向量数据库：Milvus (BGE-M3 dense + sparse)
- 对象存储：MinIO
- 数据库：MongoDB
- 前端：原生 HTML/CSS/JS（暖色调武侠风）

## 快速开始

### 环境要求

- Python 3.10+
- Node.js 18+ (可选，用于前端构建)
- Milvus / MinIO / MongoDB 服务

### 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 配置

复制 `.env.example` 为 `.env` 并修改配置：

```bash
cp .env.example .env
```

主要配置项：
- `MILVUS_URL`: Milvus 服务地址
- `MINIO_ENDPOINT`: MinIO 服务地址
- `MONGODB_URI`: MongoDB 连接地址
- `LLM_API_KEY`: 大模型 API 密钥

### 启动服务

**方式一：使用管理面板（推荐）**

```bash
# Windows
start_dashboard.bat

# Linux/Mac
python -m uvicorn app.api.http.control_server:app --host 0.0.0.0 --port 8080
```

访问 http://127.0.0.1:8080 在面板中一键启动导入服务和查询服务。

**方式二：命令行启动**

```bash
# 终端 1：启动导入服务
python -m uvicorn app.api.http.import_server:app --host 0.0.0.0 --port 8000

# 终端 2：启动查询服务
python -m uvicorn app.api.http.query_server:app --host 0.0.0.0 --port 8001
```

### 导入样例数据

项目包含 10 本听书样例数据（`output/sample_data/` 目录）：

- 三体简介.md
- 人类简史简介.md
- 小王子简介.md
- 撒哈拉的故事简介.md
- 明朝那些事儿简介.md
- 活着简介.md
- 蛤蟆先生去看心理医生简介.md
- 解忧杂货店简介.md
- 追风筝的人简介.md
- 非暴力沟通简介.md

可通过管理面板的"导入文档"页面上传，或调用导入 API：

```bash
curl -X POST "http://127.0.0.1:8000/upload" -F "files=@output/sample_data/三体简介.md"
```

### 访问查询界面

- 对话查询：http://127.0.0.1:8001/html
- 管理面板：http://127.0.0.1:8080

## 项目结构

```
listening_to_books_rag/
├── app/
│   ├── api/http/              # HTTP 服务入口
│   │   ├── import_server.py   # 导入服务
│   │   ├── query_server.py    # 查询服务
│   │   └── control_server.py  # 管理面板服务
│   ├── infra/                 # 基础设施层
│   │   ├── llm/               # LLM 提供商
│   │   ├── vectorstore/       # Milvus 操作
│   │   └── persistence/       # MongoDB 持久化
│   ├── process/               # 业务流程层
│   │   ├── import_/           # 导入流程
│   │   │   └── agent/         # LangGraph 图定义
│   │   └── query/             # 查询流程
│   │       └── agent/         # LangGraph 图定义
│   ├── rag/                   # RAG 领域服务
│   │   ├── import_/           # 导入服务
│   │   └── query/             # 查询服务
│   ├── shared/                # 共享模块
│   └── resources/             # 静态资源
│       └── html/              # 前端页面
├── output/                    # 输出目录
│   └── sample_data/           # 样例数据
├── .env.example               # 环境变量模板
├── requirements.txt           # Python 依赖
└── start_dashboard.bat        # Windows 启动脚本
```

## API 文档

### 导入服务 (port 8000)

- `POST /upload` - 上传文件并导入
- `GET /status/{task_id}` - 查询导入任务状态
- `GET /health` - 健康检查

### 查询服务 (port 8001)

- `POST /query` - 智能问答（支持流式/非流式）
- `POST /recommend` - 书籍推荐
- `GET /book/{book_title}` - 书籍详情
- `POST /notes` - 听书笔记检索
- `GET /history/{session_id}` - 查询对话历史
- `DELETE /history/{session_id}` - 清空对话历史
- `GET /html` - 对话页面
- `GET /health` - 健康检查

## 听书场景说明

系统支持以下听书场景：

1. **书籍推荐** - "推荐几本科幻小说"
2. **书籍详情** - "《三体》的详细介绍"
3. **听书笔记** - "《活着》有哪些精彩书评"
4. **知识问答** - "《三体》主要讲了什么"

系统会自动识别用户意图并路由到对应的处理流程。

## License

MIT
