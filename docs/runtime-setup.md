# SmartRecipe DeepReason 本地运行配置

这份文档只针对新副本 `F:\agent+项目\SmartRecipe-DeepReason`。不要复制或复用旧项目的 `.env`。

## 1. 两种运行模式

- `DEEPREASON_DEMO_MODE=true`：不访问真实模型和数据库，用于先理解 Coordinator、并行 Agent、Reviewer、Evidence 与 Gate。
- `DEEPREASON_DEMO_MODE=false`：访问自己的 Qwen、MySQL、Neo4j、Redis Stack 和 Milvus，才算完整真实链路。

## 2. 申请自己的模型凭证

1. 登录阿里云百炼，在华北 2（北京）创建 API Key，建议放在默认业务空间。
2. 创建成功时立即保存 `API Key` 和 `API Host`，关闭弹窗后无法再次查看明文 Key。
3. 打开项目根目录 `.env`，只修改：

```dotenv
DASHSCOPE_API_KEY=你的真实Key
BAILIAN_WORKSPACE_ID=API Host 中的业务空间ID
```

不要把 Key 发到聊天、截图、简历或 Git。其余模型共用这一个业务空间 Key：

- 推理/结构化输出：`qwen-plus`
- 文本向量：`text-embedding-v4`，1024 维
- 文本重排：`qwen3-rerank`
- 图片理解：`qwen3-vl-plus`
- 文生图：`qwen-image-plus`（可选，最后再验）

先检查前三个模型接口：

```powershell
cd F:\agent+项目\SmartRecipe-DeepReason
.\.venv\Scripts\python.exe scripts\check_environment.py --models
```

## 3. 安装并启动 Docker 基础设施

本机目前没有 Docker Desktop。安装 Docker Desktop 后，确认 WSL 2/虚拟化可用并启动 Docker Engine，然后执行：

```powershell
cd F:\agent+项目\SmartRecipe-DeepReason
docker compose up -d
docker compose ps
.\.venv\Scripts\python.exe scripts\check_environment.py --services
```

项目使用以下宿主机端口：

| 服务 | 端口 | 用途 |
| --- | ---: | --- |
| MySQL | 13306 | 结构化菜谱与统计查询，供 Text2SQL 使用 |
| Neo4j Bolt | 17687 | 菜谱关系图谱，供 GraphRAG/Text2Cypher 使用 |
| Neo4j Browser | 17474 | 可视化查看图数据 |
| Redis Stack | 16379 | LangGraph Checkpoint 与会话状态 |
| RedisInsight | 18081 | Redis 可视化 |
| Milvus | 19530 | 菜谱文本向量与语义召回 |
| Attu | 18000 | Milvus 可视化 |

Redis 使用 16379 是因为本机 6379 已被旧版原生 Redis 占用，而且它没有 RedisJSON/RediSearch 模块。

## 4. 初始化数据

MySQL 首次创建 Volume 时会自动执行：

- `gustobot/data/init_mysql.sql`
- `gustobot/data/insert_sample_data.sql`

Neo4j 数据需要手动导入：

```powershell
docker compose exec neo4j cypher-shell -f /import/neo4j_import.cypher
docker compose exec neo4j cypher-shell "MATCH (n) RETURN count(n) AS nodes"
```

Milvus 需要由 Neo4j 菜谱数据生成向量索引：

```powershell
.\.venv\Scripts\python.exe -m gustobot.infrastructure.persistence.rebuild_milvus_index
```

## 5. 切到真实模式并启动

当下面命令全部通过后，将 `.env` 中 `DEEPREASON_DEMO_MODE` 改为 `false`：

```powershell
.\.venv\Scripts\python.exe scripts\check_environment.py --all
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

访问：

- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`
- DeepReason 状态：`http://127.0.0.1:8000/api/v1/deepreason/status`

## 6. 面试时怎么讲配置

“项目在 Windows 11、Python 3.12 环境运行，应用依赖由 uv/venv 管理。Qwen 使用阿里云百炼 OpenAI-compatible API，主模型 qwen-plus，Embedding 使用 text-embedding-v4 1024 维，Rerank 使用 qwen3-rerank。MySQL、Neo4j、Redis Stack、Milvus 通过 Docker Compose 管理。我写了独立环境自检脚本，分别执行数据库最小查询和模型最小请求，全部通过后才切换真实模式。”
