# RAG Agent

一个只通过 CLI 使用的本地知识库问答工具。它支持 PDF、Markdown 和 TXT，先从你的
资料中检索证据，再让 DeepSeek 根据证据回答问题。

```text
资料 -> 抽取 -> 分块 -> 本地索引 -> Top-K 检索 -> DeepSeek 回答
```

项目地址：<https://github.com/Ribacha/RAG-Agent>

> **日常使用看这里**：[docs/使用指南.md](docs/使用指南.md) —— 按场景组织的操作手册
>（怎么提问、怎么加资料、哪些命令不花钱）。本 README 偏安装配置和完整参考。

## 从底层学习本项目

如果你想理解 RAG 是如何从文件一步步搭建出来的，先阅读
[`docs/rag-guide/00_RAG学习路线_目录.md`](docs/rag-guide/00_RAG学习路线_目录.md)，再按目录
依次阅读。教程覆盖资料导入、数据契约、分块、Embedding、向量索引、检索、证据上下文、
评测、增量更新、安全、Agent 和 LangGraph；不展开 Transformer/LLM 内部原理。

## 安装为命令行工具（推荐）

像 Codex/Claude Code 一样，`rag-agent` 安装后在任意目录可用。工具通过 `.rag-agent/`
标记定位"工作区"（数据目录 + `.env` 配置都放在工作区内），不依赖仓库检出。

```bash
git clone git@github.com:Ribacha/RAG-Agent.git
cd RAG-Agent
bash scripts/setup.sh          # 一键：建 venv + 安装 .[all] + 生成 .env + 自检
# 或者手动：
#   python3 -m venv .venv && source .venv/bin/activate
#   python -m pip install -e ".[all]"
```

安装后在任何目录初始化一个知识库工作区并开始使用：

```bash
rag-agent init ~/my-kb         # 创建 data/ 目录结构，交互式（或 --api-key）生成 .env
cd ~/my-kb
rag-agent doctor               # 检查配置、依赖和索引状态
cp 你的资料.pdf data/inbox/
rag-agent ingest data/inbox
rag-agent chat                 # 交互式问答；--retrieval-only 离线检索，--agent 工具调用模式
```

在仓库检出内开发时，`python -m rag_agent ...` 的用法完全保留；`RAG_AGENT_ROOT`
环境变量可显式指定工作区（脚本和 CI 场景）。优化路线与任务清单见
[docs/RAG优化工程文档.md](docs/RAG优化工程文档.md)。

## 5 分钟配置 DeepSeek

### 1. 安装

需要 Python 3.11 或更高版本，以及一个 DeepSeek API Key。推荐先执行上一节的
`bash scripts/setup.sh`；下面的手动步骤与其等价。

```bash
git clone git@github.com:Ribacha/RAG-Agent.git
cd RAG-Agent

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[llm]"
export PYTHONPATH=src
```

`.[llm]` 会安装 OpenAI SDK。项目使用它调用 DeepSeek 的 OpenAI 兼容接口；导入资料
和离线检索本身不需要聊天模型。

### 2. 配置 Key

复制配置模板：

```bash
cp .env.example .env
```

编辑 `.env`，至少填写这一项：

```dotenv
LLM_API_KEY=sk-你的DeepSeekKey
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

程序启动时会自动读取项目根目录的 `.env`。Shell 中已经设置的同名变量优先级更高，
命令行参数优先级最高。`.env` 已被 `.gitignore` 排除，绝不要把真实 Key 提交到 Git。

也可以不使用 `.env`，直接在当前终端配置：

```bash
export LLM_API_KEY='sk-你的DeepSeekKey'
export LLM_BASE_URL='https://api.deepseek.com'
export LLM_MODEL='deepseek-chat'
```

单次命令也可以覆盖配置：

```bash
python -m rag_agent ask "你的问题" \
  --llm-api-key 'sk-你的DeepSeekKey' \
  --llm-base-url 'https://api.deepseek.com' \
  --llm-model 'deepseek-chat'
```

### 3. 导入资料

把资料放进 `data/inbox/`，然后构建索引：

```bash
mkdir -p data/inbox
cp /你的资料目录/* data/inbox/
python -m rag_agent ingest data/inbox
```

默认使用离线 `hash` embedding，所以这一步不会调用 DeepSeek，也不会产生额外模型下载。
成功后会在 `data/index/` 生成索引；这些生成文件不会提交到 Git。

### 4. 先检查检索，再调用 DeepSeek

先用 `search` 确认问题确实能命中你的资料：

```bash
python -m rag_agent search "你的问题" --top-k 5
```

然后让 DeepSeek 根据检索到的证据回答：

```bash
python -m rag_agent ask "你的问题" --top-k 5
```

回答会附带 `[1]`、`[2]` 等引用，以及文件路径、页码或 Markdown 标题。机器读取可加
`--json`：

```bash
python -m rag_agent ask "你的问题" --top-k 5 --json
```

### 5. 不消耗 Key 的验证方式

`--dry-run` 只检索并展示将要发送给模型的证据，不调用 DeepSeek：

```bash
python -m rag_agent ask "你的问题" --dry-run
```

它适合检查 RAG 是否真的命中了资料；要验证 Key、网络和 DeepSeek 生成，则运行不带
`--dry-run` 的 `ask`。

## 命令选择

| 命令 | 用途 | 是否调用 DeepSeek |
| --- | --- | --- |
| `init` | 初始化工作区：创建 data/ 目录并生成 .env | 否 |
| `doctor` | 检查工作区、配置和可选依赖 | 否 |
| `ingest` | 抽取资料、分块并建立索引 | 否 |
| `ingest-url` | 爬取网站文字内容，清洗整理后导入索引 | 否 |
| `search` | 只检索 Top-K 证据 | 否 |
| `ask` | Python 先检索，再让模型回答 | 是（`--dry-run` 除外） |
| `chat` | 交互式问答会话，支持续问 | 是（`--retrieval-only` / `--dry-run` 除外） |
| `agent` | 让模型自主决定何时调用受控检索工具 | 是 |
| `evaluate` | 用 JSONL 标注集评测 Recall@K | 否 |
| `list-documents` | 查看最近一次导入清单 | 否 |
| `rebuild-index` | 用已有 chunks 重新建立索引 | 否 |
| `version` | 打印版本号 | 否 |

第一次使用建议按 `ingest -> search -> ask` 顺序执行。`agent` 是需要模型工具调用能力
的高级模式，确认普通 `ask` 正常后再使用：

```bash
python -m rag_agent agent "Transformer 的注意力如何计算" --json
```

## 抓取网站内容

除了本地文件，还可以直接把一个网站的文字内容导入知识库（需要 `pip install -e ".[web]"`
安装 BeautifulSoup）：

```bash
# 抓入口页 + 直接链接（默认同域、遵守 robots.txt、请求间隔 1 秒、最多 10 页）
rag-agent ingest-url https://docs.example.com/tutorial

# 爬得更深更多；重复执行时内容未变的页面自动复用旧向量
rag-agent ingest-url https://docs.example.com/tutorial --max-pages 50 --max-depth 2

# 之后照常检索和提问，引用会显示来源 URL
rag-agent ask "这份教程怎么配置超时？"
```

只做静态抓取（JavaScript 渲染的站点抓不到）；设计思路与函数讲解见
[docs/网页爬取功能文档.md](docs/网页爬取功能文档.md)。

## 配置参考

### 聊天模型（DeepSeek）

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` | 无 | 必填；也支持兼容别名 `CHAT_API_KEY` |
| `LLM_BASE_URL` | `https://api.deepseek.com` | DeepSeek OpenAI 兼容接口地址 |
| `LLM_MODEL` | `deepseek-chat` | 聊天模型名称 |

兼容别名 `CHAT_BASE_URL`、`CHAT_MODEL` 也会被读取。命令行的 `--llm-api-key`、
`--llm-base-url`、`--llm-model` 会覆盖环境变量。

### Embedding

聊天模型和 embedding 是两套配置，不能混为一谈：

- 默认 `hash`：完全离线，适合先跑通流程；中文语义能力有限。
- `chinese`：内置中文字符 n-gram 基线，仍然离线，不是训练好的神经语义模型。
- `openai`：只有在你的服务确实提供 `/embeddings` 接口时才使用，需要单独的
  `EMBEDDING_API_KEY`、`EMBEDDING_BASE_URL` 和 `EMBEDDING_MODEL`。

例如使用中文离线基线，导入和查询必须使用同一个 provider：

```bash
python -m rag_agent ingest data/inbox --embedding-provider chinese --embedding-dimension 512
python -m rag_agent search "三次握手如何建立连接" \
  --embedding-provider chinese --embedding-dimension 512
```

更换 provider 或维度后，必须重新构建索引：

```bash
python -m rag_agent rebuild-index \
  --embedding-provider chinese --embedding-dimension 512
```

## 常用选项

```bash
# 调整检索数量和最低相似度
python -m rag_agent ask "你的问题" --top-k 8 --min-score 0.08

# 只搜索指定文件类型或来源
python -m rag_agent search "你的问题" --file-type pdf
python -m rag_agent search "你的问题" --source "/绝对路径/资料.pdf"

# 资料变化后增量更新，复用未变化文件的 chunks 和向量
python -m rag_agent ingest data/inbox --incremental

# 扫描 PDF（需要系统安装 Tesseract 和中文语言包）
python -m rag_agent ingest data/inbox --ocr --ocr-language "chi_sim+eng"
```

扫描 PDF 的可选依赖：

```bash
brew install tesseract tesseract-lang
python -m pip install -r requirements-ocr.txt
```

## 对话历史和 Agent

`agent` 可以把已完成的问答轮次保存为 JSONL，并在下一次命令中继续追问：

```bash
python -m rag_agent agent "Transformer 的注意力如何计算" \
  --history data/state/conversation.jsonl
python -m rag_agent agent "它为什么需要 Key？" \
  --history data/state/conversation.jsonl
```

需要 LangGraph 时额外安装并加上 `--graph`：

```bash
python -m pip install -e ".[graph]"
python -m rag_agent agent "你的问题" --graph --json
```

## 离线评测和测试

评测不调用聊天模型。JSONL 每行至少包含 `query` 和一个相关的
`relevant_chunk_ids` 或 `relevant_source_paths`：

```json
{"name":"attention","query":"Query、Key 和 Value 有什么关系？","relevant_chunk_ids":["chunk-attention"]}
```

运行：

```bash
python -m rag_agent evaluate data/eval/retrieval.jsonl --top-k 5 --json
python -m pytest -q
```

当前仓库回归结果为 59 个测试通过。`Recall@K` 衡量检索命中率；
`citation_accuracy` 是检索层引用正确率代理，不等同于聊天模型事实正确率。

## 常见问题

**提示“缺少聊天模型 API Key”**

确认 `.env` 位于项目根目录、变量名是 `LLM_API_KEY`，并且已经安装
`python-dotenv`（`pip install -r requirements.txt` 会安装）。也可以先执行
`export LLM_API_KEY='你的Key'` 验证环境变量路径。

**提示“聊天模型需要 OpenAI SDK”**

在已激活的虚拟环境中执行：

```bash
python -m pip install -e ".[llm]"
```

**返回“知识库中没有找到足够相关的内容”**

先运行 `list-documents` 确认资料已导入，再用 `search "同一个问题"` 查看结果。
确认 `ask` 和 `ingest` 使用相同的 embedding provider；必要时谨慎降低 `--min-score`。

**返回 401 / 403**

检查 Key 是否有效、账户是否有余额或权限，`LLM_BASE_URL` 是否为
`https://api.deepseek.com`。不要把 `/chat/completions` 再拼到 Base URL 后面。

**网络连接失败**

先用 `ask --dry-run` 验证本地索引和 RAG 证据；如果 dry-run 正常而普通 `ask` 失败，
问题在 Key、代理或外网连接，不在本地检索链路。

## 数据、安全和项目边界

- `data/inbox/` 放原始资料；`data/index/`、`data/failed/` 和缓存默认不提交。
- `.env`、API Key 和对话历史不要提交到 Git。
- 文档内容只作为不可信证据，模型不会因此获得任意文件读取或命令执行权限。
- 本项目没有 Web UI，完整功能通过 CLI 使用。
- 真实外部模型生成取决于你的 Key、账户和网络；仓库自带测试只验证本地链路。

代码分层、阶段推进记录和设计取舍见 [docs/工程规划.md](docs/工程规划.md)；RAG 优化路线与
可执行任务清单见 [docs/RAG优化工程文档.md](docs/RAG优化工程文档.md)。
