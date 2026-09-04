# RAG Agent（PDF / Markdown / TXT）

这是一个按阶段实现的、可追踪的个人知识库 RAG Agent。当前版本已经完成：

```text
PDF / Markdown / TXT
    -> 文本抽取（PDF 文本层，低质量页可 OCR）
    -> 统一 TextBlock
    -> 确定性分块
    -> 本地 hash embedding（离线基线）
    -> JSONL 向量索引
    -> Top-K 检索
    -> 可选的 OpenAI 兼容聊天模型回答
```

项目和旁边的 `weather-agent` 完全独立，虚拟环境也独立。

## 路径

```text
/Users/Zhuanz/Documents/AI Agent/rag-agent
```

把资料放进：

```text
/Users/Zhuanz/Documents/AI Agent/rag-agent/data/inbox/
```

`data/` 下的原始文件、索引和运行结果默认不会提交到 Git；API Key 也不应写进代码。

## 安装

```bash
cd "/Users/Zhuanz/Documents/AI Agent/rag-agent"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

`requirements.txt` 直接列出 PDF 解析和测试依赖，适合从源码 checkout 安装，不会先构建
本地项目。要使用 DeepSeek/OpenAI 兼容聊天接口，再安装：

```bash
python -m pip install -e ".[llm]"
```

要使用可选 LangGraph 状态图，再安装：

```bash
python -m pip install -e ".[graph]"
```

当前 Python 3.13 环境会忽略带隐藏标志的 `.pth` 文件，因此不依赖 editable 安装。
上面的 `export PYTHONPATH=src` 让后续命令始终加载当前源码；源码变更后不需要重装。
新开终端时重新激活虚拟环境并设置该变量，也可以按单条命令显式运行：

```bash
PYTHONPATH=src .venv/bin/python -m rag_agent --help
```

### 扫描 PDF 的 OCR（可选）

文本型 PDF 不需要 OCR。扫描 PDF 需要 macOS 系统安装 Tesseract 和中文语言包：

```bash
brew install tesseract tesseract-lang
tesseract --list-langs       # 应能看到 chi_sim 和 eng
python -m pip install -r requirements-ocr.txt
```

`requirements-ocr.txt` 会复用基础依赖并追加 Pillow、pytesseract；系统仍需安装
Tesseract 可执行文件和对应语言包。

没有 Tesseract 时，`--ocr` 不会静默产生空文本，而会把 `ocr_unavailable` 写入块和文档清单的 warning。

## 第一次导入

在项目目录执行：

```bash
source ".venv/bin/activate"
python -m rag_agent ingest data/inbox
```

默认使用完全离线的 `hash` embedding，因此这一步不需要模型 Key。会生成：

```text
data/index/chunks.jsonl       # 分块正文和来源元数据
data/index/documents.jsonl    # 每个文件的处理摘要
data/index/vectors.jsonl      # embedding + chunk 的本地索引
data/failed/ingestion.jsonl   # 失败文件和可读原因
```

成功输出的 `documents_failed` 为 0 时，退出码是 0；有文件失败但其他文件成功时，退出码是 2，成功结果仍会保留。

常用导入参数：

```bash
# 控制块大小和重叠
python -m rag_agent ingest data/inbox --max-chars 1200 --overlap-chars 120

# 对文本不足的 PDF 页面启用 OCR
python -m rag_agent ingest data/inbox --ocr --ocr-language "chi_sim+eng"

# 只抽取和分块，不构建向量索引
python -m rag_agent ingest data/inbox --no-index

# 在已有 chunks/manifest/index 上只处理新增、变化和删除的来源
PYTHONPATH=src python -m rag_agent ingest data/inbox --incremental

# 绝对路径可以把产物写到指定位置；相对路径以项目根目录为基准
python -m rag_agent ingest "/path/to/docs" --index "/path/to/vectors.jsonl"
```

## 检索

导入完成后可以先不调用聊天模型，直接检查检索质量：

```bash
python -m rag_agent search "Query、Key 和 Value 有什么关系" --top-k 5
```

机器可读 JSON：

```bash
python -m rag_agent search "三次握手" --top-k 3 --json
```

结果会包含：

```text
score、chunk_id、source_path、file_type、page_start/page_end、heading_path、text
```

可用 `--min-score` 调整最低相似度，`--file-type pdf` 或 `--source /绝对路径` 做过滤。hash embedding 是可解释的离线基线，不等同于训练好的语义 embedding；它适合先验证工程链路，中文语义质量有限。

## 离线评测

可以用 JSONL 标注集检查检索质量，不需要聊天模型或网络。每行至少包含一个
相关 chunk 或来源路径标注：

```json
{"name":"attention","query":"Query、Key 和 Value 有什么关系？","relevant_chunk_ids":["chunk-attention"]}
{"query":"TCP 如何建立连接？","relevant_source_paths":["/docs/network.txt"]}
```

字段说明：`query` 必填；`relevant_chunk_ids` 和 `relevant_source_paths` 都是
字符串数组，至少有一个非空；`name`（也可写 `id`）用于逐条结果标识；`min_score`
可为单条查询覆盖命令行阈值。来源路径必须与索引中的 `source_path` 完全一致。

运行评测：

```bash
python -m rag_agent evaluate data/eval/retrieval.jsonl --top-k 5 --json
```

报告包含 `Recall@K` 和 `citation_accuracy`。前者是每条查询在 Top-K 中命中任一
相关标签的比例；后者是返回结果中匹配标签的比例，是检索层的引用正确率代理，
不是聊天模型的事实正确率。报告同时保留每条查询的结果、命中 chunk/source 和
实际阈值，空评测文件或空标注会直接报错。

### 真实 CLI 闭环验证

在提交代码前，使用临时知识库实际跑过一遍导入、检索、评测和证据注入链路。验证
使用项目自己的 README 与工程规划作为两份 Markdown 来源，命令形态如下（输出路径
可以替换为任意临时目录）：

```bash
python -m rag_agent ingest /path/to/inbox \
  --output /tmp/rag-run/chunks.jsonl \
  --manifest /tmp/rag-run/documents.jsonl \
  --failures /tmp/rag-run/failures.jsonl \
  --index /tmp/rag-run/vectors.jsonl \
  --embedding-provider chinese --embedding-dimension 128

python -m rag_agent search "报告包含 Recall@K 和 citation_accuracy" \
  --index /tmp/rag-run/vectors.jsonl \
  --embedding-provider chinese --embedding-dimension 128 \
  --top-k 5 --min-score 0.08 --json

python -m rag_agent evaluate /path/to/evaluation.jsonl \
  --index /tmp/rag-run/vectors.jsonl \
  --embedding-provider chinese --embedding-dimension 128 \
  --top-k 5 --min-score 0.08 --json

python -m rag_agent ask "报告包含 Recall@K 和 citation_accuracy" \
  --index /tmp/rag-run/vectors.jsonl \
  --embedding-provider chinese --embedding-dimension 128 \
  --top-k 5 --min-score 0.08 --dry-run --json
```

本次实跑结果：2 个文档成功、0 个失败、162 个 chunk；索引使用
`chinese:chinese-ngram-v1:d128`。标注集两条查询的 `Recall@5` 为 `1.0`，
`citation_accuracy` 为 `0.4`（这是检索引用正确率代理，不是模型事实正确率）。
`ask --dry-run` 的第一条证据来自标注的“阶段 7 推进记录：离线评测”段落，并保留
来源路径、章节和 `chunk_id`；把陌生问题的阈值提高到 `0.5` 时返回“知识库中没有找到
足够相关的内容”，不会调用聊天 API。

这证明了从文件抽取、分块、embedding、向量索引、Top-K 检索到 RAG 证据上下文的真实
离线闭环。当前环境没有 `LLM_API_KEY`，所以尚未声称外部聊天模型生成成功；配置有效
的 OpenAI 兼容 Key 后，再按“RAG 回答”章节运行不带 `--dry-run` 的命令，验证生成文本
和 `[1]`、`[2]` 引用即可。项目保持 CLI-only，不依赖 Web UI。

同一流程有离线集成测试 `tests/integration/test_cli_rag.py`，运行全部测试：

```bash
python -m pytest -q
```

当前回归结果为 59 个测试通过；测试不需要网络、聊天 Key 或 Web UI。

查看文档清单：

```bash
python -m rag_agent list-documents
python -m rag_agent list-documents --json
```

## RAG 回答

### 先做 dry-run（不需要聊天 Key）

```bash
python -m rag_agent ask "Transformer 的注意力是怎么计算的" --dry-run
```

这会显示将要发送给模型的、带编号的证据。证据不足时会直接返回“知识库中没有找到足够依据”，不会调用模型。

### 接入 DeepSeek

DeepSeek 的聊天接口可以使用 OpenAI SDK 的兼容形式，但聊天 Key 不代表一定提供 Embedding 接口。因此本项目把聊天模型和 Embedding 分开配置：

```bash
export LLM_API_KEY='你的 DeepSeek Key'
export LLM_BASE_URL='https://api.deepseek.com'
export LLM_MODEL='deepseek-chat'

python -m rag_agent ask "你的问题" --top-k 5
```

也可以只在命令中传入聊天参数：

```bash
python -m rag_agent ask "你的问题" \
  --llm-api-key '你的 DeepSeek Key' \
  --llm-base-url 'https://api.deepseek.com' \
  --llm-model 'deepseek-chat'
```

Key 不要带额外的双引号字符；Shell 命令中的单引号只是防止特殊字符被 Shell 解释，实际传给程序的值不包含引号。

返回答案时会保留模型使用的 `[1]`、`[2]` 证据编号，并在答案下方列出对应的文件、页码、标题和 `chunk_id`。

### 更换 Embedding

默认配置是：

```text
EMBEDDING_PROVIDER=hash
```

如果你有确实支持 `/embeddings` 的 OpenAI 兼容服务，可以这样重建和查询：

```bash
export EMBEDDING_PROVIDER=openai
export EMBEDDING_API_KEY='你的 Embedding Key'
export EMBEDDING_BASE_URL='https://你的服务/v1'
export EMBEDDING_MODEL='你的 embedding 模型'

python -m rag_agent rebuild-index --embedding-provider openai
python -m rag_agent search "你的问题" --embedding-provider openai
```

索引首行会保存 provider、模型、维度和配置指纹。查询时配置不一致会明确报错，必须重新构建索引，不能混用不同模型生成的向量。

### 中文离线增强基线

没有可下载的本地神经模型时，可以使用内置的 `chinese` provider。它用中文字符
1-4 元组、英文/数字 token 和确定性特征哈希增强短语匹配，不需要网络、模型文件或
额外依赖。它比通用 `hash` 更适合中文资料，但仍属于词法基线，不等同于
`bge-m3`、`text2vec` 等训练好的语义 embedding。

```bash
python -m rag_agent ingest data/inbox \
  --embedding-provider chinese --embedding-dimension 512
python -m rag_agent search "三次握手如何建立连接" \
  --embedding-provider chinese --embedding-dimension 512
```

切换 provider 会改变索引指纹，必须重新构建索引；不能用 `hash` provider 查询
`chinese` 索引，程序会明确拒绝配置不一致。

## Agent 工具循环

`ask` 是固定的“先检索、再回答”流程，便于调试。`agent` 是真正的工具调用
流程：模型看到唯一的 `search_knowledge_base` 工具，自己决定是否检索；Python
验证 `query/top_k/min_score`，执行只读索引查询，再把结果交还模型。最多执行
`--max-steps` 轮，避免模型反复调用。

```bash
python -m rag_agent agent "Transformer 的注意力如何计算" \
  --llm-api-key '你的 DeepSeek Key' \
  --llm-base-url 'https://api.deepseek.com' \
  --llm-model 'deepseek-chat'
```

需要审计完整状态时使用 JSON：

```bash
python -m rag_agent agent "你的问题" --json
```

Agent 不会把任意文件路径暴露给模型，也不会执行证据中的命令。若模型没有
调用工具，它可以直接回答；若达到轮数上限，命令返回退出码 2 并保留调用记录。

### LangGraph 状态图（可选）

默认 `agent` 使用无依赖的手写循环；安装 `.[graph]` 后，可以用同一套工具和
历史契约运行 LangGraph 编排：

```bash
python -m rag_agent agent "Transformer 的注意力如何计算" --graph --json
```

状态图固定包含 `agent`、`tools`、`finalize` 三个节点：模型节点决定回答或检索，
工具节点只执行已校验的 `search_knowledge_base`，达到步数上限或得到回答后进入
收口节点。`--json` 的 `state` 保存本轮消息、工具调用、证据、步数和停止原因，
可用于节点级审计。当前环境若未安装 LangGraph，`--graph` 会明确提示安装命令，
普通 `agent`、`ask`、`search` 和离线评测不受影响。

### 对话历史

Agent 可以用 JSONL 文件保存已完成的 user/assistant 问答轮次，并在下一次命令中
继续追问：

```bash
python -m rag_agent agent "Transformer 的注意力如何计算" \
  --history data/state/conversation.jsonl
python -m rag_agent agent "它为什么需要 Key？" \
  --history data/state/conversation.jsonl
```

`--history` 会读取旧历史并在成功完成本轮后原子更新；`--save-history` 可以把更新
后的历史写到另一个路径，`--history-max-turns` 控制新建历史最多保留的轮数（默认
20，最多 100）。历史文件只保存完整问答轮次，不保存悬空的工具调用；工具消息、
每步编号、检索证据和停止原因会保留在 `--json` 输出的 `state` 中。达到
`--max-steps` 的中止运行不会写入对话历史，避免把中止提示当成事实带入后续问题。

## 命令总览

```text
ingest <file-or-directory>   抽取、分块并默认构建索引
                             可加 --incremental 复用未变化来源
search <query>               查询已有索引
evaluate <jsonl>             用标注集离线评测 Recall@K 和引用正确率代理
ask <question>               检索证据并调用聊天模型
agent <question>             模型自主调用受控检索工具再回答
list-documents               查看导入清单
rebuild-index                用现有 chunks 重新生成索引
```

## 代码阅读顺序

1. `src/rag_agent/models.py`：`TextBlock`、`DocumentRecord`、`Chunk` 和来源身份。
2. `src/rag_agent/ingest/pipeline.py`：文件发现、大小限制和加载器选择。
3. `src/rag_agent/ingest/text.py`：UTF-8、UTF-16、GB18030 和规范化换行。
4. `src/rag_agent/ingest/markdown.py`：标题路径和代码围栏。
5. `src/rag_agent/ingest/pdf.py`：逐页文本层提取和 OCR 降级。
6. `src/rag_agent/chunking/splitter.py`：自然边界、overlap 和稳定 ID。
7. `src/rag_agent/embeddings/`：可替换的 embedding 接口和离线基线。
8. `src/rag_agent/retrieval/index.py`：索引格式、余弦相似度和来源过滤。
9. `src/rag_agent/answering/rag.py`：上下文上限、证据隔离、引用解析。
10. `src/rag_agent/evaluation.py`：JSONL 标注校验、Recall@K 和引用正确率代理。
11. `src/rag_agent/agent/history.py`：有界对话历史、JSONL 持久化和消息转换。
12. `src/rag_agent/agent/runtime.py`：工具循环、AgentState 快照和停止原因。
13. `src/rag_agent/agent/graph.py`：可选 LangGraph 节点、路由和状态转换。
14. `src/rag_agent/cli.py`：命令行如何把各层串起来。

## 当前边界和下一步

当前版本没有引入 FAISS、Chroma 或 Qdrant；LangGraph 已作为可选编排层接入，默认
仍使用无依赖的手写 Agent 循环。底层数据、检索结果、对话历史和节点状态均保持可
观察、可测试，真实 CLI RAG 离线闭环已经验证；后续阶段只继续增强 PDF 版面/OCR 和
CLI 发布体验。

计划中的后续工作：

```text
已完成：导入 / OCR 降级 / 分块 / 本地 embedding / Top-K 检索 / 引用上下文 / 受控工具循环
已推进：增量导入基础版（新增/修改/删除、失败保留旧版本、向量复用）
已推进：离线评测集（Recall@K、引用正确率代理）
已推进：对话历史基础版（有界 JSONL、跨轮 Agent 上下文、可观察状态）
已推进：中文离线增强 embedding（字符 n-gram + token，免模型下载）
已完成：LangGraph 状态图基础版（可选依赖，保留手写循环基线）
已验证：真实 CLI 离线 RAG 闭环（ingest/search/evaluate/ask --dry-run）
最后：表格/双栏版面、OCR 置信度、CLI 发布验证和权限隔离
```

文档内容始终被当作不可信证据。后续 Agent 不应允许模型读取任意文件路径、执行文档中的命令，或让检索文本覆盖系统提示词。

## Git 与下载运行

仓库使用 `main` 作为默认分支。原始资料、Key、`.venv`、缓存、索引和失败日志都由
`.gitignore` 排除，提交中只保留源码、测试、依赖声明、目录占位文件和技术文档。

下载仓库后，按下面步骤即可运行 CLI（不需要 Web UI）：

```bash
git clone <repository-url>
cd rag-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH=src
python -m rag_agent --help
```

本地 Git 已初始化并创建提交；远端 URL 和认证不写入代码或文档。首次推送前需要先
完成 GitHub CLI 登录，再在仓库目录执行：

```bash
gh auth login -h github.com
git remote add origin <repository-url>
git push -u origin main
```

如果 `origin` 已存在，改用 `git remote set-url origin <repository-url>`。当前开发环境
的 GitHub token 已失效且尚未配置 `origin`，所以远端创建与推送必须在有效认证后继续；
不能把本地提交称为已发布。
