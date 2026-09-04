# RAG 学习路线与目录

这套文档只讲如何设计和实现一个可运行的 RAG（Retrieval-Augmented Generation，检索增强生成）系统，不展开 Transformer 或 LLM 的内部结构。

## 先建立一张总图

一个最小但完整的 RAG 系统可以写成下面这条数据流：

```text
原始资料
  -> 文件发现与格式解析
  -> 统一文本块 TextBlock
  -> 切分为 Chunk
  -> 为每个 Chunk 计算 Embedding
  -> 保存 Chunk、元数据和向量索引

用户问题
  -> 用同一个 Embedding 配置计算查询向量
  -> 在索引中做相似度检索
  -> 取 Top-K 证据并应用阈值/过滤
  -> 拼成受边界保护的 Evidence Context
  -> 交给聊天模型生成回答和引用
```

RAG 的关键不是“让模型记住文件”，而是每次提问时临时找到与问题最相关的证据，再把证据放进当前请求。资料更新时通常更新索引，而不是重新训练聊天模型。

## 这套教程回答什么问题

- 为什么导入、分块、Embedding、索引和生成必须分层。
- 一个文件如何变成可以检索的 `Chunk`。
- 向量为什么能用于相似度检索，以及本项目如何实现可检查的本地索引。
- `top-k`、`min-score`、上下文长度和元数据过滤如何影响结果。
- 怎样判断问题出在资料解析、检索还是模型回答。
- 如何做离线评测、增量更新、错误保留和安全边界。
- 为什么 `ask` 是固定 RAG 流程，`agent`/LangGraph 是在它之上增加的工具编排。

## 推荐阅读顺序

1. [[01_RAG的边界与整体架构]]：先区分 RAG、普通问答和模型微调。
2. [[02_资料导入与数据契约]]：从 PDF/Markdown/TXT 到统一数据对象。
3. [[03_分块与元数据设计]]：决定检索证据是否完整、可引用。
4. [[04_Embedding与向量索引]]：从文字到向量，再到可持久化索引。
5. [[05_检索器与相似度搜索]]：理解 Top-K、阈值、过滤和配置一致性。
6. [[06_证据上下文与RAG回答]]：把检索结果安全地交给聊天模型。
7. [[07_评测与调参方法]]：用数据定位问题，而不是凭感觉改参数。
8. [[08_增量更新、安全与工程化]]：处理资料变化、失败、权限和可观察性。
9. [[09_Agent与LangGraph]]：从固定 RAG 到受控工具循环和状态图。
10. [[10_本项目源码阅读与动手路线]]：按真实源码运行实验、验证每一层。

## 当前项目对应关系

| RAG 层 | 本项目实现 |
| --- | --- |
| 文件发现和解析 | `src/rag_agent/ingest/pipeline.py`、`text.py`、`markdown.py`、`pdf.py` |
| OCR 降级 | `src/rag_agent/ingest/ocr.py`、`pdf.py` |
| 数据契约 | `src/rag_agent/models.py` |
| 分块 | `src/rag_agent/chunking/splitter.py` |
| Embedding | `src/rag_agent/embeddings/` |
| 向量索引和检索 | `src/rag_agent/retrieval/index.py` |
| 证据上下文和固定问答 | `src/rag_agent/answering/rag.py` |
| OpenAI 兼容聊天调用 | `src/rag_agent/answering/chat.py` |
| 离线检索评测 | `src/rag_agent/evaluation.py` |
| 增量导入 | `src/rag_agent/ingest/incremental.py` |
| Agent 工具边界 | `src/rag_agent/agent/knowledge_tool.py`、`runtime.py` |
| LangGraph 编排 | `src/rag_agent/agent/graph.py` |
| CLI 组装 | `src/rag_agent/cli.py` |

## 学习时的核心判断

每次看到一个组件，都问四个问题：

1. 它接收什么数据，输出什么数据？
2. 它解决 RAG 链路中的哪一个具体问题？
3. 它失败时如何报告，能否保留已有正确结果？
4. 它的输出如何被测试、追踪和复现？

如果一个改动无法回答这四个问题，就先不要把它称为“RAG 优化”。

## 当前实现的边界

- 默认索引是 JSONL 文件加内存中的余弦相似度扫描，适合学习和中小规模个人资料。
- `hash` 和 `chinese` provider 是本地、可解释的词法基线，不等同于训练好的神经语义 Embedding。
- DeepSeek 只负责聊天生成；是否调用它由 `ask`/`agent` 决定。
- `ask --dry-run` 能验证检索和证据上下文，但不验证外部模型生成质量。
- LangGraph 是可选编排层，不能替代解析、分块、索引和检索本身。
