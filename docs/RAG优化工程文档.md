# RAG 优化工程文档

本文档回答两个问题：**这个 RAG 项目应该从哪几个方面优化**，以及**每个优化项具体怎么做、做到什么程度算完成**。任务使用统一编号（OPT-xx），每项都写明现状、做法、涉及模块和验收标准，便于后续逐项执行和回归。

> 使用约定：执行任何一项优化前，先确认第 1 节的基线可复现；完成一项后必须跑通 `python -m pytest -q` 和第 5 节的评测流程，用数据证明"有效"再合并。代码分层与设计取舍见 [工程规划.md](工程规划.md)，链路原理教学见 [docs/rag-guide/](rag-guide/00_RAG学习路线_目录.md)。

---

## 0. 全任务总览（2026-09-09 快照）

共 **24 个编号任务**（OPT-01～24，04/05 合并计）。状态标记：✅ 已落地 / 🔶 部分落地 / ⬜ 未开始。

| 编号 | 类别 | 任务 | 优先级 | 状态 | 备注 / 证据 |
| --- | --- | --- | --- | --- | --- |
| OPT-01 | 评测 | golden set 标注集 | P0 | 🔶 | 28 题种子入库（22 auto + 6 陷阱）；**人工复核待做** |
| OPT-02 | 评测 | MRR / nDCG 排序指标 | P0 | ⬜ | harness 有命中率，无排序质量指标——**当前最短木板** |
| OPT-03 | 评测 | 答案级评测（忠实度+引用校验） | P1 | ✅ | citation_validity + 锚点 rubric 裁判已在 harness 落地 |
| OPT-04/05 | 分块 | 抽取与 OCR 质量（表格/后处理/失败统计） | P2 | ⬜ | 爬虫侧真实站点修复完成；本地抽取侧未动 |
| OPT-06 | 分块 | Token 感知分块（token-aware-v1） | P1 | ⬜ | 需注册新指纹，旧索引隔离 |
| OPT-07 | 分块 | 结构感知分块（标题树/段落收束） | P1 | ⬜ | heading_path 已采集未用于分块决策 |
| OPT-08 | 分块 | 父子块 small-to-big | P2 | ⬜ | 需加 parent_chunk_id 契约字段 |
| OPT-09 | Embedding | 接入神经 Embedding | P0 | ⬜ | **预期收益最大**；fake 评测已证词面失配是检索瓶颈 |
| OPT-10 | Embedding | 向量缓存（重建不重复花钱） | P1 | ⬜ | chunk 级增量复用已有；provider 级缓存没有 |
| OPT-11 | 检索 | BM25 + 向量混合（RRF 融合） | P0 | ⬜ | 救缩写/专有名词类查询 |
| OPT-12 | 检索 | 重排序 Rerank（带零依赖降级） | P1 | ⬜ | 排在 OPT-09/11 之后才有意义（垃圾进垃圾出） |
| OPT-13 | 检索 | 查询改写（历史感知 + multi-query/HyDE） | P1 | 🔶 | agent 模式模型自发改写重查（实测救回 4 题）；代码化实现未做 |
| OPT-14 | 索引 | 索引后端 SQLite/FAISS | P2 | ⬜ | 压测证明万级 41ms 够用，维持低优先级 |
| OPT-15 | 生成 | 证据上下文组织 | P2 | ⬜ | WT-AI 实验3 证明位置效应仅 0.02，重排部分已降级删除 |
| OPT-16 | 生成 | 引用与拒答强化 | P1 | 🔶 | 完全体引用已带本地/网络标注；不实引用自动复核未做 |
| OPT-17 | Agent | 检索质量自查与回退（CRAG） | P1 | 🔶 | **提示层已落地**（充分性评估+网络兜底，冒烟验证）；结构化判别器遗留——治首轮唯一的编造失败 |
| OPT-18 | Agent | 流式输出（CLI） | P2 | ⬜ | iOS 端已有 SSE；CLI 没有 |
| OPT-19 | 工程 | 检索日志与观测 + 坏案例反哺 | P2 | ⬜ | — |
| OPT-20 | 工程 | 查询缓存 / 批量 embed / 分片重试 | P2 | ⬜ | — |
| OPT-21 | 爬取 | 网页抓取增强 | P2 | 🔶 | JS 渲染 ✅（A/B 验证）；遗留 sitemap/RSS、子域、HTML 快照、cookie |
| OPT-22 | 评测 | 拒答率监控（WT-AI 增补） | P1 | ✅ | harness 双向指标+交叉归因；首轮：陷阱 5/6、误拒 0 |
| OPT-23 | 生成 | reasoning 模型 max_tokens 防御 | P0 | 🔶 | 裁判路径已防（3000+明确报错，零判决错误）；**主回答路径 chat.py 未改** |
| OPT-24 | Agent | 长会话历史：固定窗口 → 按需检索 | P2 | ⬜ | WT-AI 数据：固定窗口早期信息召回 0% |

**状态统计**：✅ 2 项 / 🔶 7 项 / ⬜ 15 项（04/05 计 1 项）。

**推荐执行顺序（数据驱动，2026-09）**：

1. **OPT-23 主路径收尾**——半小时量级，OPT-22/17 的评测与防御已就位，只差 `chat.py` 的 finish_reason 提示；
2. **OPT-02 补 MRR/nDCG**——让后续 OPT-09/11/12 的收益可量化（评测闭环已建好，缺的就是排序指标）；
3. **OPT-09 神经 embedding**——首轮评测三次归因（fake 词面失配、陷阱噪声底、coverage 短板）全部指向它；
4. 之后按第 4 节路线图走，每项改动前后用 `rag-agent harness --real` 跑基线对比（见 Harness 文档）。

---

## 1. 优化从度量开始：先有基线，再谈优化

RAG 优化最容易犯的错误是"凭感觉调参数"。本项目当前的度量能力：

| 度量 | 现状 | 位置 |
| --- | --- | --- |
| Recall@K | 已实现，`evaluate` 命令离线计算 | `src/rag_agent/evaluation.py` |
| 引用正确率代理（citation proxy） | 已实现，检查 Top-K 是否覆盖标注来源 | `src/rag_agent/evaluation.py` |
| **Agent 级评测闭环** | **已实现（2026-09）**：题库+跑批+确定性指标+LLM 裁判+校准+快照，`harness` 命令 | `src/rag_agent/harness/`（[Agent评测Harness工程文档.md](Agent评测Harness工程文档.md)） |
| 排序质量（MRR / nDCG） | 缺失 | — |
| 答案质量（忠实度、答案相关性） | **已实现（LLM 裁判，带锚点 rubric 与校准）** | `harness/judge.py` |
| 标注集（golden set） | **种子已落地（28 题：22 auto + 6 陷阱），人工复核待做** | `data/eval/agent_tasks.jsonl` |

**因此第 0 优先级不是改检索，而是补齐度量**：没有 golden set 和排序指标，后面任何检索优化的收益都无法量化，也无法防止"这次改好了 A 场景、弄坏了 B 场景"。首轮真实评测的基线数字与三个发现见 Harness 文档附录 B。

---

## 2. 现状盘点：按数据链路

当前链路与已知短板（路径均为 `src/rag_agent/` 下相对路径）：

| 环节 | 现状 | 关键短板 |
| --- | --- | --- |
| 抽取 ingest | PyMuPDF 抽取 + 可选 Tesseract OCR，MD/TXT 直读；带 ingestion 指纹和失败清单（`ingest/`） | 表格、双栏、页眉页脚噪声未专门处理 |
| 分块 chunking | 字符级滑窗，`max_chars=1200`、`overlap_chars=120`，在中英文标点和换行处寻找自然边界；带 heading_path/page 元数据（`chunking/splitter.py`） | 不感知 token 数；跨标题边界会切断语义单元；无父子块结构 |
| Embedding | `hash`（离线哈希基线）、`chinese`（字符 n-gram 基线）、`openai`（OpenAI 兼容接口）三选一（`embeddings/`） | 默认基线语义能力弱；无向量缓存，重复 ingest 重复计算 |
| 索引 | JSONL 全量内存加载 + 暴力余弦扫描，schema v1，支持增量更新与指纹复用（`retrieval/index.py`） | 规模上万 chunk 后加载慢、内存高；无 ANN |
| 检索 | 单路稠密 Top-K + min_score 过滤，仅支持 source_path/file_type 过滤（`retrieval/index.py:search`） | 无关键词（BM25）一路；无重排；无查询改写；短查询/专有名词命中率差 |
| 回答 | `RagAnswerer` 单轮检索 + 证据上下文（默认 8000 字符）+ DeepSeek 生成 + `[n]` 引用解析；无证据时明确拒答（`answering/rag.py`） | 上下文按顺序截断（易受 lost-in-the-middle 影响）；引用只解析编号不做内容校验 |
| Agent | 受控检索工具 + 自主多步，可选 LangGraph，JSONL 历史续问（`agent/`） | 检索质量差时无自查/回退机制 |
| 评测 | `evaluate` 离线 Recall@K + citation proxy（`evaluation.py`） | 见第 1 节 |

---

## 3. 优化任务清单

优先级定义：**P0** 立即做（其他优化的前置）；**P1** 核心质量提升；**P2** 收益明确但不紧急。

### 3.1 评测体系（P0）

#### OPT-01 构建 golden set 标注集（P0）
- **现状**：没有任何固定评测集，README 中的 JSONL 示例是单条演示。
- **做法**：从 `data/inbox` 真实资料中选 50–100 个有代表性的问题，人工标注相关 chunk_id / source_path，格式沿用现有契约（`query` + `relevant_chunk_ids` 或 `relevant_source_paths`）。覆盖四类查询：关键词精确匹配、语义改述、跨章节综合、知识库中没有的"陷阱题"。落盘 `data/eval/retrieval.jsonl`（不提交真实资料时提交脱敏样例）。
- **验收**：`python -m rag_agent evaluate data/eval/retrieval.jsonl --top-k 5 --json` 可复现固定基线数字；评测集纳入版本管理。

#### OPT-02 补齐排序指标 MRR / nDCG（P0）
- **现状**：Recall@K 只回答"有没有命中"，不回答"排得靠不靠前"，对重排、混合检索类优化不敏感。
- **做法**：在 `evaluation.py` 的报告结构中增加 `mrr` 与 `ndcg@k`，兼容现有 `relevant_chunk_ids`（多标注按时位折损）与 `relevant_source_paths` 两种标注。
- **验收**：评测输出包含全部指标；单测覆盖多标注与未命中两种情况；现有 Recall@K 结果不变。

#### OPT-03 答案级评测：忠实度与引用校验（P1）
- **现状**：答案质量只能人工抽查；`answering/rag.py` 的 citation proxy 只在检索层。
- **做法**：两步走——(a) 离线可判定的"引用真实性"校验：回答中每个 `[n]` 必须存在，且被引 chunk 的文本与被支撑的句子有词面/向量重叠（可先做词面版）；(b) 忠实度/答案相关性评测接入 LLM-as-judge（DeepSeek 打分，注明这是成本项、非离线项），评测脚本放 `scripts/`，不进默认测试。
- **验收**：`evaluate --check-citations` 能给出引用真实性比例；LLM-judge 脚本输出 JSON 报告样例。

### 3.2 Embedding 与索引（P0）

#### OPT-09 接入神经 Embedding（P0，预期收益最大）
- **现状**：默认 `hash` 是确定性哈希，`chinese` 是 n-gram 基线，两者都没有真正的语义能力——这是当前检索质量的最大瓶颈。
- **做法**：首选通过现有 `openai` provider 接入支持中文的 embedding 服务（如硅基流动/阿里云等提供的 `bge-large-zh`、`gte-large-zh` 类模型，配置 `EMBEDDING_BASE_URL/EMBEDDING_MODEL`）；其次可选本地 `sentence-transformers`（新增可选依赖组 `[local-embedding]`，不进默认依赖）。接入后用 OPT-01 标注集对比三种 provider 的 Recall@5 / MRR。
- **验收**：新 provider 在 golden set 上 Recall@5 相对 `chinese` 基线提升可量化（预期显著）；`rebuild-index` 换 provider 流程顺畅；指纹机制防止新旧向量混用（已有）。

#### OPT-10 Embedding 向量缓存（P1）
- **现状**：增量导入已按 chunk 指纹复用向量（`update_vector_index`），但换 provider / 重建全库时全部重算，远程 API 重复花钱。
- **做法**：向量以 `(provider_fingerprint, content_hash)` 为键落盘缓存（如 `data/cache/embeddings.jsonl` 或 SQLite），`build_vector_index` 命中缓存则跳过 embed 调用。
- **验收**：同库二次重建时远程调用数为 0；`--no-cache` 可关闭；缓存键含 provider 指纹，绝不跨模型复用。

#### OPT-14 索引后端升级 SQLite/FAISS（P2）
- **现状**：JSONL 全量载入内存做暴力余弦，万级 chunk 内够用，但启动加载是 O(全库)。
- **做法**：抽象索引接口（`load/search/save`），新增 SQLite + numpy 或 FAISS 后端，JSONL 保留为默认可检查格式和导出格式。`retrieval/index.py` 模块注释已预留此演进方向。
- **验收**：后端可通过配置切换；1 万 chunk 级别加载 < 1s；JSONL↔新后端可互转且检索结果一致（确定性排序不变）。

### 3.3 检索策略（P0）

#### OPT-11 BM25 + 向量混合检索（P0）
- **现状**：只有稠密向量一路。课程编号、专有名词、缩写这类查询对哈希/语义向量都不友好，而关键词检索恰好擅长。
- **做法**：新增 `retrieval/lexical.py`：BM25（中文按字/ jieba 分词二选一，先做无依赖的字符级 BM25，避免引入分词依赖）在 chunks 上建立倒排；检索时向量 Top-K 与 BM25 Top-K 用 RRF（Reciprocal Rank Fusion）融合。`search/ask/chat` 加 `--hybrid`（或配置默认开）。
- **验收**：golden set 上含专有名词的子集 Recall@5 提升；整体指标不回退；纯离线可测（不依赖任何 API）。

#### OPT-12 重排序 Rerank（P1）
- **现状**：Top-K 直接按余弦排序作为最终结果，粗排即终排。
- **做法**：召回扩大（如 Top-20）后用 rerank 精排到 Top-5。优先接入 OpenAI 兼容的 rerank 服务或本地 cross-encoder（作为可选依赖）；同时实现一个零依赖的降级方案（query–chunk 词面重叠加权），保证离线路径可用。
- **验收**：golden set MRR / nDCG@5 提升可量化；rerank 不可用时自动降级并给出警告；`--no-rerank` 可关闭。

#### OPT-13 查询改写（P1）
- **现状**：用户原句直接 embed，口语化/指代型问题（"它为什么需要 Key？"）脱离上下文后向量质量差。
- **做法**：两级实现——(a) agent 历史存在时，把最近一轮话题拼进检索查询（纯本地，零成本）；(b) 可选 LLM 改写：multi-query（生成 2–3 个改写并行检索合并去重）或 HyDE，仅在调用聊天模型的功能路径中启用。
- **验收**：历史续问场景检索命中率提升；(a) 完全离线可测；(b) 有开关且默认可关。

### 3.4 数据与分块（P1）

#### OPT-06 Token 感知分块（P1）
- **现状**：`max_chars` 按字符数控制块大小，中英文混排时实际 token 数波动大。
- **做法**：`ChunkConfig` 增加 token-aware 策略（估算器即可：中文≈1 字 1 token、英文按 4 字符 1 token，或接入 tokenizer），在数据契约 `Chunk` 不变的前提下把策略注册进 `fingerprint`（现有 `character-natural-boundary-v1` 旁边加 `token-aware-v1`）。
- **验收**：同一文档新策略生成的 chunk token 分布更集中；旧策略结果完全不变（指纹隔离）；单测覆盖中英混排。

#### OPT-07 结构感知分块（P1）
- **现状**：滑窗会在标题边界切断语义；Markdown 已有 heading_path 但分块不按标题收束。
- **做法**：Markdown 按标题树分块（小节优先成块，超长再滑窗）；PDF 优先按段落收束，页眉页脚/目录等噪声行在抽取层过滤。
- **验收**：chunk 的 heading_path 与文本一致率提升；golden set 上"按章节提问"子集命中率提升。

#### OPT-08 父子块 small-to-big（P2）
- **现状**：一个 chunk 既是检索单元也是引用/上下文单元，块小则上下文碎，块大则向量糊。
- **做法**：检索用子块（小、准），命中后向上取父块（整节）拼上下文。数据契约需增加 `parent_chunk_id`。
- **验收**：回答中证据上下文的完整性提升（人工评估 + OPT-03 引用校验）；不改变现有 chunk_id 生成兼容性。

#### OPT-04/05 抽取与 OCR 质量（P2）
- **现状**：表格和双栏版式抽取会错序；OCR 结果无后处理。
- **做法**：抽取层增加表格块类型与最小表格启发式；OCR 常见错误后处理（全半角、断行合并）；失败清单（`data/failed/ingestion.jsonl`）已有，补充失败原因分类统计。
- **验收**：对一页含表格/扫描件的真实 PDF 出具前后对比样例；失败清单可按原因汇总。

### 3.5 生成层（P1）

#### OPT-15 证据上下文组织（P1）
- **现状**：`build_evidence_context` 按分数顺序拼接、尾部直接截断；LLM 对中段信息注意力弱（lost-in-the-middle），且重要证据可能被整体截掉。
- **做法**：按"分数最高放首尾"重排证据顺序；超预算时优先丢弃低分证据而不是截断高分证据的尾巴；可选对超长 chunk 做句子级压缩。
- **验收**：单测覆盖预算分配逻辑；golden set 上答案级指标（OPT-03）不回退或提升。
- **优先级修订（2026-09，依据实测）**：Lost-in-the-middle 复现实验（20 题 × 100 干扰段、
  约 3.4 万字上下文）显示开头/中间/结尾三位置平均分差异仅 0.02（4.95/4.94/4.96），
  **位置效应在常见证据预算（8000 字符）下可忽略**。本项降级为 P2，仅保留
  "超预算时整条丢弃低分证据、不截断高分证据尾巴"的预算分配改进。

#### OPT-16 引用格式与拒答强化（P1）
- **现状**：已有明确拒答与 `[n]` 引用解析；但模型偶发编造编号或引用不支撑结论。
- **做法**：prompt 中增加"每个结论句必须带编号，无对应证据不得引用"的硬约束；结合 OPT-03 校验器对不实引用做二次提示或标记。
- **验收**：`ask --json` 输出中 citation 与 evidence 一致率可度量提升。

### 3.6 Agent 与交互（P2）

#### OPT-17 检索质量自查与回退（CRAG 思路）（P1，部分已落地）
- **现状（2026-09 更新）**：完全体 Agent 已落地**提示层**的自查与回退——工作流引导
  提示要求模型先本地检索、评估证据充分性（改写重查一次），不足时通过
  `fetch_web_page` 自主上网核实（单页抓取+链接多跳，护栏见
  [Agent工作流工程文档.md](Agent工作流工程文档.md)）。真实冒烟验证：本地无关时
  模型两次改写重查后自选 docs.python.org 抓取并给出带 URL 引用的回答。
- **剩余**：结构化 CRAG 评估器（用检索分数判别器代替模型自评、低分自动触发
  改写/换源）与多检索策略回退仍是遗留。
- **验收**：陷阱题（知识库没有的内容）拒答率提升；正常问题步数不显著增加；
  四条工作流路径已有离线剧本测试锁定（tests/unit/test_full_agent_workflow.py）。

#### OPT-18 流式输出（P2）
- **现状**：`ask`/`chat` 等完整响应结束才打印。
- **做法**：ChatProvider 增加 `stream` 能力，`chat` REPL 逐 token 输出。
- **验收**：首字延迟可感知下降；`--json` 模式保持整块输出不变。

### 3.7 工程与观测（P2）

#### OPT-19 检索日志与观测（P2）
- **现状**：检索分数只在 stdout 展示，无法离线分析坏案例。
- **做法**：`ask/search/chat` 可选把每轮 `(query, top_k 结果, 分数, provider, 耗时)` 追加到 `data/state/query_log.jsonl`；配套一个坏案例抽样脚本，输出"低分/零命中"查询清单反哺 OPT-01 标注集。
- **验收**：日志可开关（默认关），含 schema 版本；脚本产出 Top 坏案例清单。

#### OPT-20 性能：批量与缓存（P2）
- **现状**：查询路径无缓存；同问题重复全量检索。
- **做法**：查询向量 LRU 缓存；`evaluate` 批量 embed；远程 embedding 批量分片带重试。
- **验收**：重复评测耗时下降；失败重试有单测。

### 3.8 数据获取扩展（P2）

#### OPT-21 网页抓取增强（P2）
- **现状**：`ingest-url` 已支持静态抓取与 `--render-js` 无头浏览器动态渲染
  （爬虫调度、清洗整理、增量复用见 [网页爬取功能文档.md](网页爬取功能文档.md)；
  JS 渲染已在真实动态站点上完成 A/B 验证）。
- **剩余优化项**：(a) 支持 sitemap.xml / RSS 作为页面发现来源，替代逐链接 BFS；
  (b) 同域规则支持子域（`*.example.com`）；(c) 可选的 raw HTML 快照目录，便于审计；
  (d) 登录态/cookie 注入。
- **验收**：对一个含 sitemap 的站点出具覆盖对比；增量复用行为不回退
  （重复抓取 `vectors_embedded=0`）。

### 3.9 来自实测实验的改进输入（WT-AI 复盘，2026-09）

对另一个 Agent 问答项目（WT-AI，1580 题知识库 + 评测闭环）的四组实验做了一次复盘，
结论对本项目的优化清单有三条直接输入和一处修正。实验数据原文见
`/Users/Zhuanz/Documents/WT-AI/`（实验1/3/4、Harness 对比）。

| 实验 | 关键数据 | 对本项目的启示 |
| --- | --- | --- |
| 实验4 召回率诊断 | 用原题检索自身卡：rank1 96.8%、未召回 0% | 检索层健康度要用"自身卡排名"度量，且要与答题质量交叉归因 |
| 实验1 失败根因 | 575 道失败题：拒答 89.2%、跑题 5.9%、漏要点 3.7%、事实错误 0.7% | 检索好的系统，失败集中在生成端；**拒答率**应成为一等公民指标 |
| 实验3 LostInTheMiddle | 三位置平均分差 0.02（4.95/4.94/4.96） | 位置效应在本项目证据预算下可忽略 → OPT-15 降级（见上） |
| Harness 上下文策略 | window 召回 0%；rag 式历史检索 100% 且 token 接近 window | 长会话历史不该用固定窗口，按需检索历史是更优解 |
| Harness 注脚 | reasoning 模型 max_tokens=400 被推理内容吃光 → content 为空 → 召回 0% | 本项目 `max_tokens` 默认 1200，对 reasoning 模型同样危险 |

#### OPT-22 答案级失败归因：拒答率监控（P1，已落地）
- **现状（2026-09 更新）**：harness 已实现拒答双向指标（陷阱题该拒没拒 +
  answerable 误拒）+ 裁判失败分类 + 交叉归因。首轮真实数据（附录 B）：
  陷阱 5/6 正确、answerable 0 误拒；真实失败模式不是"过度拒答"（WT-AI 的
  89%）而是**证据不足时编造**（1 例陷阱 + 2 例要点不足时用自身知识补全）——
  改进指向 OPT-17 结构化充分性评估。
- **剩余**：MRR/nDCG 排序指标（并入 OPT-02）；题库人工复核与扩充。

#### OPT-23 reasoning 模型的 max_tokens 防御（P0，改动极小）
- **现状**：`chat.py` 默认 `max_tokens=1200`；content 为空时报
  "聊天接口返回了空内容"，不提示原因。WT-AI 实测：reasoning 模型的
  reasoning_content 计入 max_tokens 预算，预算不足时 **content 直接为空**，
  表现为"莫名其妙的空回答"（他们第一版 compaction 因此召回 0%）。
- **做法**：空内容时读取 `finish_reason`，若为 `length` 则报错信息明确写
  "推理内容耗尽 max_tokens 预算，请调大 --max-tokens"；文档在配置参考中
  标注 reasoning 模型（deepseek-reasoner 类）需要显著更大的预算。
- **验收**：模拟 finish_reason=length 的桩测试；错误信息可指导用户自修复。

#### OPT-24 长会话历史：从固定窗口到按需检索（P2）
- **现状**：`ConversationHistory` 是固定 20 轮窗口，超出即截断最旧轮次。
  WT-AI 对比实验（12 轮干扰 + 3 组 needle）：固定窗口对早期信息召回 0%，
  而 **rag 式历史检索召回 100% 且 token 开销接近窗口策略**（compaction 虽然
  也 100% 但每轮额外 LLM 调用、naive 实现慢一个数量级）。
- **做法**：把已完成的问答轮写入一个独立的"会话索引"（本项目向量索引设施
  可直接复用），每轮提问时检索最相关的 K 条历史拼进上下文；固定窗口保留为
  降级模式。注意 WT-AI 的教训：若走 compaction 路线必须做增量摘要而非全量重摘。
- **验收**：长会话（>20 轮）早期信息问答命中率高于固定窗口模式；token 开销
  增幅可接受。

---

## 4. 执行路线图

按依赖关系分四个阶段，每阶段结束跑一次完整回归（`pytest` + golden set 评测），产出基线对比数字后再进入下一阶段。

| 阶段 | 目标 | 任务 | 验收 | 进度 |
| --- | --- | --- | --- | --- |
| 阶段一：度量与基线 | 让"有效"可证明 | OPT-01🔶、OPT-02⬜、OPT-10⬜ | 固定评测集入库，产出基线报告（Recall@5 / MRR / nDCG） | 01 种子已落地，基线报告已有（Harness 附录 B），差 MRR/nDCG |
| 阶段二：检索质量 | 解决最大瓶颈 | OPT-09⬜（神经 embedding）、OPT-11⬜（混合检索）、OPT-13a⬜（历史感知查询） | Recall@5 与 MRR 相对基线提升有数据；全部离线测试通过 | 未启动；首轮评测归因已三次指向本阶段 |
| 阶段三：精排与生成 | 把对的证据用好 | OPT-12⬜（rerank）、OPT-06/07⬜（分块升级）、OPT-15⬜、OPT-16🔶、OPT-03✅ | nDCG@5、引用一致率提升；答案级评测报告 | 03/16 部分完成 |
| 阶段四：体验与工程化 | 可持续迭代 | OPT-14⬜、OPT-17🔶（剩余判别器）、OPT-18⬜、OPT-19⬜、OPT-20⬜、OPT-08⬜、OPT-04/05⬜ | 索引切换基准达标；观测日志可用；回归全绿 | 17 提示层已落地 |

## 5. 优化有效性的判定流程（每项任务通用）

1. 改动前：在 golden set 上记录当前指标（`evaluate --json` 存档）。
2. 改动后：同集复测，**指标不回退且目标指标提升**才算有效；探索性实验在分支上做。
3. `python -m pytest -q` 全绿（当前 71 个测试是回归底线）。
4. 陷阱题子集（知识库外问题）拒答行为不回退。
5. 涉及成本的项目（远程 embedding/LLM-judge）报告调用量变化。

## 6. 约束与风险

- **离线优先**：仓库自带测试不允许依赖网络与真实 Key；所有需要远程服务的优化（OPT-09 远程方案、OPT-03b、OPT-12 远程方案）必须保留离线降级路径（基线 provider、词面 rerank、`--dry-run`）。
- **确定性**：检索与评测路径必须保持同输入同输出（现有 `search` 排序按 `(-score, chunk_id)` 确定性排序），融合/重排的新逻辑同样需要确定性，否则评测不可信。
- **数据契约稳定**：`Chunk`/manifest/索引 schema 变更必须升级对应 fingerprint 或 schema_version（现有机制），旧索引要么可迁移要么明确要求 `rebuild-index`。
- **安全边界不变**：文档内容始终是不可信证据；prompt 注入防护（`answering/rag.py` 系统规则）不因优化削弱；Key、`.env`、历史文件不入 Git。
- **成本**：远程 embedding 与 LLM-judge 都是花钱项，必须配 OPT-10 缓存与明确的调用量报告，避免评测一遍跑一遍钱。
