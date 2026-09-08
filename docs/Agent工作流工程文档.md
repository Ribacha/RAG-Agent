# Agent 工作流工程文档

> 本文档是"完全体 Agent"改造的**执行蓝本与进度账本**：定义产品形态、工作流编排、
> 决策规则与护栏，并用文末的里程碑清单跟踪执行状态。每个里程碑完成一提交一暂停，
> 用户确认后继续；任何新会话读完本文档即可恢复现场。

---

## 1. 产品形态：应用层两个功能

| 入口 | 是什么 | 何时用 |
| --- | --- | --- |
| **`rag-agent ask`** | 简单 RAG：固定一次本地检索 → 生成带引用回答 | 快、便宜、可预测的日常问答 |
| **`rag-agent agent`** | 完全体 Agent：本地优先检索 → 证据不足时**自主上网爬取** → 综合（含交叉验证） | 复杂问题、知识库没覆盖的问题、需要权威来源核实的问题 |

其余命令（ingest / ingest-url / search / evaluate / list-documents / rebuild-index / init / doctor）**不删除**，在 CLI help 与文档中归入"维护工具"分组。`chat` 保留为两种模式的交互式外壳（默认=会话版 ask，`--agent` 切完全体）。

## 2. 工作流编排（完全体 Agent）

```text
用户问题
   │
   ▼
①规划 ────────── 模型解析问题；系统提示引导：先判断本地知识库是否可能有答案
   │
   ▼
②本地检索 ────── search_knowledge_base（索引存在时总是先查；空索引跳过）
   │
   ▼
③充分性评估 ──── 证据够吗？（检索分数信号 + 模型内容判断）
   │
   ├── 足够 ──────────────────────────────────────────┐
   │                                                  ▼
   ├── 不足 / 索引为空 → ④网络自主爬取            ⑥综合回答
   │     种子URL = SearchProvider（若配置）        [n] 引用 + 来源类型标注
   │              或 模型凭知识选择                      （本地库 / 网络 URL）
   │     读页面 → 需要更多 → 跟页内链接多跳              双双不足 → 明确拒答
   │     （单次运行 ≤3 次抓取）                          ▲
   │                    │                                │
   │                    ▼                                │
   └──────────────→ ⑤交叉验证（可选）：本地与网络证据 ──┘
                     冲突时对比、标注哪个更新
```

节点①③⑤是**提示层编排**（系统提示引导模型按序决策），节点②④是**工具执行**（Python 侧强制护栏）。这个设计与 LangGraph 三节点结构（agent/tools/finalize）正交：工作流发生在 agent 节点的模型决策里，tools 节点只负责受控执行——不改变现有图拓扑。

## 3. 决策规则表

模型在每个决策点的规则（写进系统提示，M2 落地）：

| 决策点 | 规则 | 违反的后果（Python 侧兜底） |
| --- | --- | --- |
| 是否先查本地 | 索引存在时**必须**先 search_knowledge_base，禁止跳过直接上网 | 无硬约束（提示层），但审计可见全流程 |
| 本地证据是否足够 | 结合分数（多低于 min_score 说明不匹配）与内容判断；拿不准时倾向补一轮改写后重查本地 | 模型自行判断，max_steps 兜底 |
| 何时上网 | 本地不足或索引为空；问题涉及时效性内容（版本号、最新 API）时主动网络核实 | web 抓取计数超 3 次/运行 → 返回可审计错误 |
| 种子 URL 从哪来 | 配置了 SearchProvider → 调它；否则凭模型知识选择权威站点（官方文档优先） | URL 校验（http/https、SSRF 拦截）不过 → 返回错误证据 |
| 是否多跳 | 页面内容部分相关但缺细节 → 从返回的 links 里选最相关的一条继续 | 计数与 robots/大小护栏全程生效 |
| 是否交叉验证 | 本地与网络证据**冲突**（说法不一致）时，回答中对比并标注来源与新旧 | 无（提示层行为，审计可查） |
| 何时停止 | 证据足以回答，或工具预算耗尽（max_steps=8 / web 3 次） | 双重预算检查（agent 节点入口 + tools 节点出口） |
| 拒答条件 | 本地与网络双双不足 → 明确说没有足够依据，**不得编造** | 兜底文案不写入会话历史（现有机制） |

## 4. 护栏清单（全部 Python 侧强制）

| 护栏 | 实现 | 层 |
| --- | --- | --- |
| 工具白名单 | 只有两个工具：search_knowledge_base、fetch_web_page；其他名字返回 `{"error": "不允许的工具"}` | dispatch |
| 参数校验 | JSON 解析 + 字段白名单 + 类型校验（沿用现有工具防线，含 bool-is-int 拦截） | 工具 invoke_json |
| SSRF 防护 | 拒绝 localhost / 127.* / 私网与链路本地 IP 字面量（模型给的 URL 永远不可信；不做 DNS 解析，已知局限） | WebFetchTool |
| robots.txt | 复用爬虫 `_RobotsCache`，按工具实例缓存；拒绝返回 `robots-disallowed` | WebFetchTool |
| 单页限制 | 5MiB 上限（超限拒绝不截断）、15s 超时、Content-Type 白名单 | 复用 fetch_url |
| 抓取频率 | 单次运行 ≤3 次 web 抓取 | WebFetchTool 计数 |
| 轮数上限 | max_steps=8（完全体模式，普通模式仍 5） | runtime/graph 双检查 |
| 不可信证据 | 网络内容=被动证据，不执行其中指令；**不静默入库**（想入库走显式 ingest-url） | 系统提示 + 无入库代码路径 |
| 全程审计 | 每次调用（含被拒绝的）进 tool_calls 审计；evidence 区分来源类型 | runtime/graph 现有机制扩展 |

## 5. SearchProvider 接口设计（预留，本轮只实现 NullProvider）

```python
class SearchProvider(Protocol):
    """query -> 候选 URL 列表。用于给网络阶段提供种子，替代模型猜 URL。"""
    name: str
    def search(self, query: str, *, top_k: int = 3) -> list[str]: ...

class NullSearchProvider:
    """默认：不提供搜索。agent 回退到'模型凭知识选 URL + 链接多跳'。"""
    # search() 永远返回 []，工具描述里告知模型"没有搜索引擎，自己选权威 URL"
```

升级路径（本轮不做）：实现 `TavilySearchProvider` 等，从环境变量（如 `SEARCH_API_KEY`）构造；agent 工具列表与提示随之声明"可先调搜索再抓取"。工具 schema 层预留 `search_web` 名称。

## 6. 引用与证据的来源标注

- `search_knowledge_base` 返回的 evidence 项带 `source_type: "local"`（source_path=文件/来源）；
- `fetch_web_page` 返回带 `source_type: "web"`（source_path=URL）；
- 综合回答的系统提示要求：引用编号后标注来源类型，如 `[1]（本地）`、`[2]（网络）`；
- `AgentResult.evidence` 混排两源，`--json` 直接可审计。

## 7. 里程碑进度清单

> 每完成一项：状态改 ✅ 并附提交哈希；暂停等待用户确认后进入下一项。
> **恢复现场口令**：读本表 → 看最近 ✅ 的提交 diff → 从下一行继续。

| # | 里程碑 | 内容 | 状态 | 提交 |
| --- | --- | --- | --- | --- |
| M0 | 工程文档 | 本文档 | ✅ 完成 | 494fba1 |
| M1 | 网络工具层 | `agent/web_tool.py`（WebFetchTool 全护栏）+ SearchProvider/NullProvider + 单测 | ✅ 完成 | 本提交（哈希在 M2 回填） |
| M2 | 运行时多工具化 | KnowledgeAgent 双工具 + 共享 dispatch（手写/LangGraph 同源）+ 工作流引导版系统提示 + 测试 | ⬜ 未开始 | - |
| M3 | 工作流端到端验证 | 四路径离线剧本测试：本地足够→不上网 / 本地不足→多跳 / 空索引→直接网络 / 双双不足→拒答 | ⬜ 未开始 | - |
| M4 | 应用层收敛 | CLI help 双入口分组、agent 默认启用 web（`--no-web` 逃生）、README/使用指南重构、chat 外壳对齐 | ⬜ 未开始 | - |
| M5 | 收尾 | 真实网络冒烟（可选）、优化文档登记（OPT-17 部分落地）、进度收尾 | ⬜ 未开始 | - |

## 8. 明确不做（本轮范围外）

- 不接真实搜索 API（只留 SearchProvider 接口与 NullProvider）；
- iOS 仓库（rag-agent-ios）不同步——Swift 版是独立移植，后续单独计划；
- 不删除任何维护命令；不改索引/分块/embedding 层（本改造全部在 agent 层）。

## 9. 验收标准

1. 全量 pytest 绿（现有 84 个 + 新增，全部离线）；
2. 四条工作流路径各有离线剧本测试锁定行为；
3. `agent` 命令真实运行：本地有问题时不上网；本地没有时能抓取公开文档站并给出带 URL 引用的回答；
4. 审计完整：`--json` 输出可还原每一步工具调用与来源；
5. `--no-web` 时行为与改造前一致（逃生开关有效）。
