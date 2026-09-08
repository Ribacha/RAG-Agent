# Agent 评测 Harness 工程文档

> 给完全体 Agent 造一台**可重复运行的评测机器**：固定题库 → 批量执行 → 确定性指标
> + LLM 裁判 → 归因分类 → 快照对比。方法论移植自 WT-AI 项目的评测闭环（1580 题
> 知识库 + 评测→归因→改→再评循环），七个评测组件全部落地。
>
> 本文档同时是**执行蓝本与进度账本**（文末里程碑清单）；每步一提交一暂停。

---

## 1. 目标与 WT-AI 七组件映射

| WT-AI 评测组件 | 本项目落地 |
| --- | --- |
| 题库（标注任务集） | `data/eval/agent_tasks.jsonl`，含陷阱题（应拒答） |
| 跑批器 | `rag-agent harness` 命令，逐题执行 agent 并收集审计 |
| 确定性指标 | 检索命中 / 引用真实性 / 拒答正确性 / 工具效率（免费、可进 CI） |
| LLM 裁判 | 忠实度 / 覆盖度 / 总分（1-5）+ 失败分类 |
| 裁判校准 | 内置人工基准样本，输出裁判与人工的偏差 |
| 抖动量化 | `--repeat N` 重跑，均值±方差 + 拒答翻转率 |
| 快照对比 | 报告落盘 + `--compare` 前后 diff（对应"修复前/修复后v2"） |

## 2. 架构与数据流

```text
data/eval/agent_tasks.jsonl ──► runner（逐题跑 agent，收集 AgentResult 审计）
                                  │
                    ┌─────────────┼──────────────┐
              确定性指标      LLM 裁判         抖动（--repeat N）
              （免费）      （忠实/覆盖/总分    （均值±方差、拒答翻转率）
              检索命中       + 失败分类）
              引用真实性         │
              拒答正确性     校准（人工基准 → 一致性报告）
              工具效率          │
                    └─────────────┴──────────────┘
                                  ▼
              报告 JSON 落盘 data/eval/reports/harness-<时间戳>.json
              + 终端摘要 + --compare 旧报告 → 前后 diff
```

模块划分（`src/rag_agent/harness/`）：

| 模块 | 职责 |
| --- | --- |
| `tasks.py` | 题库加载、逐行校验（带行号的错误信息）、schema 版本 |
| `metrics.py` | 确定性指标（纯函数，输入 AgentResult.to_dict() 输出指标字典） |
| `judge.py` | LLM 裁判：评分锚点 + 失败分类 + 校准样本；OPT-23 空内容防御 |
| `runner.py` | 编排：跑批 → 指标 → 裁判 → 聚合 → 报告落盘 → 对比 |
| CLI | `rag-agent harness <tasks.jsonl> [选项]` |

## 3. 题库格式（schema v1）

```jsonl
{"id":"q001","query":"TCP 三次握手的过程是什么？",
 "task_type":"answerable",
 "relevant_chunk_ids":["<chunk_id>"],"relevant_source_paths":["<path>"],
 "expected_points":["SYN","SYN-ACK","ACK"],
 "category":"传输层","auto":true}
```

- `task_type`：`answerable`（知识库应有答案）/ `refusal`（陷阱题——知识库和网络
  都不该有，期望明确拒答；WT-AI 拒答归因的原料）；
- `relevant_chunk_ids` / `relevant_source_paths`：沿用 `evaluation.py` 的标注约定，
  命中任一即算 retrieval_hit；`refusal` 题不需要标注；
- `expected_points`：裁判覆盖度评分的依据（该答案应覆盖的要点，2-5 个）；
- `category`：聚合分解维度（如"传输层/网络层/数据链路层"）；
- `auto: true`：由离线工具自动生成的候选题（检索命中的真实 chunk 作标注），
  **待人工复核**——auto 题的结论权重低于人工题，报告分开展示；
- 校验规则：id 唯一非空、query 非空、answerable 题至少一种标注、expected_points
  1-8 条非空字符串。

## 4. 指标定义

### 4.1 确定性指标（每题，零成本）

| 指标 | 定义 | 判定 |
| --- | --- | --- |
| `retrieval_hit` | agent 的 search 工具结果命中任一标注 chunk/来源 | bool |
| `citation_validity` | 回答中每个 `[n]` 编号存在且 ≤ 证据数 | ok / invalid / **uncited**（有实质结论但零引用） |
| `refusal_correct` | answerable 题未拒答 且 refusal 题拒答 | bool（拒答判定：回答等于拒答文案，或证据为空且回答含"没有找到"） |
| `tool_calls` / `web_used` | 工具调用次数 / 是否上网 | 计数（效率观察项，不判对错） |

### 4.2 裁判分（每题，1-5 整数）

| 维度 | 评分锚点 |
| --- | --- |
| `faithfulness` | 5=全部结论有证据支撑且引用真实；3=大体有支撑有个别越界；1=大量编造或与证据矛盾 |
| `coverage` | 5=覆盖全部 expected_points；4=缺一个次要要点；2=只覆盖小半；1=完全没覆盖 |
| `overall` | 综合分（faithfulness 权重高于 coverage——忠实优先） |
| `failure_type` | `正常` / `拒答`（有知识却说没有）/ `跑题` / `事实错误`（与证据矛盾）/ `漏要点` ——沿用 WT-AI 分类法 |

裁判输入：query + 证据摘要（带编号与来源类型）+ 回答全文 + expected_points；
要求输出结构化 JSON。**防御**（OPT-23）：裁判 `max_tokens ≥ 3000`，空 content
报"疑似推理内容耗尽预算，请调大裁判 max_tokens"。

### 4.3 聚合与归因

- 总分与各维度均值、failure_type 分布、按 category 分解、auto/人工分开统计；
- **交叉归因矩阵**（WT-AI 实验4 方法论）：

| | 裁判分低 | 裁判分高 |
| --- | --- | --- |
| 检索命中 | **生成端问题**（改 prompt/模型） | 健康 |
| 检索未命中 | **检索端问题**（改分块/embedding） | 模型自行补全（需人工抽查是否幻觉） |

## 5. 裁判校准（先验证裁判，再信判决）

- 内置 5 条人工基准样本（固定的问题/证据/回答 + 人工判定分数与分类），随包分发；
- `--calibrate`：对基准样本跑裁判，输出逐条偏差与平均偏差；
- **判读规则**：平均偏差 > 1.0 分或分类不一致 > 2 条 → 报告显著警示
  "本次裁判结论可信度低"（对应 WT-AI 的抽检校准样本思路）。

## 6. 抖动与快照

- `--repeat N`（默认 1）：real 模式下整套重跑 N 次，输出各指标均值±标准差、
  每题回答是否翻转（尤其拒答翻转率——WT-AI "重跑成功率"的对应物）；
- 报告落盘 `data/eval/reports/harness-YYYYMMDD-HHMMSS.json`（含 schema 版本、
  模型名、题库指纹、全部逐题明细）；
- `--compare <旧报告>`：输出新旧关键指标 diff（分数、分类占比、翻转的题）。

## 7. 双模式与诚实边界

```bash
rag-agent harness data/eval/agent_tasks.jsonl                  # fake：免费
rag-agent harness data/eval/agent_tasks.jsonl --real           # 真实质量评测
rag-agent harness ... --real --repeat 3                        # 抖动量化
rag-agent harness ... --real --compare <旧报告.json>
```

- **fake 模式**：注入"理想剧本"假模型（严格按工作流执行：先本地检索、按证据作答、
  陷阱题拒答）——**应得满分**。它测的是 harness 自身计算正确 + 工作流机制回归，
  可进 CI、零成本；
- **real 模式**：真实模型质量评测，消费 API Key；两种模式的结论**绝不混用**——
  fake 满分不等于系统好，只等于"尺子没坏"。

## 8. 里程碑进度清单

> 恢复现场口令：读本表 → 看最近 ✅ 的提交 → 从下一行继续。

| # | 里程碑 | 内容 | 状态 | 提交 |
| --- | --- | --- | --- | --- |
| H0 | 工程文档 | 本文档 | ✅ 完成 | 2bc47ef |
| H1 | 题库与指标 | tasks.py（格式+校验）+ 种子题库（自动生成+陷阱题）+ metrics.py + 单测 | ✅ 完成 | 3df2954 |
| H2 | 跑批与报告 | runner.py + 报告落盘 + --compare + CLI 命令 + fake 端到端离线测试 | ✅ 完成 | 0dde537 |
| H3 | 裁判 | judge.py（评分+分类+校准+OPT-23 防御）+ 抖动 + 假裁判离线测试 | ✅ 完成 | 本提交（哈希在 H4 回填） |
| H4 | 真实评测 | 真实跑一轮 + 校准核查 + 快照 + 优化文档登记（OPT-01/02/03/22 部分） | ⬜ | - |

## 9. 范围外

不做并发跑批（礼貌+确定性）；不自动把 bad case 写回题库（人工审后手动加）；
iOS 仓库不同步；不做多裁判交叉（单裁判+校准已满足当前规模，留作后续）。

## 10. 验收标准

1. 全量 pytest 绿（新增测试全部离线）；
2. fake 模式满分（尺子自证）；
3. real 模式产出完整报告：总分 / 分类占比 / 交叉归因矩阵 / auto 与人工分开统计；
4. 校准偏差在报告内可见且带判读规则；
5. `--repeat` 给出方差与翻转率；题库与报告带 schema 版本，可回归。
