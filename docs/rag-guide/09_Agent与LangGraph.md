# Agent 与 LangGraph

## 1. 固定 RAG 和 Agent RAG

固定 RAG：

```text
问题 -> Python 检索 -> Prompt -> 模型回答
```

优点是确定、容易测试、费用可控。

Agent RAG：

```text
问题 -> 模型决定是否检索
     -> 调用受控工具
     -> Python 执行检索
     -> 工具结果回到模型
     -> 模型回答或继续检索
```

Agent 增加的是“何时调用工具、调用几次、如何组合结果”的决策，不是新的知识库格式。

## 2. 当前唯一工具

`src/rag_agent/agent/knowledge_tool.py` 声明：

```text
search_knowledge_base(
    query: string,
    top_k: integer 1..20,
    min_score: number -1..1,
)
```

模型返回工具调用后，Python 不直接相信参数，而是：

1. 解析 JSON。
2. 拒绝未知字段。
3. 校验 query、top_k、min_score。
4. 调用既有 `LocalVectorIndex.search`。
5. 将 JSON-safe 结果返回给模型。

这就是“模型提出请求，应用决定能不能执行”。

## 3. 手写 Agent runtime

`src/rag_agent/agent/runtime.py` 维护一个有界循环。每步大致是：

```text
读取历史
-> 调用 complete_with_tools
-> 没有工具调用：收口并保存答案
-> 有工具调用：逐个执行白名单工具
-> 记录调用、结果和证据
-> 达到 max_steps：停止且不把中止答案写入历史
```

`max_steps` 防止模型反复搜索：

```bash
python -m rag_agent agent "问题" --max-steps 3 --json
```

Agent state 保存问题、消息、调用、证据、步数和停止原因，方便审计。

## 4. 为什么先实现手写循环

手写循环能先把业务契约说清楚：

- 工具输入输出是什么。
- 工具失败如何传回。
- 历史何时保存。
- 达到步数上限如何结束。
- 哪些状态需要审计。

如果一开始只依赖图框架，容易把框架节点当成业务设计，出现“图能跑但边界不清”的问题。

## 5. LangGraph 在这里解决什么

LangGraph 把循环显式表示为节点和路由：

```text
agent -> tools -> agent
  |                 |
  v                 v
finalize <----------
```

当前 `src/rag_agent/agent/graph.py` 固定三个节点：

- `agent`：调用聊天模型，决定回答还是工具调用。
- `tools`：执行校验后的搜索工具，追加工具消息和证据。
- `finalize`：生成最终 AgentResult，并只在完成时更新对话历史。

路由函数决定：

- 模型有工具调用时去 `tools`。
- 没有工具调用时去 `finalize`。
- 达到 `max_steps` 或出现停止原因时去 `finalize`。

## 6. LangGraph 是可选依赖

基础 CLI 不应该因为没有 LangGraph 就不能导入资料或运行普通 `ask`。项目只在构建图时惰性导入：

```bash
python -m pip install -e ".[graph]"
python -m rag_agent agent "问题" --graph --json
```

未安装时，`--graph` 明确报错；普通 `ingest`、`search`、`evaluate`、`ask` 和手写 `agent` 不受影响。

## 7. GraphState 为什么是普通字典

图状态包含：

```text
question
messages
pending_tool_calls
tool_calls
evidence
step
answer
stopped_reason
history
```

使用普通可序列化字段有利于：

- LangGraph 检查点。
- JSON 审计输出。
- 单元测试节点。
- 在手写 runtime 和图 runtime 之间转换。

`agent_result_from_graph_state` 将图状态转换回现有 `AgentResult`，下游不需要维护两套输出格式。

## 8. Agent 何时比 `ask` 合适

适合：

- 问题可能需要多次不同查询。
- 需要模型根据第一次证据决定下一步。
- 未来会增加多个只读工具。
- 需要显式审计每个节点和调用。

不适合：

- 只是一次简单检索。
- 需要严格稳定的延迟和费用。
- 还没有检索评测基线。
- 工具权限边界还没有定义。

推荐顺序是先调通 `ask`，再调通手写 `agent`，最后用 `--graph` 替换编排方式。

## 9. 不要把 LangGraph 当成 RAG 本身

LangGraph 不能替代：

- 文件解析。
- 分块。
- Embedding。
- 向量索引。
- 相似度检索。
- 证据边界。
- 生成质量评测。

它主要解决的是有状态、多节点、可路由的执行流程。RAG 的知识质量仍由前面的数据和检索层决定。

## 10. Agent 层验收标准

- 工具列表是白名单。
- 参数在 Python 侧严格校验。
- 调用次数有上限。
- 工具结果和证据可审计。
- 中止运行不会污染对话历史。
- 无 LangGraph 时基础功能仍可用。
- 手写 runtime 和图 runtime 输出契约一致。
