# Embedding 与向量索引

## 1. Embedding 在 RAG 中做什么

Embedding 是一个表示转换器：

```text
文本 -> 一组固定长度的数字
```

导入时，把每个 Chunk 转成向量；查询时，把用户问题也转成向量。检索器比较两者的相似度，找到可能回答问题的 Chunk。

这里最重要的不是某个数字本身有什么含义，而是相关文本在同一个向量空间中应该更接近。

## 2. 索引和聊天模型是两套系统

Embedding provider 和聊天 provider 分工不同：

```text
Embedding：负责“找什么”
Chat model：负责“怎么说”
```

DeepSeek 聊天 Key 不自动等于 Embedding 服务。当前项目默认用本地 Embedding，DeepSeek 只在 `ask` 或 `agent` 的生成阶段被调用。

## 3. 当前项目的三个 provider

### `hash`

`HashEmbeddingProvider` 使用稳定哈希把词法特征映射到固定维度。它不下载模型、不联网、容易检查和测试，适合先验证完整 RAG 链路。

它不是理解语义的神经模型。两个意思相近但词面完全不同的句子，可能检索得不好。

### `chinese`

`ChineseNgramEmbeddingProvider` 使用中文字符 1-4 元组、英文/数字 token 和确定性特征哈希，通常比通用 hash 更适合中文短语，但仍是词法基线。

```bash
python -m rag_agent ingest data/inbox \
  --embedding-provider chinese --embedding-dimension 512
```

### `openai`

`OpenAICompatibleEmbeddingProvider` 调用提供 `/embeddings` 的 OpenAI 兼容服务，需要单独配置：

```text
EMBEDDING_API_KEY
EMBEDDING_BASE_URL
EMBEDDING_MODEL
```

不能假设任意聊天服务都提供这个接口。使用前先确认服务文档和返回向量维度。

## 4. 为什么索引要保存 provider 指纹

向量只有在同一空间中才可比较。如果用 provider A 建索引，却用 provider B 生成查询向量，余弦相似度的数字没有可靠意义。

因此索引首行保存：

```text
schema_version
provider_name
provider_model
provider_fingerprint
dimension
chunk_count
```

查询时 `LocalVectorIndex._validate_provider` 会比较指纹和维度，不一致就明确报错，而不是静默返回错误结果。

## 5. 余弦相似度

当前索引使用余弦相似度：

```text
cosine(a, b) = (a · b) / (||a|| ||b||)
```

它关注两个向量的方向，常用于比较文本表示。实现会检查维度、数值是否有限和零向量，避免 NaN 或不一致数据污染排名。

对每个候选 Chunk：

1. 计算查询向量和 Chunk 向量的相似度。
2. 丢弃低于 `min_score` 的结果。
3. 按分数从高到低排序。
4. 以 `chunk_id` 作为稳定的并列排序依据。
5. 返回前 `top_k` 个 `SearchResult`。

## 6. 为什么当前使用 JSONL 索引

本项目是学习和个人资料规模，JSONL 有几个优势：

- 文件格式透明，可以直接查看 meta、Chunk 和向量。
- 不需要额外数据库服务。
- 测试和备份简单。
- 可以把索引实现和上层检索接口隔离。

代价是每次查询需要扫描内存中的候选，规模大后会受内存和线性时间影响。以后可以把 `LocalVectorIndex` 替换成 FAISS、SQLite、Qdrant 等实现，尽量保持 `search` 接口和 `SearchResult` 不变。

## 7. 建索引的过程

`build_vector_index` 的逻辑是：

```text
校验每个 Chunk 有非空 chunk_id 和 text
-> 批量调用 provider.embed
-> 检查返回数量和维度
-> 检查每个向量为有限数字
-> 写入 meta 行
-> 写入 chunk + vector 行
-> 重新加载索引验证格式
```

写入使用原子替换，避免进程中断留下半个索引。

## 8. 增量索引为什么能复用向量

增量导入产生新一批 Chunk 后，`update_vector_index` 会：

1. 尝试加载旧索引。
2. 只有 provider 指纹一致时才考虑复用。
3. 按 `chunk_id` 找旧行。
4. 比较正文、内容哈希和处理指纹。
5. 未变化的向量直接复用，新增或变化的 Chunk 才重新 Embedding。

这既节省远程 Embedding 成本，也避免不同配置下混合向量。

## 9. 如何选择 provider

先用 `hash` 跑通：

```bash
python -m rag_agent ingest data/inbox
python -m rag_agent search "测试问题"
```

中文词面匹配不足时，再试 `chinese`。如果仍然需要更好的语义召回，接入真正的 Embedding 服务，并用评测集比较，而不是只根据个别问题下结论。

对比时保持其他条件不变：同一批资料、同一批问题、同一个 `top_k` 和阈值。否则无法知道提升来自 provider 还是来自其他参数。

## 10. Embedding 层的验收标准

- 导入和查询使用相同 provider/model/dimension。
- 索引 meta 可以说明自己由什么配置生成。
- 维度错误、空向量、非有限值会明确失败。
- 对同一输入得到稳定结果。
- 资料变化时可以只重新计算必要向量。
- 结果仍保留原文和来源元数据，而不是只返回数字。
