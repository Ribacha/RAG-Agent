"""从现有索引离线生成种子题库（候选，标注 auto:true 待人工复核）。

用法：.venv/bin/python scripts/generate_seed_tasks.py [输出路径]

answerable 题的问题与要点为人工拟定，标注（相关 chunk）由离线检索自动
对齐——检索排名第一的 chunk 记为相关标注；refusal 陷阱题为纯手写。
检索不到的题会被跳过并列出，提示换措辞或确认资料覆盖。
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_agent.embeddings import create_embedding_provider
from rag_agent.retrieval.index import LocalVectorIndex
from rag_agent.workspace import find_workspace_root


# (query, category, expected_points)
ANSWERABLE_CANDIDATES = [
    ("TCP 三次握手的过程是什么？", "传输层", ["SYN", "SYN-ACK", "ACK"]),
    ("四次挥手为什么需要四次？", "传输层", ["FIN", "ACK", "半关闭"]),
    ("滑动窗口机制的作用是什么？", "传输层", ["流量控制", "接收窗口"]),
    ("流量控制和拥塞控制有什么区别？", "传输层", ["点对点", "全局/网络"]),
    ("ARP 协议的作用是什么？", "网络层", ["IP 地址", "MAC 地址", "广播"]),
    ("DNS 解析的完整过程是怎样的？", "应用层", ["递归查询", "迭代查询", "缓存"]),
    ("子网掩码的作用是什么？", "网络层", ["网络位", "主机位"]),
    ("IP 数据报分片是怎么回事？", "网络层", ["MTU", "分片", "重组"]),
    ("CRC 差错检测的原理？", "数据链路层", ["循环冗余校验", "帧检验序列"]),
    ("什么是透明传输？", "数据链路层", ["转义", "字节填充"]),
    ("停止等待协议如何保证可靠传输？", "数据链路层", ["确认", "超时重传"]),
    ("TCP 和 UDP 有什么区别？", "传输层", ["面向连接", "可靠性", "面向字节流/报文"]),
    ("UDP 适合什么场景？", "传输层", ["实时性", "开销小"]),
    ("快速重传的触发条件是什么？", "传输层", ["重复确认"]),
    ("慢启动算法的行为？", "传输层", ["拥塞窗口", "指数增长"]),
    ("NAT 的基本原理？", "网络层", ["地址转换", "端口"]),
    ("ping 命令使用了什么协议？", "网络层", ["ICMP"]),
    ("信道复用有哪些方式？", "物理层", ["频分", "时分"]),
    ("数据链路层要解决的三个基本问题？", "数据链路层", ["封装成帧", "透明传输", "差错检测"]),
    ("FTP 的主动模式和被动模式？", "应用层", ["控制连接", "数据连接"]),
    ("局域网的三要素是什么？", "数据链路层", ["拓扑", "传输介质", "介质访问控制"]),
    ("透明网桥如何工作？", "数据链路层", ["自学习", "转发表"]),
]

# 知识库与网络都不该有的内容：期望明确拒答。
REFUSAL_TRAPS = [
    ("量子纠缠通信属于 OSI 模型的哪一层？", "陷阱"),
    ("这本资料里记录了我的银行卡密码吗？", "陷阱"),
    ("书中第三章作者的生日是哪一天？", "陷阱"),
    ("书里推荐的考研政治复习计划是什么？", "陷阱"),
    ("资料中关于火星地球化改造的章节讲了什么？", "陷阱"),
    ("第一章提到的那个餐厅的招牌菜是什么？", "陷阱"),
]


def main() -> int:
    root = find_workspace_root()
    index_path = root / "data/index/vectors.jsonl"
    if not index_path.exists():
        print(f"索引不存在：{index_path}，请先 ingest。", file=sys.stderr)
        return 2
    index = LocalVectorIndex.load(index_path)
    provider = create_embedding_provider("hash", dimension=index.dimension)

    output = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "data/eval/agent_tasks.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped: list[str] = []
    for number, (query, category, points) in enumerate(ANSWERABLE_CANDIDATES, start=1):
        results = index.search(query, provider=provider, top_k=1, min_score=0.0)
        if not results:
            skipped.append(query)
            continue
        top = results[0]
        rows.append(
            {
                "id": f"q{number:03d}",
                "query": query,
                "task_type": "answerable",
                "relevant_chunk_ids": [top.chunk_id],
                "relevant_source_paths": [top.source_path],
                "expected_points": points,
                "category": category,
                "auto": True,
            }
        )
    for number, (query, category) in enumerate(REFUSAL_TRAPS, start=1):
        rows.append(
            {
                "id": f"r{number:03d}",
                "query": query,
                "task_type": "refusal",
                "expected_points": [],
                "category": category,
                "auto": False,
            }
        )

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"已生成 {len(rows)} 题 -> {output}")
    print(f"  answerable（auto 标注）：{len(ANSWERABLE_CANDIDATES) - len(skipped)}")
    print(f"  refusal 陷阱题（手写）：{len(REFUSAL_TRAPS)}")
    if skipped:
        print("检索不到、已跳过（建议换措辞或确认资料覆盖）：")
        for query in skipped:
            print(f"  - {query}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
