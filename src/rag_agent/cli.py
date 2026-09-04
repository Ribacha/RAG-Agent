"""RAG Agent 命令行入口。

命令按数据流拆开：``ingest`` 负责抽取/分块并构建索引，``search`` 只查询
已有索引，``rebuild-index`` 允许在更换 embedding 配置后显式重建。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .agent import ConversationHistory, KnowledgeSearchTool, run_knowledge_graph
from .agent.runtime import KnowledgeAgent
from .answering import OpenAICompatibleChatProvider, RagAnswerer
from .chunking.splitter import ChunkConfig, chunk_document
from .embeddings import create_embedding_provider
from .evaluation import evaluate, load_evaluation_samples
from .ingest.pdf import PdfOptions
from .ingest.incremental import incremental_ingest
from .ingest.pipeline import ingest_path
from .retrieval.index import LocalVectorIndex, build_vector_index, update_vector_index
from .storage.jsonl import read_jsonl, write_jsonl_atomic


def _project_root() -> Path:
    """找到默认数据目录；可用 RAG_AGENT_ROOT 覆盖。"""

    configured = os.getenv("RAG_AGENT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
            return candidate
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").exists() and (source_root / "src").is_dir():
        return source_root
    return current


PROJECT_ROOT = _project_root()


def _load_local_env() -> None:
    """Load ``.env`` from the project root without overriding shell values.

    ``python-dotenv`` is intentionally optional at import time: offline CLI
    commands still work when only explicit shell variables are available.
    """

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(PROJECT_ROOT / ".env", override=False)


_load_local_env()

DEFAULT_CHUNKS = PROJECT_ROOT / "data/index/chunks.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "data/index/documents.jsonl"
DEFAULT_FAILURES = PROJECT_ROOT / "data/failed/ingestion.jsonl"
DEFAULT_INDEX = PROJECT_ROOT / "data/index/vectors.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-agent",
        description="分阶段 RAG Agent：导入、检索、离线评测和受控问答。",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser(
        "ingest",
        help="导入一个文件或目录，并生成带来源元数据的 JSONL chunks",
    )
    ingest.add_argument("input", type=Path, help="PDF、Markdown、TXT 文件或目录")
    ingest.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CHUNKS,
        help=f"chunks 输出路径（默认：{DEFAULT_CHUNKS}）",
    )
    ingest.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"文档清单路径（默认：{DEFAULT_MANIFEST}）",
    )
    ingest.add_argument(
        "--failures",
        type=Path,
        default=DEFAULT_FAILURES,
        help=f"失败文件清单（默认：{DEFAULT_FAILURES}）",
    )
    ingest.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help=f"向量索引路径（默认：{DEFAULT_INDEX}）",
    )
    _add_embedding_arguments(ingest)
    ingest.add_argument(
        "--no-index",
        action="store_true",
        help="只写 chunks，不构建向量索引",
    )
    ingest.add_argument(
        "--incremental",
        action="store_true",
        help="在已有 chunks/manifest/index 上增量更新，只处理新增或变化来源",
    )
    ingest.add_argument("--max-chars", type=int, default=1200)
    ingest.add_argument("--overlap-chars", type=int, default=120)
    ingest.add_argument(
        "--ocr",
        action="store_true",
        help="对文本不足的 PDF 页面启用 Tesseract OCR",
    )
    ingest.add_argument("--ocr-language", default="chi_sim+eng")
    ingest.add_argument("--ocr-dpi", type=int, default=300)
    ingest.add_argument("--min-native-chars", type=int, default=40)
    ingest.add_argument("--max-pages", type=int, default=500)
    ingest.add_argument(
        "--max-file-mb",
        type=float,
        default=50,
        help="单文件大小上限（MiB，默认 50）",
    )
    ingest.set_defaults(handler=_handle_ingest)

    search = commands.add_parser(
        "search",
        help="在已有索引中检索最相关的知识片段",
    )
    search.add_argument("query", help="自然语言查询")
    search.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help=f"向量索引路径（默认：{DEFAULT_INDEX}）",
    )
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--min-score", type=float, default=0.0)
    search.add_argument("--source", type=str, default=None, help="只搜索指定来源路径")
    search.add_argument("--file-type", type=str, choices=["txt", "markdown", "pdf"])
    search.add_argument("--json", action="store_true", dest="as_json")
    _add_embedding_arguments(search)
    search.set_defaults(handler=_handle_search)

    evaluate_command = commands.add_parser(
        "evaluate",
        help="用离线 JSONL 标注集评测 Top-K 检索质量",
    )
    evaluate_command.add_argument(
        "evaluation_file",
        type=Path,
        help="评测集 JSONL（每行含 query 和相关 chunk/source 标注）",
    )
    evaluate_command.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help=f"向量索引路径（默认：{DEFAULT_INDEX}）",
    )
    evaluate_command.add_argument("--top-k", type=int, default=5)
    evaluate_command.add_argument("--min-score", type=float, default=0.0)
    evaluate_command.add_argument("--json", action="store_true", dest="as_json")
    _add_embedding_arguments(evaluate_command)
    evaluate_command.set_defaults(handler=_handle_evaluate)

    ask = commands.add_parser(
        "ask",
        help="检索知识库并调用聊天模型生成带引用的回答",
    )
    ask.add_argument("question", help="要询问知识库的问题")
    ask.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help=f"向量索引路径（默认：{DEFAULT_INDEX}）",
    )
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument(
        "--min-score",
        type=float,
        default=0.08,
        help="最低余弦相似度；hash 基线可先从 0.08 调整",
    )
    ask.add_argument("--max-context-chars", type=int, default=8000)
    ask.add_argument("--json", action="store_true", dest="as_json")
    ask.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将发送给模型的证据，不调用聊天 API",
    )
    _add_embedding_arguments(ask)
    ask.add_argument(
        "--llm-api-key",
        "--chat-api-key",
        dest="llm_api_key",
        default=None,
        help="聊天模型 Key（也可设置 LLM_API_KEY）",
    )
    ask.add_argument(
        "--llm-base-url",
        "--chat-base-url",
        dest="llm_base_url",
        default=None,
        help="聊天接口 Base URL（默认读取 LLM_BASE_URL）",
    )
    ask.add_argument(
        "--llm-model",
        "--chat-model",
        dest="llm_model",
        default=None,
        help="聊天模型名称（默认读取 LLM_MODEL 或 deepseek-chat）",
    )
    ask.add_argument("--temperature", type=float, default=0.2)
    ask.add_argument("--max-tokens", type=int, default=1200)
    ask.set_defaults(handler=_handle_ask)

    agent = commands.add_parser(
        "agent",
        help="让聊天模型通过受控检索工具自主完成知识库问答",
    )
    agent.add_argument("question", help="要询问知识库的问题")
    agent.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help=f"向量索引路径（默认：{DEFAULT_INDEX}）",
    )
    agent.add_argument("--embedding-provider", choices=["hash", "chinese", "openai"], default=None)
    agent.add_argument("--embedding-dimension", type=int, default=None)
    agent.add_argument("--embedding-model", default=None)
    agent.add_argument("--embedding-api-key", default=None)
    agent.add_argument("--embedding-base-url", default=None)
    agent.add_argument("--llm-api-key", "--chat-api-key", dest="llm_api_key", default=None)
    agent.add_argument("--llm-base-url", "--chat-base-url", dest="llm_base_url", default=None)
    agent.add_argument("--llm-model", "--chat-model", dest="llm_model", default=None)
    agent.add_argument("--temperature", type=float, default=0.2)
    agent.add_argument("--max-tokens", type=int, default=1200)
    agent.add_argument("--max-steps", type=int, default=5)
    agent.add_argument(
        "--graph",
        action="store_true",
        help="使用可选的 LangGraph 状态图运行（未安装时明确报错）",
    )
    agent.add_argument(
        "--history",
        type=Path,
        default=None,
        help="读取并在本轮完成后更新 JSONL 对话历史",
    )
    agent.add_argument(
        "--save-history",
        type=Path,
        default=None,
        help="把更新后的对话历史写入指定 JSONL 路径（可不读取旧历史）",
    )
    agent.add_argument(
        "--history-max-turns",
        type=int,
        default=20,
        help="新建历史时最多保留的问答轮数（默认 20）",
    )
    agent.add_argument("--json", action="store_true", dest="as_json")
    agent.set_defaults(handler=_handle_agent)

    list_documents = commands.add_parser(
        "list-documents",
        help="查看最近一次导入生成的文档清单",
    )
    list_documents.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"文档清单路径（默认：{DEFAULT_MANIFEST}）",
    )
    list_documents.add_argument("--json", action="store_true", dest="as_json")
    list_documents.set_defaults(handler=_handle_list_documents)

    rebuild = commands.add_parser(
        "rebuild-index",
        help="从 chunks JSONL 重新生成向量索引",
    )
    rebuild.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS,
        help=f"chunks 路径（默认：{DEFAULT_CHUNKS}）",
    )
    rebuild.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help=f"向量索引路径（默认：{DEFAULT_INDEX}）",
    )
    _add_embedding_arguments(rebuild)
    rebuild.set_defaults(handler=_handle_rebuild_index)
    return parser


def _add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-provider",
        choices=["hash", "chinese", "openai"],
        default=None,
        help="embedding 提供方（默认 hash；也可由 EMBEDDING_PROVIDER 设置）",
    )
    parser.add_argument("--embedding-dimension", type=int, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--embedding-api-key", default=None)
    parser.add_argument("--embedding-base-url", default=None)


def _handle_ingest(args: argparse.Namespace) -> int:
    config = ChunkConfig(
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    pdf_options = PdfOptions(
        use_ocr=args.ocr,
        ocr_language=args.ocr_language,
        ocr_dpi=args.ocr_dpi,
        min_native_chars=args.min_native_chars,
        max_pages=args.max_pages,
    )
    output_path = _resolve_path(args.output)
    manifest_path = _resolve_path(args.manifest)
    failures_path = _resolve_path(args.failures)
    index_path = _resolve_path(args.index)

    if args.incremental:
        incremental = incremental_ingest(
            args.input,
            existing_chunks=_read_optional_jsonl(output_path),
            existing_manifests=_read_optional_jsonl(manifest_path),
            chunk_config=config,
            pdf_options=pdf_options,
            max_file_bytes=max(1, int(args.max_file_mb * 1024 * 1024)),
        )
        chunks = list(incremental.chunks)
        manifests = list(incremental.manifests)
        failures = incremental.failures
        documents_succeeded = (
            incremental.documents_added
            + incremental.documents_updated
            + incremental.documents_unchanged
        )
    else:
        batch = ingest_path(
            args.input,
            pdf_options=pdf_options,
            max_file_bytes=max(1, int(args.max_file_mb * 1024 * 1024)),
        )
        chunks = []
        manifests = []
        for document in batch.records:
            document_chunks = chunk_document(document, config=config)
            chunks.extend(chunk.to_dict() for chunk in document_chunks)
            manifests.append(document.to_manifest_dict(len(document_chunks)))
        failures = batch.failures
        documents_succeeded = len(batch.records)

    write_jsonl_atomic(output_path, chunks)
    write_jsonl_atomic(manifest_path, manifests)
    write_jsonl_atomic(failures_path, (failure.to_dict() for failure in failures))

    index_provider = None
    index_stats = None
    if not args.no_index:
        index_provider = create_embedding_provider(
            args.embedding_provider,
            dimension=args.embedding_dimension,
            model=args.embedding_model,
            api_key=args.embedding_api_key,
            base_url=args.embedding_base_url,
        )
        if args.incremental:
            _, index_stats = update_vector_index(
                chunks,
                provider=index_provider,
                path=index_path,
            )
        else:
            build_vector_index(chunks, provider=index_provider, path=index_path)

    summary = {
        "documents_succeeded": documents_succeeded,
        "documents_failed": len(failures),
        "chunks_written": len(chunks),
        "chunks_path": str(output_path),
        "manifest_path": str(manifest_path),
        "failures_path": str(failures_path),
        "index_path": str(index_path) if not args.no_index else None,
        "embedding_provider": index_provider.fingerprint if index_provider else None,
    }
    if args.incremental:
        summary.update(
            {
                "incremental": True,
                "documents_added": incremental.documents_added,
                "documents_updated": incremental.documents_updated,
                "documents_unchanged": incremental.documents_unchanged,
                "documents_deleted": incremental.documents_deleted,
                "chunks_reused": incremental.chunks_reused,
                "chunks_generated": incremental.chunks_generated,
                "vectors_reused": index_stats.reused_vectors if index_stats else None,
                "vectors_embedded": index_stats.embedded_vectors if index_stats else None,
            }
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if failures else 0


def _handle_search(args: argparse.Namespace) -> int:
    index = LocalVectorIndex.load(_resolve_path(args.index))
    provider = create_embedding_provider(
        args.embedding_provider,
        dimension=args.embedding_dimension or index.dimension,
        model=args.embedding_model,
        api_key=args.embedding_api_key,
        base_url=args.embedding_base_url,
    )
    results = index.search(
        args.query,
        provider=provider,
        top_k=args.top_k,
        min_score=args.min_score,
        source_path=args.source,
        file_type=args.file_type,
    )
    if args.as_json:
        print(json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2))
    else:
        _print_search_results(results)
    return 0


def _handle_evaluate(args: argparse.Namespace) -> int:
    """Run retrieval-only evaluation and print aggregate diagnostics."""

    index = LocalVectorIndex.load(_resolve_path(args.index))
    provider = create_embedding_provider(
        args.embedding_provider,
        dimension=args.embedding_dimension or index.dimension,
        model=args.embedding_model,
        api_key=args.embedding_api_key,
        base_url=args.embedding_base_url,
    )
    samples = load_evaluation_samples(_resolve_path(args.evaluation_file))
    report = evaluate(
        index,
        samples,
        provider=provider,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    if args.as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"样本数：{report.sample_count}")
        print(f"Recall@{report.top_k}：{report.recall_at_k:.4f}")
        print(f"引用正确率代理（retrieval citation proxy）：{report.citation_accuracy:.4f}")
        print("\n逐条结果：")
        for sample in report.samples:
            print(
                f"- {sample.name}: recall={sample.recall_at_k:.0f}, "
                f"citation={sample.citation_accuracy:.4f}, "
                f"retrieved={len(sample.retrieved)}"
            )
    return 0


def _handle_ask(args: argparse.Namespace) -> int:
    index = LocalVectorIndex.load(_resolve_path(args.index))
    embedding_provider = create_embedding_provider(
        args.embedding_provider,
        dimension=args.embedding_dimension or index.dimension,
        model=args.embedding_model,
        api_key=args.embedding_api_key,
        base_url=args.embedding_base_url,
    )
    chat_provider = None
    if not args.dry_run:
        chat_provider = OpenAICompatibleChatProvider.from_environment(
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            model=args.llm_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    answerer = RagAnswerer(
        index,
        embedding_provider=embedding_provider,
        chat_provider=chat_provider,
        min_score=args.min_score,
        max_context_chars=args.max_context_chars,
        top_k=args.top_k,
    )
    result = answerer.answer(args.question)
    if args.as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    elif args.dry_run:
        print("将发送给聊天模型的证据：")
        print(result.answer)
    else:
        _print_answer(result)
    return 0


def _handle_agent(args: argparse.Namespace) -> int:
    index = LocalVectorIndex.load(_resolve_path(args.index))
    embedding_provider = create_embedding_provider(
        args.embedding_provider,
        dimension=args.embedding_dimension or index.dimension,
        model=args.embedding_model,
        api_key=args.embedding_api_key,
        base_url=args.embedding_base_url,
    )
    chat_provider = OpenAICompatibleChatProvider.from_environment(
        api_key=args.llm_api_key,
        base_url=args.llm_base_url,
        model=args.llm_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    agent = KnowledgeAgent(
        KnowledgeSearchTool(index, embedding_provider),
        chat_provider,
        max_steps=args.max_steps,
    )
    history = ConversationHistory(max_turns=args.history_max_turns)
    if args.history is not None:
        history = ConversationHistory.load(
            _resolve_path(args.history),
            max_turns=args.history_max_turns,
        )
    if args.graph:
        result = run_knowledge_graph(agent, args.question, history=history)
    else:
        result = agent.run(args.question, history=history)
    history_output = args.save_history or args.history
    if history_output is not None:
        history_path = _resolve_path(history_output)
        result.history.save(history_path)
    if args.as_json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.answer)
        if result.evidence:
            print(f"\nAgent 检索到 {len(result.evidence)} 条证据。")
            for number, evidence in enumerate(result.evidence, start=1):
                location = str(evidence.get("source_path", ""))
                if evidence.get("page_start") is not None:
                    location += f"，第 {evidence['page_start']} 页"
                if evidence.get("heading_path"):
                    location += "，章节：" + " / ".join(evidence["heading_path"])
                print(f"[{number}] {location} | chunk_id={evidence.get('chunk_id', '')}")
        if history_output is not None:
            print(f"\n对话历史已写入：{_resolve_path(history_output)}")
    return 0 if result.stopped_reason == "completed" else 2


def _handle_list_documents(args: argparse.Namespace) -> int:
    rows = list(read_jsonl(_resolve_path(args.manifest)))
    if args.as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print("当前没有已导入文档。")
        return 0
    for row in rows:
        warnings = ", ".join(row.get("warnings", [])) or "无"
        print(
            f"{row.get('source_path', '')} | {row.get('file_type', '')} | "
            f"chunks={row.get('chunk_count', 0)} | warnings={warnings}"
        )
    return 0


def _handle_rebuild_index(args: argparse.Namespace) -> int:
    chunks = [row for row in read_jsonl(_resolve_path(args.chunks)) if row.get("text")]
    provider = create_embedding_provider(
        args.embedding_provider,
        dimension=args.embedding_dimension,
        model=args.embedding_model,
        api_key=args.embedding_api_key,
        base_url=args.embedding_base_url,
    )
    index = build_vector_index(
        chunks,
        provider=provider,
        path=_resolve_path(args.index),
    )
    print(
        json.dumps(
            {
                "chunks_indexed": index.size,
                "index_path": str(_resolve_path(args.index)),
                "embedding_provider": index.provider_fingerprint,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _print_answer(result: object) -> None:
    print(result.answer)
    if result.citations:
        print("\n来源：")
        result_numbers = {
            citation.chunk_id: number
            for number, citation in enumerate(result.results, start=1)
        }
        for citation in result.citations:
            number = result_numbers.get(citation.chunk_id, "?")
            location = citation.source_path
            if citation.page_start is not None:
                location += f"，第 {citation.page_start} 页"
            if citation.heading_path:
                location += "，章节：" + " / ".join(citation.heading_path)
            print(f"[{number}] {location} | chunk_id={citation.chunk_id}")


def _resolve_path(path: Path) -> Path:
    """相对默认路径以项目根为基准，显式绝对路径保持不变。"""

    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _read_optional_jsonl(path: Path) -> list[dict[str, object]]:
    """读取增量快照；第一次运行时不存在的文件视为空快照。"""

    if not path.exists():
        return []
    return list(read_jsonl(path))


def _print_search_results(results: list[object]) -> None:
    if not results:
        print("没有找到满足阈值的知识片段。")
        return
    for number, result in enumerate(results, start=1):
        # SearchResult exposes these fields; keeping formatting here avoids coupling
        # the CLI to the index's internal storage rows.
        page = ""
        if result.page_start is not None:
            page = f" 第 {result.page_start} 页"
        heading = " / ".join(result.heading_path)
        if heading:
            heading = f" | {heading}"
        print(
            f"[{number}] score={result.score:.4f} | {result.source_path}{page}{heading}"
        )
        print(result.text)
        print()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("已取消。", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2
