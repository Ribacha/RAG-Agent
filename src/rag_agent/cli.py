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
from .webfetch import (
    CrawlConfig,
    build_web_document,
    crawl_site,
    merge_web_documents,
)
from . import __version__
from .workspace import (
    find_workspace_root,
    is_repo_root,
    is_workspace_root,
    paths_for,
    create_workspace,
    render_env_template,
)


def _project_root() -> Path:
    """找到默认数据目录；可用 RAG_AGENT_ROOT 覆盖。"""

    return find_workspace_root()


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
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser(
        "init",
        help="初始化工作区：创建 data/ 目录结构并生成 .env 配置",
        description=(
            "在指定目录创建 rag-agent 工作区（data/inbox、data/index、"
            "data/failed、data/state 和 .rag-agent/ 标记），并生成 .env。"
            "初始化后可在该工作区内运行其余命令。"
        ),
    )
    init.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=None,
        help="工作区目录（默认当前目录）",
    )
    init.add_argument(
        "--api-key",
        default=None,
        help="聊天模型 Key，写入 .env 的 LLM_API_KEY",
    )
    init.add_argument("--base-url", default="https://api.deepseek.com")
    init.add_argument("--model", default="deepseek-chat")
    init.add_argument(
        "--embedding-provider",
        choices=["hash", "chinese", "openai"],
        default="hash",
    )
    init.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的 .env（默认保留）",
    )
    init.add_argument(
        "--no-input",
        action="store_true",
        help="不进行交互提问，适合脚本和 CI",
    )
    init.set_defaults(handler=_handle_init)

    doctor = commands.add_parser(
        "doctor",
        help="检查工作区、配置和可选依赖是否就绪",
    )
    doctor.add_argument("--json", action="store_true", dest="as_json")
    doctor.set_defaults(handler=_handle_doctor)

    chat = commands.add_parser(
        "chat",
        help="进入交互式问答会话，支持连续追问和会话历史",
        description=(
            "交互式命令行会话：每行一个问题，`exit`/`quit` 退出，"
            "`/reset` 清空历史，`/search 问题` 只检索不回答。"
            "默认使用单轮 RAG；--agent 切换为工具调用模式，"
            "--retrieval-only 完全离线。"
        ),
    )
    chat.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help=f"向量索引路径（默认：{DEFAULT_INDEX}）",
    )
    _add_embedding_arguments(chat)
    chat.add_argument("--top-k", type=int, default=5)
    chat.add_argument(
        "--min-score",
        type=float,
        default=0.08,
        help="最低余弦相似度；hash 基线可先从 0.08 调整",
    )
    chat.add_argument("--max-context-chars", type=int, default=8000)
    chat.add_argument(
        "--retrieval-only",
        action="store_true",
        help="只检索并展示证据，不调用聊天模型（离线可用）",
    )
    chat.add_argument(
        "--agent",
        action="store_true",
        help="使用 agent 工具调用模式回答（默认为单轮 RAG）",
    )
    chat.add_argument("--max-steps", type=int, default=5)
    chat.add_argument(
        "--history",
        type=Path,
        default=None,
        help="会话历史 JSONL 路径；存在则载入，每轮和退出时更新",
    )
    chat.add_argument(
        "--history-max-turns",
        type=int,
        default=20,
        help="最多保留的问答轮数（默认 20）",
    )
    chat.add_argument(
        "--llm-api-key",
        "--chat-api-key",
        dest="llm_api_key",
        default=None,
    )
    chat.add_argument(
        "--llm-base-url",
        "--chat-base-url",
        dest="llm_base_url",
        default=None,
    )
    chat.add_argument(
        "--llm-model",
        "--chat-model",
        dest="llm_model",
        default=None,
    )
    chat.add_argument("--temperature", type=float, default=0.2)
    chat.add_argument("--max-tokens", type=int, default=1200)
    chat.set_defaults(handler=_handle_chat)

    version_command = commands.add_parser(
        "version",
        help="打印 rag-agent 版本",
    )
    version_command.set_defaults(handler=_handle_version)

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

    ingest_url = commands.add_parser(
        "ingest-url",
        help="爬取一个网站的文字内容，清洗整理后导入知识库索引",
        description=(
            "从入口 URL 出发抓取网页（默认同域、尊重 robots.txt、请求间隔 1 秒），"
            "抽取正文并按标题层级整理成知识块，合并进现有索引。已导入且内容未变"
            "的页面会复用旧向量，重复执行只对新增或变化的页面消耗 embedding。"
        ),
    )
    ingest_url.add_argument("url", help="入口 URL（http/https）")
    ingest_url.add_argument(
        "--output", type=Path, default=DEFAULT_CHUNKS, help=f"chunks 输出路径（默认：{DEFAULT_CHUNKS}）"
    )
    ingest_url.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST, help=f"文档清单路径（默认：{DEFAULT_MANIFEST}）"
    )
    ingest_url.add_argument(
        "--failures", type=Path, default=DEFAULT_FAILURES, help=f"失败清单路径（默认：{DEFAULT_FAILURES}）"
    )
    ingest_url.add_argument(
        "--index", type=Path, default=DEFAULT_INDEX, help=f"向量索引路径（默认：{DEFAULT_INDEX}）"
    )
    _add_embedding_arguments(ingest_url)
    ingest_url.add_argument("--max-pages", type=int, default=10, help="最多抓取的页面数（默认 10）")
    ingest_url.add_argument(
        "--max-depth", type=int, default=1, help="从入口页面算起的最大链接深度（默认 1）"
    )
    ingest_url.add_argument(
        "--delay", type=float, default=1.0, help="相邻请求的间隔秒数（默认 1.0，礼貌抓取）"
    )
    ingest_url.add_argument(
        "--cross-domain", action="store_true", help="允许跟随跨域链接（默认只抓同域）"
    )
    ingest_url.add_argument(
        "--no-robots", action="store_true", help="忽略 robots.txt（仅用于抓取你自己控制的站点）"
    )
    ingest_url.add_argument("--timeout", type=float, default=15.0, help="单页请求超时秒数")
    ingest_url.add_argument(
        "--max-page-mb", type=float, default=5, help="单页大小上限（MiB，默认 5）"
    )
    ingest_url.add_argument("--max-chars", type=int, default=1200)
    ingest_url.add_argument("--overlap-chars", type=int, default=120)
    ingest_url.add_argument(
        "--no-index", action="store_true", help="只写 chunks 和清单，不更新向量索引"
    )
    ingest_url.set_defaults(handler=_handle_ingest_url)

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
    search.add_argument("--file-type", type=str, choices=["txt", "markdown", "pdf", "html"])
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


def _handle_version(_args: argparse.Namespace) -> int:
    print(f"rag-agent {__version__}")
    return 0


def _handle_init(args: argparse.Namespace) -> int:
    root = args.path if args.path is not None else Path.cwd()
    paths = create_workspace(root)

    api_key = args.api_key
    if api_key is None and not args.no_input and sys.stdin.isatty():
        # 密钥是敏感输入，用 getpass 隐藏回显；留空表示稍后手动填写。
        import getpass

        try:
            api_key = getpass.getpass(
                "DeepSeek API Key（回车跳过，稍后可编辑 .env）："
            ).strip() or None
        except (EOFError, KeyboardInterrupt):
            print("\n已取消交互输入。")
            api_key = None

    env_written = False
    env_existed = paths.env_file.exists()
    if not env_existed or args.force:
        paths.env_file.write_text(
            render_env_template(
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                embedding_provider=args.embedding_provider,
            ),
            encoding="utf-8",
        )
        env_written = True
        os.chmod(paths.env_file, 0o600)

    summary = {
        "workspace_root": str(paths.root),
        "inbox": str(paths.inbox),
        "index": str(paths.index),
        "env_file": str(paths.env_file),
        "env_written": env_written,
        "env_kept_existing": env_existed and not args.force,
        "api_key_configured": bool(api_key),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if env_existed and not args.force:
        print(".env 已存在且未改动；如需重新生成请加 --force。")
    if not api_key and not _env_has_api_key(paths.env_file):
        print("尚未配置 LLM_API_KEY：ask/agent/chat 需要它，编辑 .env 或运行 "
              "rag-agent init --force --api-key sk-... 重新写入。")
    print("下一步：把资料放入 data/inbox/，然后运行 rag-agent ingest data/inbox。")
    return 0


def _env_has_api_key(env_file: Path) -> bool:
    if not env_file.exists():
        return False
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "LLM_API_KEY" and value.strip() and not value.strip().startswith("sk-你的"):
            return True
    return False


def _handle_doctor(args: argparse.Namespace) -> int:
    """检查本地环境并输出诊断清单；有 error 项时退出码为 1。"""

    import importlib.util
    import shutil

    checks: list[dict[str, object]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    root = find_workspace_root()
    paths = paths_for(root)
    if is_workspace_root(root):
        add("工作区", "ok", str(root))
    elif is_repo_root(root):
        add("工作区", "ok", f"{root}（仓库检出，可用 init 添加 .rag-agent 标记）")
    else:
        add("工作区", "warn", f"{root}（未找到工作区标记，运行 rag-agent init 创建）")

    missing_dirs = [
        str(directory)
        for directory in (paths.inbox, paths.chunks.parent, paths.failures.parent)
        if not directory.exists()
    ]
    if missing_dirs:
        add("数据目录", "warn", "缺少：" + "、".join(missing_dirs))
    else:
        add("数据目录", "ok", "data/ 目录结构完整")

    if _env_has_api_key(paths.env_file) or os.getenv("LLM_API_KEY"):
        add("聊天模型 Key", "ok", "已配置 LLM_API_KEY")
    else:
        add("聊天模型 Key", "warn", "未配置；ask/agent/chat 需要它，检索类命令可离线运行")

    version_ok = sys.version_info >= (3, 11)
    add(
        "Python 版本",
        "ok" if version_ok else "error",
        f"{sys.version_info.major}.{sys.version_info.minor}"
        + ("" if version_ok else "（需要 3.11+）"),
    )

    optional_modules = [
        ("dotenv", "python-dotenv（.env 读取）", "error"),
        ("fitz", "PyMuPDF（PDF 抽取）", "warn"),
        ("openai", "OpenAI SDK（ask/agent/chat）", "warn"),
        ("bs4", "BeautifulSoup（网页抓取 ingest-url）", "warn"),
        ("pytesseract", "pytesseract（OCR）", "warn"),
        ("langgraph", "LangGraph（agent --graph）", "warn"),
    ]
    for module_name, label, missing_status in optional_modules:
        found = importlib.util.find_spec(module_name) is not None
        add(
            f"依赖 {module_name}",
            "ok" if found else missing_status,
            label + ("" if found else "：未安装，相关功能不可用"),
        )
    tesseract = shutil.which("tesseract")
    add(
        "Tesseract 可执行文件",
        "ok" if tesseract else "warn",
        tesseract or "未找到；OCR 需要 brew install tesseract tesseract-lang",
    )

    summary = _index_meta_summary(paths.index)
    if summary is None:
        add("向量索引", "warn", f"{paths.index} 不存在，先运行 rag-agent ingest")
    elif "error" in summary:
        add("向量索引", "error", str(summary["error"]))
    else:
        add(
            "向量索引",
            "ok",
            f"chunks={summary['chunk_count']}，维度={summary['dimension']}，"
            f"provider={summary['provider']}",
        )

    has_error = any(check["status"] == "error" for check in checks)
    if args.as_json:
        print(json.dumps({"workspace_root": str(root), "checks": checks}, ensure_ascii=False, indent=2))
    else:
        icons = {"ok": "✅", "warn": "⚠️ ", "error": "❌"}
        for check in checks:
            print(f"{icons[check['status']]} {check['name']}：{check['detail']}")
        print("\n结论：" + ("存在错误项，请先修复。" if has_error else "本地环境可用。"))
    return 1 if has_error else 0


def _index_meta_summary(index_path: Path) -> dict[str, object] | None:
    """只读索引首行 meta，避免为诊断加载整个索引。"""

    if not index_path.exists():
        return None
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            meta = json.loads(handle.readline())
    except (OSError, json.JSONDecodeError):
        return {"error": f"索引文件无法读取：{index_path}"}
    if meta.get("_type") != "meta":
        return {"error": "索引首行不是 meta，可能不是本项目生成的索引"}
    return {
        "chunk_count": meta.get("chunk_count"),
        "dimension": meta.get("dimension"),
        "provider": meta.get("provider_fingerprint"),
    }


def _handle_chat(args: argparse.Namespace) -> int:
    index = LocalVectorIndex.load(_resolve_path(args.index))
    embedding_provider = create_embedding_provider(
        args.embedding_provider,
        dimension=args.embedding_dimension or index.dimension,
        model=args.embedding_model,
        api_key=args.embedding_api_key,
        base_url=args.embedding_base_url,
    )

    history = ConversationHistory(max_turns=args.history_max_turns)
    history_path = _resolve_path(args.history) if args.history is not None else None
    if history_path is not None and history_path.exists():
        history = ConversationHistory.load(history_path, max_turns=args.history_max_turns)

    chat_provider = None
    if not args.retrieval_only:
        chat_provider = OpenAICompatibleChatProvider.from_environment(
            api_key=args.llm_api_key,
            base_url=args.llm_base_url,
            model=args.llm_model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    agent = None
    if args.agent and chat_provider is not None:
        agent = KnowledgeAgent(
            KnowledgeSearchTool(index, embedding_provider),
            chat_provider,
            max_steps=args.max_steps,
        )
    answerer = None
    if chat_provider is not None and not args.agent:
        answerer = RagAnswerer(
            index,
            embedding_provider=embedding_provider,
            chat_provider=chat_provider,
            min_score=args.min_score,
            max_context_chars=args.max_context_chars,
            top_k=args.top_k,
        )

    mode = (
        "离线检索（--retrieval-only）"
        if args.retrieval_only
        else ("agent 工具调用" if args.agent else "单轮 RAG")
    )
    print(f"rag-agent chat（{mode}）| 索引 {index.size} 条 | 输入 exit 退出，/help 查看命令。")
    while True:
        try:
            line = input("你> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break
        if not line:
            continue
        lowered = line.lower()
        if lowered in {"exit", "quit", ":q", "退出"}:
            break
        if lowered in {"help", "/help"}:
            print("exit / quit：退出会话；/reset：清空会话历史；/search 问题：只检索证据。")
            continue
        if lowered == "/reset":
            history = ConversationHistory(max_turns=args.history_max_turns)
            print("已清空会话历史。")
            continue
        if lowered.startswith("/search"):
            query = line[len("/search"):].strip()
            if not query:
                print("用法：/search 你的问题")
                continue
            _print_search_results(
                index.search(
                    query,
                    provider=embedding_provider,
                    top_k=args.top_k,
                    min_score=args.min_score,
                )
            )
            continue
        if lowered.startswith("/"):
            print(f"未知命令：{line}；输入 /help 查看可用命令。")
            continue

        try:
            if args.retrieval_only:
                _print_search_results(
                    index.search(
                        line,
                        provider=embedding_provider,
                        top_k=args.top_k,
                        min_score=args.min_score,
                    )
                )
            elif agent is not None:
                result = agent.run(line, history=history)
                history = result.history
                print(result.answer)
                if result.evidence:
                    print(f"\nAgent 检索到 {len(result.evidence)} 条证据。")
            else:
                assert answerer is not None
                result = answerer.answer(line)
                _print_answer(result)
        except KeyboardInterrupt:
            print("\n本轮已取消（再次 Ctrl+C 或输入 exit 退出会话）。")
            continue
        if history_path is not None:
            history.save(history_path)
    if history_path is not None:
        history.save(history_path)
        print(f"会话历史已写入：{history_path}")
    return 0


def _handle_ingest_url(args: argparse.Namespace) -> int:
    crawl_config = CrawlConfig(
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        delay_seconds=args.delay,
        respect_robots=not args.no_robots,
        same_domain=not args.cross_domain,
        timeout=args.timeout,
        max_page_bytes=max(1, int(args.max_page_mb * 1024 * 1024)),
    )
    chunk_config = ChunkConfig(max_chars=args.max_chars, overlap_chars=args.overlap_chars)
    output_path = _resolve_path(args.output)
    manifest_path = _resolve_path(args.manifest)
    failures_path = _resolve_path(args.failures)
    index_path = _resolve_path(args.index)

    print(
        f"开始抓取 {args.url}（最多 {args.max_pages} 页，深度 {args.max_depth}，"
        f"{'同域' if not args.cross_domain else '允许跨域'}，"
        f"{'尊重 robots.txt' if not args.no_robots else '忽略 robots.txt'}）…"
    )
    crawl = crawl_site(args.url, crawl_config)

    records = []
    failures: list[dict[str, str]] = []
    for failure in crawl.failures:
        failures.append(
            {"source_path": failure.url, "error_type": failure.error_type, "message": failure.message}
        )
    for page in crawl.pages:
        try:
            records.append(build_web_document(page.result))
        except Exception as error:  # 单页整理失败不影响其他页面
            failures.append(
                {
                    "source_path": page.url,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    merge = merge_web_documents(        records,
        _read_optional_jsonl(output_path),
        _read_optional_jsonl(manifest_path),
        chunk_config=chunk_config,
    )

    write_jsonl_atomic(output_path, merge.chunks)
    write_jsonl_atomic(manifest_path, merge.manifests)
    write_jsonl_atomic(failures_path, failures)

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
        _, index_stats = update_vector_index(
            merge.chunks,
            provider=index_provider,
            path=index_path,
        )

    summary = {
        "pages_crawled": len(crawl.pages),
        "pages_added": merge.counts.added,
        "pages_updated": merge.counts.updated,
        "pages_unchanged": merge.counts.unchanged,
        "pages_failed": len(failures),
        "chunks_written": len(merge.chunks),
        "chunks_path": str(output_path),
        "manifest_path": str(manifest_path),
        "failures_path": str(failures_path),
        "index_path": None if args.no_index else str(index_path),
        "embedding_provider": index_provider.fingerprint if index_provider else None,
        "vectors_reused": index_stats.reused_vectors if index_stats else None,
        "vectors_embedded": index_stats.embedded_vectors if index_stats else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if failures else 0


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
