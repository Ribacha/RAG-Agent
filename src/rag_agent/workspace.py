"""工作区发现与创建。

安装后的 ``rag-agent`` 命令要能像 Codex/Claude Code 那样在任意目录使用，
而不是只在仓库检出内工作：``rag-agent init`` 用 ``.rag-agent/`` 标记一个
工作区并创建 ``data/`` 目录结构，后续命令从当前目录逐级向上查找该标记来
定位数据。仓库检出（pyproject.toml + src）仍按原逻辑解析，开发流程不变。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from . import __version__


WORKSPACE_DIR = ".rag-agent"
WORKSPACE_META = "workspace.json"

# init 生成的 .env 模板。{api_key} 为空时保持占位注释，提示用户手动填写。
ENV_TEMPLATE = """# rag-agent 配置（由 `rag-agent init` 生成）。
# Shell 中已 export 的同名变量优先于 .env；命令行 --llm-* 参数优先级最高。
# ingest、search、evaluate 和 chat --retrieval-only 都可以离线运行；
# 只有 ask、agent 和 chat（非 retrieval-only）需要聊天模型 Key。
LLM_API_KEY={api_key}
LLM_BASE_URL={base_url}
LLM_MODEL={model}

# Embedding 与聊天模型分开配置。hash/chinese 为本地离线基线；
# openai 需要服务确实提供 /embeddings 接口，取消注释并填写后才能使用。
# 修改 provider 或维度后必须用 rebuild-index 重建索引。
EMBEDDING_PROVIDER={embedding_provider}
# EMBEDDING_API_KEY=你的EmbeddingKey
# EMBEDDING_BASE_URL=https://你的服务/v1
# EMBEDDING_MODEL=你的embedding模型
"""


@dataclass(frozen=True)
class WorkspacePaths:
    """一个工作区内约定俗成的数据文件位置。"""

    root: Path
    env_file: Path
    inbox: Path
    chunks: Path
    manifest: Path
    failures: Path
    index: Path
    state: Path


def paths_for(root: Path) -> WorkspacePaths:
    index_dir = root / "data" / "index"
    return WorkspacePaths(
        root=root,
        env_file=root / ".env",
        inbox=root / "data" / "inbox",
        chunks=index_dir / "chunks.jsonl",
        manifest=index_dir / "documents.jsonl",
        failures=root / "data" / "failed" / "ingestion.jsonl",
        index=index_dir / "vectors.jsonl",
        state=root / "data" / "state",
    )


def is_repo_root(candidate: Path) -> bool:
    return (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir()


def is_workspace_root(candidate: Path) -> bool:
    return (candidate / WORKSPACE_DIR).is_dir()


def find_workspace_root(start: Path | None = None) -> Path:
    """按优先级解析生效的工作区根目录。

    1. ``RAG_AGENT_ROOT`` 环境变量（显式覆盖，脚本和测试使用）；
    2. 从起始目录逐级向上查找 ``.rag-agent/`` 标记或仓库布局；
    3. 包安装位置所在的仓库检出（支持从任意 cwd 以源码运行）；
    4. 兜底返回起始目录（配合 ``rag-agent init`` 创建工作区）。
    """

    configured = os.getenv("RAG_AGENT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if is_workspace_root(candidate) or is_repo_root(candidate):
            return candidate
    source_root = Path(__file__).resolve().parents[2]
    if is_repo_root(source_root):
        return source_root
    return current


def create_workspace(root: Path) -> WorkspacePaths:
    """创建工作区目录结构和标记文件；对已存在的结构是幂等的。"""

    root = root.expanduser().resolve()
    paths = paths_for(root)
    for directory in (
        paths.inbox,
        paths.chunks.parent,
        paths.failures.parent,
        paths.state,
        root / WORKSPACE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    meta_path = root / WORKSPACE_DIR / WORKSPACE_META
    if not meta_path.exists():
        meta = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tool_version": __version__,
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return paths


def render_env_template(
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    embedding_provider: str,
) -> str:
    return ENV_TEMPLATE.format(
        api_key=api_key or "sk-你的DeepSeekKey",
        base_url=base_url,
        model=model,
        embedding_provider=embedding_provider,
    )
