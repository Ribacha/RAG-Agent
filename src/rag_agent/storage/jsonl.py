"""JSON Lines 读写工具。

写入先落到同一目录下的随机临时文件，再用 ``replace`` 完成切换。随机名称
避免两个终端同时执行 ingest 时互相删除对方的临时文件。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """先写临时文件，再原子替换目标文件，避免中途失败留下半个索引。"""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """逐行读取 JSONL，并在行格式错误时报告文件和行号。"""

    path = path.expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL 文件 {path} 第 {line_number} 行格式错误：{error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL 文件 {path} 第 {line_number} 行不是对象"
                )
            yield value
