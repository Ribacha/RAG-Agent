"""Local vector index and source-aware search results."""

from .index import (
    INDEX_SCHEMA_VERSION,
    IndexUpdateStats,
    LocalVectorIndex,
    SearchResult,
    build_vector_index,
    update_vector_index,
)

__all__ = [
    "INDEX_SCHEMA_VERSION",
    "IndexUpdateStats",
    "LocalVectorIndex",
    "SearchResult",
    "build_vector_index",
    "update_vector_index",
]
