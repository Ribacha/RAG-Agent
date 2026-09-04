"""Small provider contract for document and query embeddings.

The ingestion and retrieval layers only depend on this contract.  A local
deterministic provider is included for learning and offline tests; a hosted
provider can be added without changing the index format.
"""

from __future__ import annotations

from typing import Protocol, Sequence


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot create valid vectors."""


class EmbeddingProvider(Protocol):
    """The minimal interface required by the local vector index."""

    @property
    def name(self) -> str:
        """Short provider name, for example ``hash`` or ``openai``."""

    @property
    def model(self) -> str:
        """Model identifier recorded in the index manifest."""

    @property
    def dimension(self) -> int:
        """Number of coordinates in every returned vector."""

    @property
    def fingerprint(self) -> str:
        """Configuration identity used to prevent incompatible reuse."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one finite vector for every input text."""
