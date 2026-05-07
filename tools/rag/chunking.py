"""
Chunking strategies for the RAG document loader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ChunkPayload:
    """Intermediate chunk representation before Document construction."""

    content: str
    metadata: Dict[str, Any]


class ChunkingStrategy:
    """Base class for document chunking strategies."""

    name = "base"

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[ChunkPayload]:
        raise NotImplementedError

    def _build_payload(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_index: int | None = None,
    ) -> ChunkPayload:
        payload_metadata = {**metadata, "chunk_strategy": self.name}
        if chunk_index is not None:
            payload_metadata["chunk_index"] = chunk_index
        return ChunkPayload(content=content.strip(), metadata=payload_metadata)


class BasicChunkingStrategy(ChunkingStrategy):
    """Legacy paragraph-first chunking with character overlap."""

    name = "basic"

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[ChunkPayload]:
        if len(text) <= self.chunk_size:
            return [self._build_payload(text, metadata)]

        chunks: List[ChunkPayload] = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
                continue

            if current_chunk:
                chunks.append(
                    self._build_payload(
                        current_chunk,
                        metadata,
                        chunk_index=len(chunks),
                    )
                )
                overlap_text = (
                    current_chunk[-self.chunk_overlap:]
                    if len(current_chunk) > self.chunk_overlap
                    else ""
                )
                current_chunk = f"{overlap_text}\n\n{para}" if overlap_text else para
                continue

            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(current_chunk) + len(sent) + 1 <= self.chunk_size:
                    current_chunk = f"{current_chunk} {sent}" if current_chunk else sent
                else:
                    if current_chunk:
                        chunks.append(
                            self._build_payload(
                                current_chunk,
                                metadata,
                                chunk_index=len(chunks),
                            )
                        )
                    current_chunk = sent

        if current_chunk.strip():
            chunks.append(
                self._build_payload(
                    current_chunk,
                    metadata,
                    chunk_index=len(chunks),
                )
            )

        return chunks


class SemanticChunkingStrategy(ChunkingStrategy):
    """
    Structure-aware chunking.

    Preserves headings, prefers sentence boundaries, and applies overlap on
    semantic units instead of raw characters.
    """

    name = "semantic"

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[ChunkPayload]:
        if len(text) <= self.chunk_size:
            return [self._build_payload(text, metadata)]

        units = self._build_semantic_units(text)
        if not units:
            return BasicChunkingStrategy(self.chunk_size, self.chunk_overlap).chunk(text, metadata)

        chunks: List[ChunkPayload] = []
        current_units: List[str] = []
        current_len = 0

        for unit in units:
            unit_len = len(unit)
            separator_len = 2 if current_units else 0
            if current_units and current_len + separator_len + unit_len > self.chunk_size:
                chunks.append(
                    self._build_payload(
                        "\n\n".join(unit.strip() for unit in current_units if unit.strip()),
                        metadata,
                        chunk_index=len(chunks),
                    )
                )
                current_units, current_len = self._build_overlap_units(current_units)
                while current_units and current_len + 2 + unit_len > self.chunk_size:
                    current_units.pop(0)
                    current_len = self._units_length(current_units)

            if current_units:
                current_len += 2
            current_units.append(unit)
            current_len += unit_len

        if current_units:
            chunks.append(
                self._build_payload(
                    "\n\n".join(unit.strip() for unit in current_units if unit.strip()),
                    metadata,
                    chunk_index=len(chunks),
                )
            )

        return chunks

    def _build_semantic_units(self, text: str) -> List[str]:
        blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
        units: List[str] = []
        current_heading = ""

        for block in blocks:
            normalized = self._normalize_block(block)
            if not normalized:
                continue

            if self._is_heading_block(normalized):
                current_heading = normalized
                continue

            for piece in self._split_block_for_semantics(normalized):
                units.extend(self._attach_heading_to_piece(current_heading, piece))

        return units

    def _split_block_for_semantics(self, block: str) -> List[str]:
        if len(block) <= self.chunk_size:
            return [block]

        sentences = self._split_sentences(block)
        if len(sentences) <= 1:
            return self._split_oversized_text(block)

        pieces: List[str] = []
        current = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > self.chunk_size:
                if current:
                    pieces.append(current.strip())
                    current = ""
                pieces.extend(self._split_oversized_text(sentence))
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    pieces.append(current.strip())
                current = sentence

        if current:
            pieces.append(current.strip())

        return pieces or self._split_oversized_text(block)

    def _build_overlap_units(self, units: List[str]) -> Tuple[List[str], int]:
        if self.chunk_overlap <= 0 or not units:
            return [], 0

        overlap_units: List[str] = []
        total = 0

        for unit in reversed(units):
            extra = len(unit) + (2 if overlap_units else 0)
            if overlap_units and total + extra > self.chunk_overlap:
                break
            if not overlap_units and len(unit) > self.chunk_overlap:
                break
            overlap_units.insert(0, unit)
            total += extra

        return overlap_units, total

    def _split_sentences(self, text: str) -> List[str]:
        normalized = self._normalize_block(text)
        if not normalized:
            return []

        parts = re.split(r"(?<=[。！？；])\s*|(?<=[.!?;])\s+", normalized)
        sentences = [part.strip() for part in parts if part.strip()]
        return sentences or [normalized]

    def _split_oversized_text(self, text: str, limit: int | None = None) -> List[str]:
        cleaned = self._normalize_block(text)
        if not cleaned:
            return []

        size_limit = max(1, limit or self.chunk_size)
        parts: List[str] = []
        start = 0
        text_len = len(cleaned)

        while start < text_len:
            end = min(start + size_limit, text_len)
            if end < text_len:
                split_at = cleaned.rfind(" ", start, end)
                if split_at <= start:
                    split_at = cleaned.rfind("，", start, end)
                if split_at <= start:
                    split_at = cleaned.rfind(",", start, end)
                if split_at <= start:
                    split_at = end
            else:
                split_at = end

            part = cleaned[start:split_at].strip()
            if part:
                parts.append(part)
            start = split_at
            while start < text_len and cleaned[start].isspace():
                start += 1

        return parts

    def _attach_heading_to_piece(self, heading: str, piece: str) -> List[str]:
        piece = self._normalize_block(piece)
        if not piece:
            return []
        if not heading or piece.startswith(heading):
            return [piece]

        headed_piece = f"{heading}\n{piece}"
        if len(headed_piece) <= self.chunk_size:
            return [headed_piece]

        available = max(1, self.chunk_size - len(heading) - 1)
        return [
            f"{heading}\n{sub_piece}"
            for sub_piece in self._split_oversized_text(piece, limit=available)
        ]

    def _units_length(self, units: List[str]) -> int:
        if not units:
            return 0
        return sum(len(unit) for unit in units) + (2 * (len(units) - 1))

    def _is_heading_block(self, block: str) -> bool:
        stripped = block.strip()
        if not stripped:
            return False
        if stripped.startswith("#"):
            return True
        if re.match(r"^\[Page \d+\]$", stripped):
            return True
        if re.match(r"^(chapter|section|part)\s+\d+[\w.-]*", stripped, flags=re.IGNORECASE):
            return True
        if re.match(r"^\d+(?:\.\d+){0,3}[\)\.]?\s+\S+", stripped):
            return len(stripped) <= 120
        if re.match(r"^[A-Z][A-Z0-9\s\-:]{3,80}$", stripped):
            return True
        return False

    def _normalize_block(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()


def build_chunking_strategy(
    strategy_name: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> ChunkingStrategy:
    """Factory for chunking strategies."""
    normalized = (strategy_name or "semantic").strip().lower()
    strategies = {
        BasicChunkingStrategy.name: BasicChunkingStrategy,
        SemanticChunkingStrategy.name: SemanticChunkingStrategy,
    }
    strategy_cls = strategies.get(normalized)
    if strategy_cls is None:
        raise ValueError(
            f"Unsupported chunk strategy: {normalized}. Expected one of: "
            f"{', '.join(sorted(strategies))}."
        )
    return strategy_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
