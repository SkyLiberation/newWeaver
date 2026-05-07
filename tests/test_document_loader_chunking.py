from tools.rag.chunking import (
    BasicChunkingStrategy,
    SemanticChunkingStrategy,
    build_chunking_strategy,
)
from tools.rag.document_loader import DocumentLoader


def test_semantic_chunking_preserves_headings_and_sentence_boundaries():
    text = """
## Introduction

Sentence one explains the topic. Sentence two adds more context. Sentence three keeps going.

## Details

Point A is important. Point B is also important. Point C closes the section.
""".strip()

    loader = DocumentLoader(
        chunk_size=80,
        chunk_overlap=20,
        chunk_strategy="semantic",
    )

    chunks = loader._chunk_text(text, {"source": "demo.md", "filename": "demo.md"})

    assert len(chunks) >= 3
    assert all(chunk.metadata["chunk_strategy"] == "semantic" for chunk in chunks)
    assert chunks[0].content.startswith("## Introduction")
    assert any("## Details" in chunk.content for chunk in chunks)
    assert all(not chunk.content.startswith("topic.") for chunk in chunks[1:])
    assert all(len(chunk.content) <= 80 for chunk in chunks)


def test_basic_chunking_strategy_remains_available():
    text = (
        "A" * 60
        + "\n\n"
        + "B" * 60
        + "\n\n"
        + "C" * 60
    )

    loader = DocumentLoader(
        chunk_size=70,
        chunk_overlap=10,
        chunk_strategy="basic",
    )

    chunks = loader._chunk_text(text, {"source": "demo.txt", "filename": "demo.txt"})

    assert len(chunks) >= 2
    assert all(chunk.metadata["chunk_strategy"] == "basic" for chunk in chunks)
    assert any("AAAAAAAAAA" in chunk.content for chunk in chunks[1:])


def test_semantic_chunking_splits_oversized_text_without_exceeding_chunk_size():
    sentence = "This is a very long sentence with enough words to require splitting"
    text = " ".join([sentence] * 20)

    loader = DocumentLoader(
        chunk_size=90,
        chunk_overlap=0,
        chunk_strategy="semantic",
    )

    chunks = loader._chunk_text(text, {"source": "long.txt", "filename": "long.txt"})

    assert len(chunks) > 1
    assert all(len(chunk.content) <= 90 for chunk in chunks)


def test_chunking_strategy_factory_returns_expected_classes():
    basic = build_chunking_strategy("basic", chunk_size=100, chunk_overlap=10)
    semantic = build_chunking_strategy("semantic", chunk_size=100, chunk_overlap=10)

    assert isinstance(basic, BasicChunkingStrategy)
    assert isinstance(semantic, SemanticChunkingStrategy)
