"""
RAG Tool for Research Pipeline.

Provides a LangChain-compatible tool for searching local documents.
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from tools.rag.document_loader import DocumentLoader
from tools.rag.embedder import Embedder
from tools.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Global RAG instances (lazy initialized, keyed by collection name)
_RAG_BY_COLLECTION: dict[str, "RAGTool"] = {}


class RAGTool:
    """
    RAG Tool that integrates document loading, embedding, and retrieval.

    Used by the research pipeline to search local documents.
    """

    def __init__(
        self,
        collection_name: str = "weaver_documents",
        persist_directory: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_strategy: str = "semantic",
        http_endpoint: Optional[str] = None,
        http_headers: Optional[Dict[str, str]] = None,
        retrieval_mode: str = "hybrid",
        hybrid_dense_weight: float = 0.65,
        hybrid_keyword_weight: float = 0.35,
        keyword_candidate_limit: int = 200,
    ):
        """
        Initialize the RAG tool.

        Args:
            collection_name: Name for the vector collection
            persist_directory: Directory for persistent storage
            embedding_model: OpenAI embedding model name
            chunk_size: Document chunk size
            chunk_overlap: Chunk overlap for context
            chunk_strategy: Chunking strategy for document splitting
            http_endpoint: Optional Chroma HTTP endpoint
            http_headers: Optional headers for Chroma HTTP auth
            retrieval_mode: Retrieval mode ("dense" or "hybrid")
            hybrid_dense_weight: Weight for dense vector scores in hybrid mode
            hybrid_keyword_weight: Weight for keyword scores in hybrid mode
            keyword_candidate_limit: Max documents scanned for keyword retrieval
        """
        self.loader = DocumentLoader(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunk_strategy=chunk_strategy,
        )
        self.embedder = Embedder(model=embedding_model)
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            http_endpoint=http_endpoint,
            http_headers=http_headers,
            embedding_function=self.embedder,
            retrieval_mode=retrieval_mode,
            hybrid_dense_weight=hybrid_dense_weight,
            hybrid_keyword_weight=hybrid_keyword_weight,
            keyword_candidate_limit=keyword_candidate_limit,
        )

    def add_document(
        self,
        file_path: str = None,
        content: bytes = None,
        filename: str = None,
    ) -> Dict[str, Any]:
        """
        Add a document to the RAG store.

        Args:
            file_path: Path to document file
            content: Document content as bytes
            filename: Original filename (required if using content)

        Returns:
            Dict with document info
        """
        try:
            if file_path:
                documents = self.loader.load(file_path)
                source = file_path
            elif content and filename:
                documents = self.loader.load_from_bytes(content, filename)
                source = filename
            else:
                raise ValueError("Either file_path or (content, filename) required")

            if not documents:
                return {
                    "success": False,
                    "error": "No content extracted from document",
                    "source": source,
                }

            # Generate embeddings
            texts = [doc.content for doc in documents]
            embeddings = self.embedder.embed_documents(texts)

            # Add to vector store
            ids = self.vector_store.add_documents(documents, embeddings)

            logger.info(f"Added document: {source} ({len(documents)} chunks)")

            return {
                "success": True,
                "source": source,
                "chunks": len(documents),
                "ids": ids,
            }

        except Exception as e:
            logger.error(f"Add document error: {e}")
            return {
                "success": False,
                "error": str(e),
                "source": file_path or filename,
            }

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query
            n_results: Number of results
            filter_source: Filter by source file

        Returns:
            List of result dicts with content and metadata
        """
        filter_metadata = None
        if filter_source:
            filter_metadata = {"source": filter_source}

        results = self.vector_store.search(
            query=query,
            n_results=n_results,
            filter_metadata=filter_metadata,
        )

        return [
            {
                "content": doc.content,
                "score": score,
                "source": doc.metadata.get("source", "unknown"),
                "filename": doc.metadata.get("filename", "unknown"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
            }
            for doc, score in results
        ]

    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List all documents in the store."""
        return self.vector_store.list_documents(limit=limit)

    def delete_document(self, source: str) -> Dict[str, Any]:
        """
        Delete all chunks from a specific source.

        Args:
            source: Source file path to delete

        Returns:
            Result dict
        """
        try:
            count = self.vector_store.delete_documents(
                filter_metadata={"source": source}
            )
            return {"success": True, "deleted": count}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def count(self) -> int:
        """Get total number of chunks in store."""
        return self.vector_store.count()


def get_rag_tool(*, collection_name: Optional[str] = None) -> Optional[RAGTool]:
    """
    Get a cached RAG tool instance.

    Weaver can run in either:
    - single-user/dev mode: a single global collection
    - enterprise-internal mode: per-principal isolated collections (caller chooses collection_name)
    """

    from common.config import settings

    if not getattr(settings, "rag_enabled", False):
        return None

    try:
        base_collection = getattr(settings, "rag_collection_name", "weaver_documents")
        resolved_collection = (collection_name or base_collection or "weaver_documents").strip()

        existing = _RAG_BY_COLLECTION.get(resolved_collection)
        if existing is not None:
            return existing

        chroma_header = (getattr(settings, "rag_chroma_api_header", "") or "").strip()
        chroma_key = (getattr(settings, "rag_chroma_api_key", "") or "").strip()
        rag = RAGTool(
            collection_name=resolved_collection,
            persist_directory=getattr(settings, "rag_store_path", None),
            embedding_model=(
                getattr(settings, "embedding_model", "")
                or getattr(settings, "rag_embedding_model", "text-embedding-3-small")
            ),
            chunk_size=getattr(settings, "rag_chunk_size", 1000),
            chunk_overlap=getattr(settings, "rag_chunk_overlap", 200),
            chunk_strategy=getattr(settings, "rag_chunk_strategy", "semantic"),
            http_endpoint=getattr(settings, "rag_chroma_endpoint", None),
            http_headers={chroma_header: chroma_key} if chroma_header and chroma_key else None,
            retrieval_mode=getattr(settings, "rag_retrieval_mode", "hybrid"),
            hybrid_dense_weight=getattr(settings, "rag_hybrid_dense_weight", 0.65),
            hybrid_keyword_weight=getattr(settings, "rag_hybrid_keyword_weight", 0.35),
            keyword_candidate_limit=getattr(settings, "rag_keyword_candidate_limit", 200),
        )
        _RAG_BY_COLLECTION[resolved_collection] = rag
        return rag
    except Exception as e:
        logger.error(f"Failed to initialize RAG tool: {e}")
        return None


@tool
def rag_search(query: str, n_results: int = 5) -> str:
    """
    Search local documents for information relevant to the query.

    Use this tool when you need to find information from uploaded documents,
    PDFs, or other local files that have been added to the knowledge base.

    Args:
        query: The search query describing what information you need
        n_results: Number of results to return (default 5)

    Returns:
        Relevant excerpts from local documents with source information
    """
    rag = get_rag_tool()
    if rag is None:
        return "RAG search is not enabled. Please enable it in settings."

    results = rag.search(query, n_results=n_results)

    if not results:
        return "No relevant documents found."

    output_parts = []
    for i, r in enumerate(results, 1):
        output_parts.append(
            f"[{i}] Source: {r['filename']} (score: {r['score']:.2f})\n"
            f"{r['content'][:500]}..."
        )

    return "\n\n---\n\n".join(output_parts)
