"""
Vector Store for RAG Pipeline.

Stores and retrieves document embeddings using ChromaDB.
"""

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from tools.rag.document_loader import Document

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None
    CHROMADB_AVAILABLE = False


class VectorStore:
    """
    Vector storage and retrieval using ChromaDB.

    Supports:
    - Local persistent storage
    - In-memory storage for testing
    - Similarity search with metadata filtering
    """

    def __init__(
        self,
        collection_name: str = "weaver_documents",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[Any] = None,
        http_endpoint: Optional[str] = None,
        http_headers: Optional[Dict[str, str]] = None,
        retrieval_mode: str = "hybrid",
        hybrid_dense_weight: float = 0.65,
        hybrid_keyword_weight: float = 0.35,
        keyword_candidate_limit: int = 200,
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage (None for in-memory)
            embedding_function: Optional custom embedding function
            http_endpoint: Optional Chroma HTTP endpoint
            http_headers: Optional headers for Chroma HTTP auth
            retrieval_mode: Retrieval mode ("dense" or "hybrid")
            hybrid_dense_weight: Weight for dense vector scores in hybrid mode
            hybrid_keyword_weight: Weight for keyword scores in hybrid mode
            keyword_candidate_limit: Max documents scanned for keyword retrieval
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required for vector storage. "
                "Install with: pip install chromadb"
            )

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.http_endpoint = (http_endpoint or "").strip()
        self.retrieval_mode = (retrieval_mode or "hybrid").strip().lower()
        self.hybrid_dense_weight = max(0.0, float(hybrid_dense_weight))
        self.hybrid_keyword_weight = max(0.0, float(hybrid_keyword_weight))
        self.keyword_candidate_limit = max(1, int(keyword_candidate_limit or 200))

        # Initialize ChromaDB client
        if self.http_endpoint:
            parsed = urlparse(self.http_endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                raise ValueError(f"Invalid Chroma HTTP endpoint: {self.http_endpoint}")

            self.client = chromadb.HttpClient(
                host=parsed.hostname,
                port=parsed.port or (443 if parsed.scheme == "https" else 80),
                ssl=parsed.scheme == "https",
                headers=http_headers or None,
                settings=Settings(anonymized_telemetry=False),
            )
        elif persist_directory:
            Path(persist_directory).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self.client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.embedding_function = embedding_function
        if self.http_endpoint:
            logger.info(f"Initialized vector store via Chroma HTTP: {collection_name} @ {self.http_endpoint}")
        else:
            logger.info(f"Initialized vector store: {collection_name}")

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects
            embeddings: Pre-computed embeddings (computed if not provided)

        Returns:
            List of document IDs
        """
        if not documents:
            return []

        # Prepare data for ChromaDB
        ids = [doc.chunk_id for doc in documents]
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Compute embeddings if not provided
        if embeddings is None and self.embedding_function:
            embeddings = self.embedding_function.embed_documents(texts)

        # Add to collection
        if embeddings:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
        else:
            # Let ChromaDB use its default embedding
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
            )

        logger.info(f"Added {len(documents)} documents to vector store")
        return ids

    def search(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            query_embedding: Pre-computed query embedding
            n_results: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of (Document, score) tuples
        """
        if self.retrieval_mode == "dense":
            return self._dense_search(
                query=query,
                query_embedding=query_embedding,
                n_results=n_results,
                filter_metadata=filter_metadata,
            )
        if self.retrieval_mode == "hybrid":
            return self._hybrid_search(
                query=query,
                query_embedding=query_embedding,
                n_results=n_results,
                filter_metadata=filter_metadata,
            )
        raise ValueError(
            f"Unsupported retrieval mode: {self.retrieval_mode}. "
            "Expected 'dense' or 'hybrid'."
        )

    def _dense_search(
        self,
        *,
        query: str,
        query_embedding: Optional[List[float]],
        n_results: int,
        filter_metadata: Optional[Dict[str, Any]],
    ) -> List[Tuple[Document, float]]:
        """Perform dense vector retrieval via Chroma."""
        # Compute query embedding if not provided
        if query_embedding is None and self.embedding_function:
            query_embedding = self.embedding_function.embed_query(query)

        # Build query kwargs
        query_kwargs = {"n_results": n_results}

        if query_embedding:
            query_kwargs["query_embeddings"] = [query_embedding]
        else:
            query_kwargs["query_texts"] = [query]

        if filter_metadata:
            query_kwargs["where"] = filter_metadata

        # Execute search
        results = self.collection.query(**query_kwargs)

        return self._convert_query_results(results)

    def _hybrid_search(
        self,
        *,
        query: str,
        query_embedding: Optional[List[float]],
        n_results: int,
        filter_metadata: Optional[Dict[str, Any]],
    ) -> List[Tuple[Document, float]]:
        """Combine dense vector retrieval with lightweight keyword retrieval."""
        dense_limit = max(n_results * 3, n_results)
        dense_results = self._dense_search(
            query=query,
            query_embedding=query_embedding,
            n_results=dense_limit,
            filter_metadata=filter_metadata,
        )
        keyword_results = self._keyword_search(
            query=query,
            n_results=dense_limit,
            filter_metadata=filter_metadata,
        )

        fused = self._fuse_ranked_results(dense_results, keyword_results, n_results=n_results)
        logger.info(
            "Hybrid retrieval returned %s results (%s dense candidates, %s keyword candidates)",
            len(fused),
            len(dense_results),
            len(keyword_results),
        )
        return fused

    def _keyword_search(
        self,
        *,
        query: str,
        n_results: int,
        filter_metadata: Optional[Dict[str, Any]],
    ) -> List[Tuple[Document, float]]:
        """Keyword-oriented retrieval over stored documents."""
        query_terms = self._tokenize(query)
        normalized_query = " ".join(query_terms)
        if not query_terms:
            return []

        get_kwargs: Dict[str, Any] = {
            "limit": self.keyword_candidate_limit,
            "include": ["documents", "metadatas"],
        }
        if filter_metadata:
            get_kwargs["where"] = filter_metadata

        try:
            result = self.collection.get(**get_kwargs)
        except Exception as e:
            logger.warning(f"Keyword retrieval fallback failed: {e}")
            return []

        documents = result.get("documents") if result else None
        metadatas = result.get("metadatas") if result else None
        ids = result.get("ids") if result else None

        if not documents:
            return []

        ranked: List[Tuple[Document, float]] = []
        for idx, text in enumerate(documents):
            metadata = metadatas[idx] if metadatas else {}
            doc_id = ids[idx] if ids else ""
            score = self._keyword_match_score(
                query_terms=query_terms,
                normalized_query=normalized_query,
                text=str(text or ""),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
            if score <= 0.0:
                continue

            ranked.append(
                (
                    Document(
                        content=str(text or ""),
                        metadata=metadata if isinstance(metadata, dict) else {},
                        chunk_id=doc_id,
                    ),
                    score,
                )
            )

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:n_results]

    def _fuse_ranked_results(
        self,
        dense_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        *,
        n_results: int,
    ) -> List[Tuple[Document, float]]:
        """Fuse dense and keyword results with weighted score blending."""
        fused: Dict[str, Dict[str, Any]] = {}
        total_weight = self.hybrid_dense_weight + self.hybrid_keyword_weight
        if total_weight <= 0:
            total_weight = 1.0

        for doc, score in dense_results:
            key = self._document_key(doc)
            fused[key] = {
                "document": doc,
                "dense": self._clamp_score(score),
                "keyword": 0.0,
            }

        for doc, score in keyword_results:
            key = self._document_key(doc)
            entry = fused.setdefault(
                key,
                {
                    "document": doc,
                    "dense": 0.0,
                    "keyword": 0.0,
                },
            )
            entry["keyword"] = max(entry["keyword"], self._clamp_score(score))
            if not entry["document"].content and doc.content:
                entry["document"] = doc

        ranked: List[Tuple[Document, float]] = []
        for entry in fused.values():
            dense_score = float(entry["dense"])
            keyword_score = float(entry["keyword"])
            blended = (
                (self.hybrid_dense_weight * dense_score)
                + (self.hybrid_keyword_weight * keyword_score)
            ) / total_weight
            if dense_score > 0 and keyword_score > 0:
                blended = min(1.0, blended + 0.05)
            ranked.append((entry["document"], blended))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:n_results]

    def _convert_query_results(self, results: Optional[Dict[str, Any]]) -> List[Tuple[Document, float]]:
        """Convert Chroma query results into Document tuples."""
        documents_with_scores: List[Tuple[Document, float]] = []

        if results and results.get("documents"):
            docs = results["documents"][0]
            metadatas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
            ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

            for text, metadata, distance, doc_id in zip(docs, metadatas, distances, ids):
                score = self._clamp_score(1.0 - float(distance))
                documents_with_scores.append(
                    (
                        Document(
                            content=text,
                            metadata=metadata,
                            chunk_id=doc_id,
                        ),
                        score,
                    )
                )

        return documents_with_scores

    def _keyword_match_score(
        self,
        *,
        query_terms: List[str],
        normalized_query: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> float:
        """Compute a lightweight lexical relevance score."""
        text_lower = text.lower()
        text_terms = self._tokenize(text)
        if not text_terms:
            return 0.0

        term_counts = Counter(text_terms)
        unique_matches = sum(1 for term in query_terms if term_counts.get(term, 0) > 0)
        if unique_matches == 0:
            return 0.0

        coverage = unique_matches / max(1, len(set(query_terms)))
        density = sum(min(term_counts.get(term, 0), 3) for term in query_terms) / max(1, len(query_terms) * 3)
        phrase_bonus = 0.15 if normalized_query and normalized_query in text_lower else 0.0

        filename_text = " ".join(
            str(metadata.get(key, "") or "").lower()
            for key in ("filename", "source")
        )
        filename_hits = sum(1 for term in set(query_terms) if term in filename_text)
        metadata_bonus = min(0.15, filename_hits * 0.05)

        return self._clamp_score((coverage * 0.55) + (density * 0.3) + phrase_bonus + metadata_bonus)

    def _document_key(self, doc: Document) -> str:
        """Build a stable key for result fusion."""
        if doc.chunk_id:
            return doc.chunk_id
        source = str(doc.metadata.get("source", "") or "")
        chunk_index = str(doc.metadata.get("chunk_index", "") or "")
        return f"{source}:{chunk_index}:{hash(doc.content)}"

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for lexical retrieval, preserving code-ish tokens."""
        if not text:
            return []
        return [
            token.lower()
            for token in re.findall(r"[A-Za-z0-9_./:-]+|[\u4e00-\u9fff]+", text)
            if token.strip()
        ]

    def _clamp_score(self, score: float) -> float:
        return max(0.0, min(1.0, float(score)))

    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Delete documents from the vector store.

        Args:
            ids: List of document IDs to delete
            filter_metadata: Delete documents matching this filter

        Returns:
            Number of documents deleted
        """
        try:
            if ids:
                self.collection.delete(ids=ids)
                return len(ids)
            elif filter_metadata:
                self.collection.delete(where=filter_metadata)
                return -1  # Unknown count
            return 0
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return 0

    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a specific document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document if found, None otherwise
        """
        try:
            result = self.collection.get(ids=[doc_id])
            if result and result.get("documents"):
                return Document(
                    content=result["documents"][0],
                    metadata=result["metadatas"][0] if result.get("metadatas") else {},
                    chunk_id=doc_id,
                )
        except Exception as e:
            logger.error(f"Get document error: {e}")
        return None

    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List documents in the store.

        Args:
            limit: Maximum documents to return
            offset: Offset for pagination

        Returns:
            List of document metadata dicts
        """
        try:
            result = self.collection.get(
                limit=limit,
                offset=offset,
                include=["metadatas"],
            )

            documents = []
            if result and result.get("ids"):
                for i, doc_id in enumerate(result["ids"]):
                    metadata = result["metadatas"][i] if result.get("metadatas") else {}
                    documents.append({
                        "id": doc_id,
                        **metadata,
                    })
            return documents

        except Exception as e:
            logger.error(f"List documents error: {e}")
            return []

    def count(self) -> int:
        """Get the number of documents in the store."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete all documents from the store."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Cleared vector store: {self.collection_name}")
