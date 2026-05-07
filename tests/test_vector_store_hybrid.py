from tools.rag.vector_store import Document, VectorStore


class _FakeEmbedder:
    def embed_query(self, query: str):
        return [0.1, 0.2, 0.3]


class _FakeCollection:
    def __init__(self):
        self.query_calls = []
        self.get_calls = []

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {
            "documents": [[
                "High level deployment guide for the platform",
                "The API_BASE_URL setting must be set in .env",
            ]],
            "metadatas": [[
                {"source": "guide.md", "filename": "guide.md", "chunk_index": 0},
                {"source": "config.md", "filename": "config.md", "chunk_index": 1},
            ]],
            "distances": [[0.05, 0.35]],
            "ids": [["dense-guide", "dense-config"]],
        }

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        return {
            "documents": [
                "High level deployment guide for the platform",
                "The API_BASE_URL setting must be set in .env",
                "Generic troubleshooting information",
            ],
            "metadatas": [
                {"source": "guide.md", "filename": "guide.md", "chunk_index": 0},
                {"source": "config.md", "filename": "config.md", "chunk_index": 1},
                {"source": "misc.md", "filename": "misc.md", "chunk_index": 2},
            ],
            "ids": ["dense-guide", "dense-config", "misc"],
        }


def _build_store(retrieval_mode: str) -> VectorStore:
    store = VectorStore.__new__(VectorStore)
    store.collection_name = "test"
    store.persist_directory = None
    store.http_endpoint = ""
    store.collection = _FakeCollection()
    store.embedding_function = _FakeEmbedder()
    store.retrieval_mode = retrieval_mode
    store.hybrid_dense_weight = 0.65
    store.hybrid_keyword_weight = 0.35
    store.keyword_candidate_limit = 50
    return store


def test_hybrid_search_promotes_keyword_exact_match():
    store = _build_store("hybrid")

    results = store.search("API_BASE_URL", n_results=2)

    assert len(results) == 2
    assert results[0][0].metadata["filename"] == "config.md"
    assert results[0][1] > results[1][1]
    assert store.collection.query_calls
    assert store.collection.get_calls


def test_dense_search_keeps_vector_order_without_keyword_fusion():
    store = _build_store("dense")

    results = store.search("API_BASE_URL", n_results=2)

    assert len(results) == 2
    assert results[0][0].metadata["filename"] == "guide.md"
    assert results[1][0].metadata["filename"] == "config.md"
    assert store.collection.query_calls
    assert not store.collection.get_calls
