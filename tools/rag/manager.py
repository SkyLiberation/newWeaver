from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List

from fastapi import HTTPException, Request, UploadFile
from langchain_core.messages import HumanMessage, SystemMessage
from starlette.concurrency import run_in_threadpool

from tools.rag.rag_tool import get_rag_tool

logger = logging.getLogger(__name__)


class RAGManager:
    """Coordinate request-scoped RAG operations for the API layer."""

    MAX_FILE_SIZE = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "md", "csv"}

    def __init__(
        self,
        *,
        settings: Any,
        metrics_registry: Any,
        format_stream_event: Callable[[str, Any], Awaitable[str]],
        chat_model_factory: Callable[..., Any],
        persona_instruction_builder: Callable[[str], str],
        contains_cjk: Callable[[str], bool],
    ) -> None:
        self.settings = settings
        self.metrics_registry = metrics_registry
        self.format_stream_event = format_stream_event
        self.chat_model_factory = chat_model_factory
        self.persona_instruction_builder = persona_instruction_builder
        self.contains_cjk = contains_cjk

    def _ensure_enabled(self, detail: str = "RAG is not enabled.") -> None:
        if not getattr(self.settings, "rag_enabled", False):
            raise HTTPException(status_code=400, detail=detail)

    def collection_for_request(self, request: Request) -> str:
        """
        Resolve the Chroma collection name for RAG documents.

        Hybrid behavior:
        - Default/dev (internal auth disabled): single shared collection
        - Enterprise internal (internal auth enabled): per-principal isolated collection
        """
        base = (
            getattr(self.settings, "rag_collection_name", "") or "weaver_documents"
        ).strip() or "weaver_documents"
        internal_key = (getattr(self.settings, "internal_api_key", "") or "").strip()
        if not internal_key:
            return base

        principal_id = (getattr(request.state, "principal_id", "") or "").strip() or "internal"
        suffix = hashlib.sha256(principal_id.encode("utf-8")).hexdigest()[:12]
        return f"{base}__u_{suffix}"

    def get_tool(self, request: Request):
        self._ensure_enabled()
        rag = get_rag_tool(collection_name=self.collection_for_request(request))
        if rag is None:
            raise HTTPException(status_code=500, detail="Failed to initialize RAG tool")
        return rag

    async def run_search(self, request: Request, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        rag = self.get_tool(request)
        return await run_in_threadpool(rag.search, query, n_results)

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        blocks: List[str] = []
        for idx, item in enumerate(results, start=1):
            if not isinstance(item, dict):
                continue
            filename = str(item.get("filename") or item.get("source") or f"Document {idx}").strip()
            source = str(item.get("source") or filename).strip()
            excerpt = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
            score = item.get("score")
            if len(excerpt) > 1200:
                excerpt = excerpt[:1200] + "..."

            header = f"[{idx}] {filename}"
            if source:
                header += f" | source={source}"
            if isinstance(score, (int, float)):
                header += f" | score={float(score):.3f}"
            blocks.append(f"{header}\n{excerpt}")

        return "\n\n".join(blocks)

    async def answer(
        self,
        *,
        request: Request,
        query: str,
        model: str,
        persona_instruction: str = "",
        n_results: int = 5,
    ) -> Dict[str, Any]:
        results = await self.run_search(request, query, n_results=n_results)
        if not results:
            empty_message = (
                "\u672a\u5728\u77e5\u8bc6\u5e93\u4e2d\u68c0\u7d22\u5230\u76f8\u5173\u5185\u5bb9\u3002"
                if self.contains_cjk(query)
                else "No relevant knowledge base documents were found."
            )
            return {"content": empty_message, "results": []}

        context = self.format_context(results)

        if not (getattr(self.settings, "openai_api_key", "") or "").strip():
            return {"content": context, "results": results}

        messages: List[Any] = [
            SystemMessage(
                content=(
                    "You are Weaver answering from the user's uploaded knowledge base. "
                    "Use the provided excerpts as the primary source of truth. "
                    "If the answer is not supported by the excerpts, say so clearly. "
                    "Keep the answer concise and mention the source filename when helpful."
                )
            )
        ]
        if persona_instruction:
            messages.append(
                SystemMessage(content=self.persona_instruction_builder(persona_instruction))
            )
        messages.append(
            HumanMessage(
                content=(
                    f"User question:\n{query}\n\n"
                    f"Knowledge base excerpts:\n{context}\n\n"
                    "Answer the user based on the excerpts above."
                )
            )
        )

        llm = self.chat_model_factory(model, temperature=0.2)
        response = await run_in_threadpool(llm.invoke, messages)
        content = getattr(response, "content", None) or str(response)
        return {"content": str(content).strip(), "results": results}

    async def stream_events(
        self,
        *,
        request: Request,
        query: str,
        thread_id: str,
        model: str,
        persona_instruction: str = "",
    ) -> Any:
        metrics = self.metrics_registry.start(thread_id, model=model, route="rag")
        try:
            yield await self.format_stream_event(
                "status",
                {"text": "Searching knowledge base...", "step": "rag_search", "thread_id": thread_id},
            )

            rag_result = await self.answer(
                request=request,
                query=query,
                model=model,
                persona_instruction=persona_instruction,
            )
            results = rag_result.get("results", []) or []
            if results:
                source_items = []
                for item in results:
                    filename = str(item.get("filename") or item.get("source") or "Knowledge Base").strip()
                    source = str(item.get("source") or filename).strip()
                    source_items.append({"title": filename, "url": source})
                yield await self.format_stream_event("sources", {"items": source_items})

            yield await self.format_stream_event(
                "completion", {"content": str(rag_result.get("content") or "")}
            )
            self.metrics_registry.finish(thread_id, cancelled=False)
            yield await self.format_stream_event(
                "done",
                {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": metrics.to_dict() if metrics else {},
                    "thread_id": thread_id,
                },
            )
        except Exception as e:
            self.metrics_registry.finish(thread_id, cancelled=False)
            logger.error(f"RAG stream error | Thread: {thread_id} | Error: {e}", exc_info=True)
            yield await self.format_stream_event(
                "error", {"message": str(e), "thread_id": thread_id}
            )

    async def upload_document(self, request: Request, file: UploadFile) -> Dict[str, Any]:
        self._ensure_enabled("RAG is not enabled. Set rag_enabled=True in settings.")

        content = await file.read()
        if len(content) > self.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {self.MAX_FILE_SIZE // (1024 * 1024)}MB.",
            )

        filename = file.filename or ""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in self.ALLOWED_EXTENSIONS:
            allowed = ", ".join(sorted(self.ALLOWED_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '.{ext}'. Allowed: {allowed}",
            )

        try:
            rag = self.get_tool(request)
            result = rag.add_document(content=content, filename=file.filename)
            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error", "Upload failed"))

            return {
                "success": True,
                "filename": file.filename,
                "chunks": result.get("chunks", 0),
                "message": (
                    f"Document '{file.filename}' uploaded successfully "
                    f"with {result.get('chunks', 0)} chunks"
                ),
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document upload error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def list_documents(self, request: Request, limit: int = 100) -> Dict[str, Any]:
        self._ensure_enabled()
        try:
            rag = self.get_tool(request)
            documents = rag.list_documents(limit=limit)
            count = rag.count()
            return {"total_chunks": count, "documents": documents}
        except Exception as e:
            logger.error(f"List documents error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def delete_document(self, request: Request, source: str) -> Dict[str, Any]:
        self._ensure_enabled()
        try:
            rag = self.get_tool(request)
            result = rag.delete_document(source)
            if not result.get("success"):
                raise HTTPException(status_code=400, detail=result.get("error", "Delete failed"))
            return {"success": True, "message": f"Document '{source}' deleted"}
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Delete document error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    async def search_documents(
        self, request: Request, query: str, n_results: int = 5
    ) -> Dict[str, Any]:
        self._ensure_enabled()
        try:
            rag = self.get_tool(request)
            results = rag.search(query, n_results=n_results)
            return {"query": query, "results": results}
        except Exception as e:
            logger.error(f"Search documents error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
