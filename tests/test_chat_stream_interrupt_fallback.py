import json
from types import SimpleNamespace

import pytest


class _FakeGraph:
    async def astream_events(self, *_args, **_kwargs):
        yield {
            "event": "on_node_start",
            "name": "writer",
            "data": {},
        }
        yield {
            "event": "on_node_end",
            "name": "writer",
            "data": {"output": {"draft_report": "draft v1"}},
        }


class _FakeCheckpointer:
    def get_tuple(self, _config):
        interrupt_payload = SimpleNamespace(
            value={
                "checkpoint": "draft",
                "instruction": "Review draft",
                "content": "draft v1",
            }
        )
        return SimpleNamespace(
            pending_writes=[
                ("task", "__interrupt__", [interrupt_payload]),
            ]
        )


@pytest.mark.asyncio
async def test_stream_agent_events_emits_interrupt_from_pending_checkpoint(monkeypatch):
    import main

    monkeypatch.setattr(main, "research_graph", _FakeGraph())
    monkeypatch.setattr(main, "checkpointer", _FakeCheckpointer())

    chunks = []
    async for chunk in main.stream_agent_events(
        input_text="hello",
        thread_id="thread-interrupt-fallback",
        model="deepseek-chat",
        search_mode={"mode": "web", "use_web": True, "use_agent": False, "use_deep": False},
        user_id="test-user",
    ):
        chunks.append(chunk)

    payloads = []
    for chunk in chunks:
        if not isinstance(chunk, str) or not chunk.startswith("0:"):
            continue
        payloads.append(json.loads(chunk[2:]))

    event_types = [payload.get("type") for payload in payloads if isinstance(payload, dict)]
    assert "interrupt" in event_types
    assert "done" not in event_types

    interrupt = next(payload for payload in payloads if payload.get("type") == "interrupt")
    assert interrupt["data"]["thread_id"] == "thread-interrupt-fallback"
    assert interrupt["data"]["prompts"][0]["checkpoint"] == "draft"
