import json
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.web import WebLLMContextTool


class SampleTool(Tool):
    @property
    def name(self) -> str:
        return "sample"

    @property
    def description(self) -> str:
        return "sample tool"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "mode": {"type": "string", "enum": ["fast", "full"]},
                "meta": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "flags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["tag"],
                },
            },
            "required": ["query", "count"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return "ok"


def test_validate_params_missing_required() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi"})
    assert "missing required count" in "; ".join(errors)


def test_validate_params_type_and_range() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 0})
    assert any("count must be >= 1" in e for e in errors)

    errors = tool.validate_params({"query": "hi", "count": "2"})
    assert any("count should be integer" in e for e in errors)


def test_validate_params_enum_and_min_length() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "h", "count": 2, "mode": "slow"})
    assert any("query must be at least 2 chars" in e for e in errors)
    assert any("mode must be one of" in e for e in errors)


def test_validate_params_nested_object_and_array() -> None:
    tool = SampleTool()
    errors = tool.validate_params(
        {
            "query": "hi",
            "count": 2,
            "meta": {"flags": [1, "ok"]},
        }
    )
    assert any("missing required meta.tag" in e for e in errors)
    assert any("meta.flags[0] should be string" in e for e in errors)


def test_validate_params_ignores_unknown_fields() -> None:
    tool = SampleTool()
    errors = tool.validate_params({"query": "hi", "count": 2, "extra": "x"})
    assert errors == []


async def test_registry_returns_validation_error() -> None:
    reg = ToolRegistry()
    reg.register(SampleTool())
    result = await reg.execute("sample", {"query": "hi"})
    assert "Invalid parameters" in result


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


async def test_web_llm_context_get_params(monkeypatch) -> None:
    called: dict[str, Any] = {}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None, headers=None, timeout=None):
            called["url"] = url
            called["params"] = params
            called["headers"] = headers
            return _FakeResponse({"context": "ok"})

    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", FakeClient)

    tool = WebLLMContextTool(api_key="k")
    out = await tool.execute("asplos 2026", count=5)
    payload = json.loads(out)
    assert payload["status"] == 200
    assert payload["endpoint"].endswith("/llm/context")
    assert payload["query"] == "asplos 2026"
    assert payload["count"] == 5
    assert payload["data"]["context"] == "ok"
    assert called["params"]["q"] == "asplos 2026"
    assert called["params"]["count"] == 5
    assert called["headers"]["X-Subscription-Token"] == "k"


async def test_web_llm_context_post_fallback(monkeypatch) -> None:
    called = {"post": 0}

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None, headers=None, timeout=None):
            return _FakeResponse({}, status_code=405)

        async def post(self, url, json=None, headers=None, timeout=None):
            called["post"] += 1
            return _FakeResponse({"context": "from-post"}, status_code=200)

    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", FakeClient)

    tool = WebLLMContextTool(api_key="k")
    out = await tool.execute("fallback")
    payload = json.loads(out)
    assert payload["status"] == 200
    assert payload["data"]["context"] == "from-post"
    assert called["post"] == 1


async def test_web_llm_context_web_fetch_mode(monkeypatch) -> None:
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None, headers=None, timeout=None):
            return _FakeResponse({"results": [{"snippet": "alpha"}, {"snippet": "beta"}]}, status_code=200)

    monkeypatch.setattr("nanobot.agent.tools.web.httpx.AsyncClient", FakeClient)

    tool = WebLLMContextTool(api_key="k")
    out = await tool.execute("mode test", output_mode="web_fetch")
    payload = json.loads(out)
    assert payload["extractor"] == "brave_llm_context"
    assert payload["status"] == 200
    assert "alpha" in payload["text"]
