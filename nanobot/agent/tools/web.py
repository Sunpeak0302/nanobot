"""Web tools: web_search and web_fetch."""

import html
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from nanobot.agent.tools.base import Tool

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5  # Limit redirects to prevent DoS attacks
BRAVE_LLM_CONTEXT_URL = "https://api.search.brave.com/res/v1/llm/context"


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r'<script[\s\S]*?</script>', '', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', '', text, flags=re.I)
    text = re.sub(r'<[^>]+>', '', text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    """Normalize whitespace."""
    text = re.sub(r'[ \t]+', ' ', text)
    return re.sub(r'\n{3,}', '\n\n', text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    """Validate URL: must be http(s) with valid domain."""
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https'):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


class WebSearchTool(Tool):
    """Search the web using Brave Search API."""
    
    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10}
        },
        "required": ["query"]
    }
    
    def __init__(self, api_key: str | None = None, max_results: int = 5):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self.max_results = max_results
    
    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        if not self.api_key:
            return "Error: BRAVE_API_KEY not configured"
        
        try:
            n = min(max(count or self.max_results, 1), 10)
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": n},
                    headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                    timeout=10.0
                )
                r.raise_for_status()
            
            results = r.json().get("web", {}).get("results", [])
            if not results:
                return f"No results for: {query}"
            
            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results[:n], 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
                if desc := item.get("description"):
                    lines.append(f"   {desc}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"


class WebLLMContextTool(Tool):
    """Fetch Brave LLM Context payload.

    Default output is Brave raw payload. A web_fetch-like output mode is
    also reserved for future replacement/A-B switching with web_fetch.
    """

    name = "web_llm_context"
    description = "Fetch LLM-ready web context for a query."
    DEFAULT_OUTPUT_MODE = "raw"
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10},
            "output_mode": {
                "type": "string",
                "description": "Output format: raw Brave payload or web_fetch-like envelope",
                "enum": ["raw", "web_fetch"],
                "default": DEFAULT_OUTPUT_MODE,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
    ):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self.max_results = max_results

    async def execute(
        self,
        query: str,
        count: int | None = None,
        output_mode: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not self.api_key:
            return "Error: BRAVE_API_KEY not configured"

        mode = output_mode if output_mode in {"raw", "web_fetch"} else self.DEFAULT_OUTPUT_MODE
        n = min(max(count or self.max_results, 1), 10)
        params = {"q": query, "count": n}
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key,
        }

        try:
            resp = await self._request_context(params, headers)
            data = resp.json()
            final_url = str(getattr(resp, "url", BRAVE_LLM_CONTEXT_URL))
            if mode == "web_fetch":
                text = self._extract_context_text(data)
                payload = {
                    "url": BRAVE_LLM_CONTEXT_URL,
                    "finalUrl": final_url,
                    "status": resp.status_code,
                    "extractor": "brave_llm_context",
                    "truncated": False,
                    "length": len(text),
                    "text": text,
                    "data": data,
                }
            else:
                payload = {
                    "query": query,
                    "count": n,
                    "endpoint": BRAVE_LLM_CONTEXT_URL,
                    "status": resp.status_code,
                    "data": data,
                }
            return json.dumps(payload)
        except Exception as e:
            return json.dumps({"error": str(e), "url": BRAVE_LLM_CONTEXT_URL})

    async def _request_context(self, params: dict[str, Any], headers: dict[str, str]):
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                BRAVE_LLM_CONTEXT_URL,
                params=params,
                headers=headers,
                timeout=20.0,
            )
            # Fallback only for method mismatch; 404 should fail explicitly.
            if resp.status_code == 405:
                resp = await client.post(
                    BRAVE_LLM_CONTEXT_URL,
                    json=params,
                    headers={**headers, "Content-Type": "application/json"},
                    timeout=20.0,
                )
            resp.raise_for_status()
            return resp

    def _extract_context_text(self, data: Any) -> str:
        if isinstance(data, dict):
            if isinstance(data.get("context"), str) and data["context"].strip():
                return data["context"].strip()
            if isinstance(data.get("summary"), str) and data["summary"].strip():
                return data["summary"].strip()
            snippets: list[str] = []
            for key in ("results", "contexts", "documents", "items"):
                value = data.get(key)
                if not isinstance(value, list):
                    continue
                for item in value:
                    if not isinstance(item, dict):
                        continue
                    for field in ("snippet", "text", "content", "description"):
                        part = item.get(field)
                        if isinstance(part, str) and part.strip():
                            snippets.append(part.strip())
                    extra = item.get("extra_snippets")
                    if isinstance(extra, list):
                        for part in extra:
                            if isinstance(part, str) and part.strip():
                                snippets.append(part.strip())
            if snippets:
                return _normalize("\n\n".join(snippets))
        return json.dumps(data, ensure_ascii=False)


class WebFetchTool(Tool):
    """Fetch and extract content from a URL using Readability."""
    
    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML → markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100}
        },
        "required": ["url"]
    }
    
    def __init__(self, max_chars: int = 50000):
        self.max_chars = max_chars
    
    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:
        from readability import Document

        max_chars = maxChars or self.max_chars

        # Validate URL before fetching
        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return json.dumps({"error": f"URL validation failed: {error_msg}", "url": url})

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=30.0
            ) as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()
            
            ctype = r.headers.get("content-type", "")
            
            # JSON
            if "application/json" in ctype:
                text, extractor = json.dumps(r.json(), indent=2), "json"
            # HTML
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                doc = Document(r.text)
                content = self._to_markdown(doc.summary()) if extractMode == "markdown" else _strip_tags(doc.summary())
                text = f"# {doc.title()}\n\n{content}" if doc.title() else content
                extractor = "readability"
            else:
                text, extractor = r.text, "raw"
            
            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]
            
            return json.dumps({"url": url, "finalUrl": str(r.url), "status": r.status_code,
                              "extractor": extractor, "truncated": truncated, "length": len(text), "text": text})
        except Exception as e:
            return json.dumps({"error": str(e), "url": url})
    
    def _to_markdown(self, html: str) -> str:
        """Convert HTML to markdown."""
        # Convert links, headings, lists before stripping tags
        text = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
                      lambda m: f'[{_strip_tags(m[2])}]({m[1]})', html, flags=re.I)
        text = re.sub(r'<h([1-6])[^>]*>([\s\S]*?)</h\1>',
                      lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n', text, flags=re.I)
        text = re.sub(r'<li[^>]*>([\s\S]*?)</li>', lambda m: f'\n- {_strip_tags(m[1])}', text, flags=re.I)
        text = re.sub(r'</(p|div|section|article)>', '\n\n', text, flags=re.I)
        text = re.sub(r'<(br|hr)\s*/?>', '\n', text, flags=re.I)
        return _normalize(_strip_tags(text))
