"""Web search and scraping tools used by the ResearchMind agents.

The public LangChain tools (``web_search`` and ``scrape_url``) keep their
original behaviour.  The additional helpers below expose the raw text plus
source metadata that the vector RAG layer needs.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Iterable
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

from .evaluation import content_to_text

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

BROWSER_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchMind/1.0)"}
DEFAULT_TIMEOUT = float(os.getenv("SCRAPE_TIMEOUT_SECONDS", "10"))
DEFAULT_MAX_SCRAPE_CHARS = int(os.getenv("MAX_SCRAPE_CHARS", "20000"))
AGENT_SCRAPE_CHARS = 3000


@dataclass
class ScrapedPage:
    """A single scraped web page with provenance metadata."""

    url: str
    title: str
    text: str = ""
    ok: bool = True
    error: str | None = None

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["char_count"] = self.char_count
        # Never send the full page text back over the API by default.
        data.pop("text", None)
        return data


def canonical_url(url: str) -> str:
    """Normalise a URL so references can be compared reliably."""

    url = (url or "").strip().rstrip(".,;:!?)]}\"'`*")
    try:
        parsed = urlparse(url)
    except ValueError:
        return url.lower()

    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/") or "/"
    scheme = "https" if parsed.scheme.lower() in {"", "http", "https"} else parsed.scheme.lower()
    return urlunparse((scheme, host, path, "", parsed.query, ""))


def extract_urls(text: str) -> list[str]:
    """Extract unique URLs from search-agent output or free text."""

    text = content_to_text(text)
    if not text:
        return []

    candidates = re.findall(r"https?://[^\s<>\"'\]\)]+", text)
    seen: set[str] = set()
    urls: list[str] = []
    for candidate in candidates:
        cleaned = candidate.rstrip(".,;:!?)]}\"'`*")
        key = canonical_url(cleaned)
        if key and key not in seen:
            seen.add(key)
            urls.append(cleaned)
    return urls


def fetch_page(
    url: str,
    *,
    max_chars: int = DEFAULT_MAX_SCRAPE_CHARS,
    timeout: float = DEFAULT_TIMEOUT,
) -> ScrapedPage:
    """Fetch a URL with Requests and return clean text extracted by BeautifulSoup."""

    try:
        response = requests.get(url, timeout=timeout, headers=BROWSER_HEADERS)
        response.raise_for_status()
    except Exception as exc:  # network/HTTP errors are expected on the open web
        return ScrapedPage(url=url, title=url, ok=False, error=str(exc)[:300])

    soup = BeautifulSoup(response.text, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else ""

    for tag in soup(
        ["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "iframe", "svg"]
    ):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text).strip()

    if not text:
        return ScrapedPage(url=url, title=title or url, ok=False, error="No extractable text")

    return ScrapedPage(url=url, title=title or url, text=text[:max_chars])


def _safe_fetch(url: str, max_chars: int) -> ScrapedPage:
    try:
        return fetch_page(url, max_chars=max_chars)
    except Exception as exc:  # defensive: fetch_page already handles most failures
        return ScrapedPage(url=url, title=url, ok=False, error=str(exc)[:300])


def scrape_urls(
    urls: Iterable[str],
    *,
    max_pages: int = 3,
    max_chars: int = DEFAULT_MAX_SCRAPE_CHARS,
    max_workers: int = 4,
) -> list[ScrapedPage]:
    """Scrape several URLs concurrently, preserving input order."""

    unique: list[str] = []
    seen: set[str] = set()
    for url in urls:
        key = canonical_url(url)
        if key and key not in seen:
            seen.add(key)
            unique.append(url)

    selected = unique[: max(0, max_pages)]
    if not selected:
        return []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(selected))) as executor:
        return list(executor.map(lambda u: _safe_fetch(u, max_chars), selected))


def tavily_search(query: str, max_results: int = 8) -> list[dict]:
    """Return structured Tavily results (the deterministic source pool)."""

    response = tavily.search(query=query, max_results=max_results)
    results: list[dict] = []
    for item in response.get("results", []):
        url = item.get("url") or ""
        if not url:
            continue
        results.append(
            {
                "title": item.get("title") or url,
                "url": url,
                "snippet": (item.get("content") or "")[:500],
            }
        )
    return results


def collect_urls_from_messages(messages: Iterable[Any]) -> list[str]:
    """Collect URLs from every agent message, including tool outputs.

    The search agent's final answer often omits URLs even though its Tavily tool
    output contained all of them. Reading every message is what prevents the
    one-source bottleneck.
    """

    seen: set[str] = set()
    urls: list[str] = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        for url in extract_urls(content):
            key = canonical_url(url)
            if key and key not in seen:
                seen.add(key)
                urls.append(url)
    return urls


@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on a topic . Returns Titles , URLs and snippets."""
    results = tavily_search(query=query, max_results=8)

    out = []

    for r in results:
        out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['snippet'][:300]}\n")

    return "\n----\n".join(out)


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    page = fetch_page(url)
    if not page.ok:
        return f"Could not scrape URL: {page.error}"
    return page.text[:AGENT_SCRAPE_CHARS]
