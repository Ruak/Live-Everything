"""Web search tool — 当本地 RAG 命中不足时的兜底。

设计目标：
* 无需第三方 API key：默认使用 DuckDuckGo HTML 接口；失败再回退到 Bing。
* 永远不抛异常，返回空列表即代表「搜不到」。
* 结果结构化为 ``WebSearchResult``，上游 ``agent_manager`` 会把它渲染成
  ``[联网补充知识]`` 段落注入到 LLM 的 ``knowledge_context``。
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import parse_qs, quote_plus, urlparse

import httpx

from .. import config

logger = logging.getLogger(__name__)


DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DDG_LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"
BING_ENDPOINT = "https://www.bing.com/search"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str

    def to_markdown(self) -> str:
        snippet = self.snippet.strip() or "(无摘要)"
        return f"- 【{self.title}】{snippet}（来源：{self.url}）"


class WebSearcher:
    """HTTP 检索器；在运行期保留一个 AsyncClient 以便复用连接。"""

    def __init__(
        self,
        *,
        provider: str = config.WEB_SEARCH_PROVIDER,
        timeout: float = config.WEB_SEARCH_TIMEOUT_SECONDS,
        top_k: int = config.WEB_SEARCH_TOP_K,
    ):
        self.provider = (provider or "duckduckgo").lower()
        self.timeout = timeout
        self.top_k = max(1, top_k)
        self._client = httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": BROWSER_UA},
        )
        self._healthy: Optional[bool] = None

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── public API ──────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
    ) -> List[WebSearchResult]:
        q = (query or "").strip()
        if not q:
            return []
        n = top_k or self.top_k

        # 依次尝试：主选 → DDG lite → Bing。只要一家出结果就返回。
        if self.provider == "bing":
            order = ["bing", "duckduckgo", "duckduckgo_lite"]
        else:
            order = ["duckduckgo", "duckduckgo_lite", "bing"]

        for name in order:
            try:
                if name == "duckduckgo":
                    items = await self._search_duckduckgo(q, n)
                elif name == "duckduckgo_lite":
                    items = await self._search_duckduckgo_lite(q, n)
                elif name == "bing":
                    items = await self._search_bing(q, n)
                else:
                    continue
                if items:
                    self._healthy = True
                    return items[:n]
            except Exception as exc:
                logger.warning("Web search via %s failed: %s", name, exc)

        self._healthy = False
        return []

    async def health_check(self) -> bool:
        """一次轻量查询；结果缓存在 ``self._healthy``。"""
        if self._healthy is not None:
            return self._healthy
        results = await self.search("hello", top_k=1)
        return bool(results)

    # ── DuckDuckGo HTML ─────────────────────────────────────────

    async def _search_duckduckgo(self, query: str, top_k: int) -> List[WebSearchResult]:
        # DuckDuckGo 对 POST 往往直接回首页；用 GET + html.duckduckgo.com 最稳定。
        resp = await self._client.get(
            DDG_HTML_ENDPOINT,
            params={"q": query, "kl": "cn-zh"},
            headers={"Referer": "https://duckduckgo.com/"},
        )
        resp.raise_for_status()
        html_text = resp.text

        # 标题：<a class="result__a" href="...">title</a>
        # 摘要：<a class="result__snippet">...</a>  或  <div class="result__snippet">...</div>
        # URL 形如 //duckduckgo.com/l/?uddg=<encoded>
        title_re = re.compile(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        snippet_re = re.compile(
            r'<(?:a|div)[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</(?:a|div)>',
            re.IGNORECASE | re.DOTALL,
        )

        # 逐条匹配：用 title 锚点切块，避免某条缺 snippet 时错位。
        titles = list(title_re.finditer(html_text))
        snippets = list(snippet_re.finditer(html_text))

        results: List[WebSearchResult] = []
        for index, title_match in enumerate(titles):
            url = _unwrap_ddg_url(title_match.group("href"))
            title = _strip_html(title_match.group("title"))
            snippet = ""
            if index < len(snippets):
                snippet = _strip_html(snippets[index].group("snippet"))
            if not url or not title:
                continue
            results.append(WebSearchResult(title=title, url=url, snippet=snippet))
            if len(results) >= top_k:
                break
        return results

    # ── Bing HTML fallback ──────────────────────────────────────

    async def _search_bing(self, query: str, top_k: int) -> List[WebSearchResult]:
        resp = await self._client.get(
            BING_ENDPOINT,
            params={"q": query, "setlang": "zh-CN"},
        )
        resp.raise_for_status()
        html_text = resp.text

        # Bing 经常在中国区返回 JS 渲染页；我们做尽力而为的解析。
        block_re = re.compile(
            r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>(.*?)</li>',
            re.IGNORECASE | re.DOTALL,
        )
        link_re = re.compile(
            r'<h2[^>]*>.*?<a[^>]*href="(?P<href>https?://[^"]+)"[^>]*>(?P<title>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        snippet_re = re.compile(r'<p[^>]*>(?P<snippet>.*?)</p>', re.IGNORECASE | re.DOTALL)

        results: List[WebSearchResult] = []
        for block in block_re.finditer(html_text):
            body = block.group(1)
            link_match = link_re.search(body)
            if not link_match:
                continue
            url = link_match.group("href")
            title = _strip_html(link_match.group("title"))
            snippet_match = snippet_re.search(body)
            snippet = _strip_html(snippet_match.group("snippet")) if snippet_match else ""
            results.append(WebSearchResult(title=title, url=url, snippet=snippet))
            if len(results) >= top_k:
                break
        return results

    # ── DuckDuckGo Lite fallback ────────────────────────────────

    async def _search_duckduckgo_lite(self, query: str, top_k: int) -> List[WebSearchResult]:
        """lite.duckduckgo.com 返回最简表格页，失联率更低。"""
        resp = await self._client.get(
            DDG_LITE_ENDPOINT,
            params={"q": query, "kl": "cn-zh"},
        )
        resp.raise_for_status()
        html_text = resp.text

        # lite 版每条结果是：<a rel="nofollow" href="..." class="result-link">title</a>
        # 接着一个 <td class="result-snippet">...</td>
        link_re = re.compile(
            r'<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            re.IGNORECASE | re.DOTALL,
        )
        snippet_re = re.compile(
            r'<td[^>]*class="[^"]*result-snippet[^"]*"[^>]*>(?P<snippet>.*?)</td>',
            re.IGNORECASE | re.DOTALL,
        )
        links = list(link_re.finditer(html_text))
        snippets = list(snippet_re.finditer(html_text))

        results: List[WebSearchResult] = []
        for index, link_match in enumerate(links):
            url = _unwrap_ddg_url(link_match.group("href"))
            title = _strip_html(link_match.group("title"))
            snippet = (
                _strip_html(snippets[index].group("snippet"))
                if index < len(snippets)
                else ""
            )
            if not url or not title:
                continue
            results.append(WebSearchResult(title=title, url=url, snippet=snippet))
            if len(results) >= top_k:
                break
        return results


# ── helpers ─────────────────────────────────────────────────────


def _unwrap_ddg_url(href: str) -> str:
    """DuckDuckGo 把跳转地址藏在 ``uddg`` query 参数里。"""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = "https://duckduckgo.com" + href

    try:
        parsed = urlparse(href)
        params = parse_qs(parsed.query)
        if "uddg" in params:
            return params["uddg"][0]
    except Exception:
        pass
    return href


_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html(text: str) -> str:
    if not text:
        return ""
    no_tags = _TAG_RE.sub(" ", text)
    unescaped = html.unescape(no_tags)
    return re.sub(r"\s+", " ", unescaped).strip()


def build_web_context(query: str, results: List[WebSearchResult]) -> str:
    """把联网结果渲染成 RAG context 片段。"""
    if not results:
        return ""
    lines = [
        "[联网补充知识]",
        f"（检索词：{query}；仅在本地知识库无法覆盖时可作为辅助依据，请注明来源）",
    ]
    lines.extend(result.to_markdown() for result in results)
    return "\n".join(lines)


__all__ = [
    "WebSearcher",
    "WebSearchResult",
    "build_web_context",
]
