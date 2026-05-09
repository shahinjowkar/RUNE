"""
Smart URL discovery for any docs site.

Priority order:
  1. robots.txt Sitemap: directive
  2. /sitemap.xml, /sitemap-0.xml, /sitemap_index.xml
  3. Recursive link crawl from root URL (fallback)
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "rune-rag/1.0"}
_SITEMAP_CANDIDATES = ["/sitemap.xml", "/sitemap-0.xml", "/sitemap_index.xml"]
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".mp4", ".pdf", ".zip"}


def _origin(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}"


def _is_page(url: str, base_origin: str) -> bool:
    p = urlparse(url)
    if p.netloc and base_origin not in f"{p.scheme}://{p.netloc}":
        return False
    return Path(p.path).suffix.lower() not in _IMAGE_EXTS


def _from_sitemap_xml(content: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(content, "lxml-xml")
    # Handle sitemap index (nested sitemaps)
    sub_sitemaps = [loc.get_text(strip=True) for loc in soup.find_all("loc") if "sitemap" in loc.get_text().lower()]
    if sub_sitemaps:
        urls: list[str] = []
        for sub in sub_sitemaps:
            try:
                r = requests.get(sub, headers=HEADERS, timeout=10)
                if r.status_code == 200:
                    urls.extend(_from_sitemap_xml(r.text, base_url))
            except requests.RequestException:
                pass
        return urls
    return [loc.get_text(strip=True) for loc in soup.find_all("loc")]


def _try_sitemap(base_url: str) -> list[str] | None:
    origin = _origin(base_url)

    # 1. robots.txt
    try:
        r = requests.get(f"{origin}/robots.txt", headers=HEADERS, timeout=8)
        for line in r.text.splitlines():
            if line.lower().startswith("sitemap:"):
                sitemap_url = line.split(":", 1)[1].strip()
                r2 = requests.get(sitemap_url, headers=HEADERS, timeout=10)
                if r2.status_code == 200 and "<url" in r2.text:
                    print(f"[crawler] sitemap found via robots.txt: {sitemap_url}")
                    return _from_sitemap_xml(r2.text, base_url)
    except requests.RequestException:
        pass

    # 2. Common paths
    for path in _SITEMAP_CANDIDATES:
        try:
            r = requests.get(f"{origin}{path}", headers=HEADERS, timeout=10)
            if r.status_code == 200 and ("<url" in r.text or "<sitemap" in r.text):
                print(f"[crawler] sitemap found: {origin}{path}")
                return _from_sitemap_xml(r.text, base_url)
        except requests.RequestException:
            continue

    return None


def _recursive_crawl(base_url: str, max_pages: int = 500) -> list[str]:
    """Follow <a href> links breadth-first, staying on the same origin."""
    print(f"[crawler] no sitemap found — recursive crawl from {base_url}")
    origin = _origin(base_url)
    visited: set[str] = set()
    queue = [base_url]
    found: list[str] = []

    while queue and len(found) < max_pages:
        url = queue.pop(0)
        norm = url.rstrip("/")
        if norm in visited:
            continue
        visited.add(norm)

        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                continue
        except requests.RequestException:
            continue

        found.append(url)
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.find_all("a", href=True):
            href = urljoin(url, a["href"]).split("#")[0]
            if _is_page(href, origin) and href.rstrip("/") not in visited:
                queue.append(href)

    print(f"[crawler] recursive crawl found {len(found)} pages")
    return found


def discover(base_url: str, scope_path: str | None = None) -> list[str]:
    """
    Return all page URLs for the given site.

    Args:
        base_url:   The root URL or docs URL (e.g. https://wiremock.org/docs/).
        scope_path: Optional path prefix to filter results (e.g. '/docs/').
                    Inferred from base_url if not provided.
    """
    if scope_path is None:
        p = urlparse(base_url)
        scope_path = p.path if p.path not in ("", "/") else None

    urls = _try_sitemap(base_url)
    if urls is None:
        urls = _recursive_crawl(base_url)

    origin = _origin(base_url)

    # Filter to pages within the target origin + scope
    filtered = [
        u for u in urls
        if origin in u
        and _is_page(u, origin)
        and (scope_path is None or urlparse(u).path.startswith(scope_path))
    ]

    # Deduplicate
    seen: set[str] = set()
    result: list[str] = []
    for u in filtered:
        key = u.rstrip("/")
        if key not in seen:
            seen.add(key)
            result.append(u)

    print(f"[crawler] {len(result)} unique pages discovered")
    return sorted(result)
