"""
Fetches pages discovered by the crawler, cleans HTML, writes .txt to data/raw/{collection}/.
Idempotent — already-fetched pages are skipped.
"""

from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "rune-rag/1.0"}
_STRIP = {"nav", "header", "footer", "aside", "script", "style", "noscript"}
_SELECTORS = ["article", "main", "[role='main']", ".content", "#content", ".docs-content", ".markdown-body"]


def _slug(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return path.replace("/", "--") or "index"


def _clean(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(_STRIP):
        tag.decompose()
    node = next(
        (soup.select_one(s) for s in _SELECTORS if soup.select_one(s)),
        soup.find("body") or soup,
    )
    lines = [l.rstrip() for l in node.get_text(separator="\n").splitlines()]
    out, prev_blank = [], False
    for line in lines:
        blank = not line.strip()
        if blank and prev_blank:
            continue
        out.append(line)
        prev_blank = blank
    return "\n".join(out).strip()


def fetch_all(urls: list[str], raw_dir: Path) -> list[Path]:
    """Fetch a list of URLs and save each as a .txt file. Returns saved paths."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    print(f"[fetcher] {len(urls)} pages to fetch")
    for url in urls:
        dest = raw_dir / f"{_slug(url)}.txt"
        if dest.exists():
            saved.append(dest)
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            content = _clean(r.text)
            if not content:
                print(f"  [skip-empty] {url}")
                continue
            dest.write_text(content, encoding="utf-8")
            saved.append(dest)
            print(f"  [saved] {dest.name}  ({len(content):,} chars)")
        except requests.RequestException as e:
            print(f"  [error] {url}: {e}")
    return saved
