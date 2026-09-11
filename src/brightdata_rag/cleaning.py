from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag
from markdownify import markdownify

from .urls import normalize_url


def clean_html(source: str, url: str) -> tuple[str, str, list[str]]:
    """Extract main content, preserving Markdown structure and all discovery links."""
    soup = BeautifulSoup(source, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    base = soup.find("base", href=True)
    base_url = urljoin(url, str(base.get("href"))) if isinstance(base, Tag) else url
    links: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        if not isinstance(anchor, Tag):
            continue
        try:
            link = normalize_url(urljoin(base_url, str(anchor.get("href"))))
        except ValueError:
            anchor.attrs.pop("href", None)
            continue
        if link not in seen:
            seen.add(link)
            links.append(link)
        anchor["href"] = link
    for node in soup.select("script, style, noscript, svg, nav, footer, aside, head"):
        node.decompose()
    for node in soup.select('[hidden], [aria-hidden="true"]'):
        node.decompose()
    content = soup.find("main") or soup.select_one('[role="main"]') or soup.find("article")
    content = content or soup.body or soup
    markdown = markdownify(str(content), heading_style="ATX", bullets="-", strip=["img"]).strip()
    return title, markdown, links
