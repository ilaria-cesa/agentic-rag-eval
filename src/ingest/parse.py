"""Turn a 10-K HTML document into plain text and extract the sections we index.

Limitations (documented on purpose):
  * tables are flattened to text, so numeric tables lose their layout;
  * section detection relies on "Item X." headings, so unusual filings may need
    manual checks. `extract_sections` returns warnings instead of failing silently.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

# Items we keep. Item 1 = business, 1A = risk factors, 7 = MD&A.
WANTED_ITEMS = {"1": "business", "1a": "risk_factors", "7": "mdna"}
MIN_SECTION_CHARS = 2000

_BLOCK_TAGS = ["p", "div", "br", "li", "tr", "table", "h1", "h2", "h3", "h4", "h5", "h6"]
_HEADING_RE = re.compile(
    r"^[ \t]*item[ \t]+(\d{1,2}[ab]?)[ \t]*[\.:\-–—]?[ \t]*(.{0,100})$",
    re.IGNORECASE | re.MULTILINE,
)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()
    # Newlines only at block boundaries, so inline links stay inside their sentence.
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    for tag in soup.find_all(["td", "th"]):
        tag.insert_after(" ")
    text = soup.get_text()
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_sections(text: str) -> tuple[dict[str, str], list[str]]:
    """Return ({section_name: text}, warnings).

    Every "Item X" heading starts a candidate segment that ends at the next heading
    of a different item. The table of contents produces tiny candidates, so for each
    wanted item we keep the longest candidate.
    """
    headings = [(m.start(), m.end(), m.group(1).lower()) for m in _HEADING_RE.finditer(text)]
    best: dict[str, str] = {}
    for i, (_, end, key) in enumerate(headings):
        if key not in WANTED_ITEMS:
            continue
        stop = len(text)
        for next_start, _, next_key in headings[i + 1 :]:
            if next_key != key:
                stop = next_start
                break
        segment = text[end:stop].strip()
        name = WANTED_ITEMS[key]
        if len(segment) > len(best.get(name, "")):
            best[name] = segment

    warnings = []
    for name in WANTED_ITEMS.values():
        length = len(best.get(name, ""))
        if length == 0:
            warnings.append(f"missing section: {name}")
        elif length < MIN_SECTION_CHARS:
            warnings.append(f"short section: {name} ({length} chars)")
    return best, warnings
