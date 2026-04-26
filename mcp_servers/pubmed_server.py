"""MCP server — PubMed E-utilities tools.

Exposes 4 tools over the NCBI PubMed E-utilities API
(https://eutils.ncbi.nlm.nih.gov/entrez/eutils/):

    search_trial_publications   → papers mentioning an NCT ID
    search_publications         → general keyword search
    get_publication_details     → full abstract + MeSH + pub type
    search_author_publications  → recent papers by an author

Response envelope matches aact_server: ``status``, ``tool``,
``source: "pubmed"``, ``timestamp``. Tools never crash — any exception
is caught and returned as a structured error.

Caching: every HTTP call is keyed by ``hashlib.md5`` of its endpoint
and params and stored under ``mcp_servers/cache/`` so repeated calls
with the same arguments are free and rate-limit-friendly.

Rate limit: PubMed allows 3 req/sec without an API key; we sleep
``RATE_LIMIT_SLEEP_SEC`` seconds before every *live* (uncached) call.

Run as a stdio server:
    venv/bin/python -m mcp_servers.pubmed_server
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

load_dotenv()

log = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
CACHE_DIR: Path = PROJECT_ROOT / "mcp_servers" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SERVER_NAME = "pubmed"
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RATE_LIMIT_SLEEP_SEC = 0.35  # 3 req/sec ceiling per PubMed E-utilities ToS
HTTP_TIMEOUT_SEC = 15
ABSTRACT_SNIPPET_CHARS = 300
MAX_RESULTS_CEILING = 50     # safety cap on any max_results argument


# ---------------------------------------------------------------------------
# Response envelope helpers (parallel aact_server.py)
# ---------------------------------------------------------------------------


def _error(tool: str, reason: str, **extra: Any) -> dict:
    """Structured error payload — every tool returns this on failure."""
    payload = {
        "status": "error",
        "tool": tool,
        "reason": reason,
        "source": SERVER_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(extra)
    return payload


def _ok(tool: str, **payload: Any) -> dict:
    """Structured success payload."""
    return {
        "status": "ok",
        "tool": tool,
        "source": SERVER_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }


# ---------------------------------------------------------------------------
# HTTP + cache layer
# ---------------------------------------------------------------------------


def _hash_key(data: Any) -> str:
    """Stable hash over an arbitrary JSON-serialisable object."""
    blob = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(blob).hexdigest()


def _cache_path(endpoint: str, params: dict) -> Path:
    return CACHE_DIR / f"pubmed_{endpoint}_{_hash_key(params)}.json"


def _http_get(endpoint: str, params: dict) -> str:
    """GET from a PubMed endpoint with caching + rate limiting.

    Cache is keyed on ``(endpoint, params)``. On cache miss we sleep
    ``RATE_LIMIT_SLEEP_SEC`` seconds before the live call so back-to-back
    tool invocations stay under the 3 req/sec ceiling.

    Returns the raw response body as text. Raises on HTTP / network
    failure — callers are responsible for turning that into a structured
    error payload.
    """
    path = _cache_path(endpoint, params)
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            log.debug("pubmed cache hit: %s", path.name)
            return cached["body"]
        except Exception as exc:  # noqa: BLE001 — fall through to live call
            log.warning("cache read failed (%s), refetching: %s", path.name, exc)

    time.sleep(RATE_LIMIT_SLEEP_SEC)
    url = f"{BASE_URL}/{endpoint}.fcgi"
    log.info("pubmed live call: %s params=%s", endpoint, params)
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT_SEC)
    r.raise_for_status()
    body = r.text

    try:
        path.write_text(json.dumps({"body": body}))
    except Exception as exc:  # noqa: BLE001 — cache write failure is non-fatal
        log.warning("cache write failed (%s): %s", path.name, exc)

    return body


# ---------------------------------------------------------------------------
# Low-level PubMed wrappers (esearch / esummary / efetch)
# ---------------------------------------------------------------------------


def _esearch(term: str, max_results: int) -> dict:
    """Run an E-utilities esearch and return the parsed JSON envelope."""
    params = {
        "db": "pubmed",
        "term": term,
        "retmax": int(max_results),
        "retmode": "json",
    }
    return json.loads(_http_get("esearch", params))


def _esummary(pmids: list[str]) -> dict:
    """Run an esummary for a list of PMIDs; returns parsed JSON envelope."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
    }
    return json.loads(_http_get("esummary", params))


def _efetch_xml(pmids: list[str]) -> ET.Element:
    """Fetch XML article records for one or more PMIDs."""
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
    }
    return ET.fromstring(_http_get("efetch", params))


def _extract_abstract(article: ET.Element) -> str:
    """Concatenate all AbstractText fragments from a PubmedArticle node."""
    parts: list[str] = []
    for text_node in article.iter("AbstractText"):
        label = text_node.attrib.get("Label")
        body = (text_node.text or "").strip()
        if not body:
            continue
        parts.append(f"{label}: {body}" if label else body)
    return " ".join(parts).strip()


def _fetch_abstract_snippets(pmids: list[str]) -> dict[str, str]:
    """Return ``{pmid: snippet}`` truncated to ABSTRACT_SNIPPET_CHARS."""
    if not pmids:
        return {}
    try:
        root = _efetch_xml(pmids)
    except Exception as exc:  # noqa: BLE001 — abstracts are best-effort
        log.warning("efetch failed, skipping abstract snippets: %s", exc)
        return {}

    out: dict[str, str] = {}
    for article in root.iter("PubmedArticle"):
        pmid_elem = article.find(".//PMID")
        if pmid_elem is None or not pmid_elem.text:
            continue
        abstract = _extract_abstract(article)
        if len(abstract) > ABSTRACT_SNIPPET_CHARS:
            abstract = abstract[:ABSTRACT_SNIPPET_CHARS].rstrip() + "…"
        out[pmid_elem.text] = abstract
    return out


def _summaries_to_records(
    esummary_payload: dict, pmids: list[str],
    snippets: Optional[dict[str, str]] = None,
) -> list[dict]:
    """Convert an esummary JSON + abstract snippets → flat record list."""
    snippets = snippets or {}
    result = esummary_payload.get("result", {}) if isinstance(
        esummary_payload, dict
    ) else {}
    records: list[dict] = []
    for pmid in pmids:
        doc = result.get(pmid)
        if not doc:
            continue
        authors = [
            a.get("name", "") for a in doc.get("authors", [])
            if isinstance(a, dict) and a.get("name")
        ]
        records.append({
            "pmid": pmid,
            "title": doc.get("title"),
            "authors": authors,
            "journal": doc.get("fulljournalname") or doc.get("source"),
            "date": doc.get("pubdate"),
            "abstract_snippet": snippets.get(pmid, ""),
        })
    return records


def _cap_results(n: int) -> int:
    """Bound ``max_results`` to MAX_RESULTS_CEILING (and to a sane floor)."""
    return max(1, min(int(n), MAX_RESULTS_CEILING))


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def tool_search_trial_publications(
    nct_id: str, max_results: int = 10,
) -> dict:
    """Return PubMed papers that mention the given NCT ID."""
    tool = "search_trial_publications"
    try:
        capped = _cap_results(max_results)
        term = f"{nct_id}[All Fields]"
        es = _esearch(term, capped)
        pmids = es.get("esearchresult", {}).get("idlist", [])
        total = int(es.get("esearchresult", {}).get("count", len(pmids)))

        snippets = _fetch_abstract_snippets(pmids) if pmids else {}
        summaries = _esummary(pmids) if pmids else {}
        publications = _summaries_to_records(summaries, pmids, snippets)

        return _ok(
            tool,
            nct_id=nct_id,
            query=term,
            total_found=total,
            results_returned=len(publications),
            max_results=capped,
            publications=publications,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("%s failed for %s: %s", tool, nct_id, exc)
        return _error(tool, str(exc), nct_id=nct_id)


def tool_search_publications(query: str, max_results: int = 10) -> dict:
    """Return PubMed papers matching a free-text query (keywords, drug, etc.)."""
    tool = "search_publications"
    try:
        if not query or not query.strip():
            return _error(tool, "query is empty")
        capped = _cap_results(max_results)
        es = _esearch(query, capped)
        pmids = es.get("esearchresult", {}).get("idlist", [])
        total = int(es.get("esearchresult", {}).get("count", len(pmids)))

        snippets = _fetch_abstract_snippets(pmids) if pmids else {}
        summaries = _esummary(pmids) if pmids else {}
        publications = _summaries_to_records(summaries, pmids, snippets)

        return _ok(
            tool,
            query=query,
            total_found=total,
            results_returned=len(publications),
            max_results=capped,
            publications=publications,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("%s failed for query=%r: %s", tool, query, exc)
        return _error(tool, str(exc), query=query)


def tool_get_publication_details(pmid: str) -> dict:
    """Return the full abstract, MeSH terms, and publication types for a PMID."""
    tool = "get_publication_details"
    try:
        if not pmid or not str(pmid).strip():
            return _error(tool, "pmid is empty")

        root = _efetch_xml([str(pmid)])
        article = next(root.iter("PubmedArticle"), None)
        if article is None:
            return _error(tool, "article not found", pmid=pmid)

        title_elem = article.find(".//ArticleTitle")
        journal_elem = article.find(".//Journal/Title")
        date_elem = article.find(".//PubDate")
        pub_date = (
            " ".join(
                (c.text or "").strip() for c in date_elem
                if c is not None and (c.text or "").strip()
            )
            if date_elem is not None else None
        )
        authors = []
        for a in article.iter("Author"):
            last = a.findtext("LastName") or ""
            init = a.findtext("Initials") or ""
            name = f"{last} {init}".strip()
            if name:
                authors.append(name)
        mesh_terms = [
            d.text for d in article.iter("DescriptorName")
            if d is not None and d.text
        ]
        pub_types = [
            p.text for p in article.iter("PublicationType")
            if p is not None and p.text
        ]
        abstract = _extract_abstract(article)

        return _ok(
            tool,
            pmid=str(pmid),
            title=title_elem.text if title_elem is not None else None,
            journal=journal_elem.text if journal_elem is not None else None,
            date=pub_date,
            authors=authors,
            abstract=abstract,
            mesh_terms=mesh_terms,
            publication_types=pub_types,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("%s failed for pmid=%s: %s", tool, pmid, exc)
        return _error(tool, str(exc), pmid=pmid)


def tool_search_author_publications(
    author_name: str, years: int = 3, max_results: int = 10,
) -> dict:
    """Return recent papers by a named author within a year-range window."""
    tool = "search_author_publications"
    try:
        if not author_name or not author_name.strip():
            return _error(tool, "author_name is empty")
        years = max(1, int(years))
        capped = _cap_results(max_results)

        end_year = datetime.now().year
        start_year = end_year - years
        term = (
            f'"{author_name}"[Author] AND '
            f'("{start_year}"[PDAT] : "{end_year}"[PDAT])'
        )
        es = _esearch(term, capped)
        pmids = es.get("esearchresult", {}).get("idlist", [])
        total = int(es.get("esearchresult", {}).get("count", len(pmids)))

        snippets = _fetch_abstract_snippets(pmids) if pmids else {}
        summaries = _esummary(pmids) if pmids else {}
        publications = _summaries_to_records(summaries, pmids, snippets)

        return _ok(
            tool,
            author_name=author_name,
            years=years,
            start_year=start_year,
            end_year=end_year,
            query=term,
            total_found=total,
            results_returned=len(publications),
            max_results=capped,
            publications=publications,
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("%s failed for %r: %s", tool, author_name, exc)
        return _error(tool, str(exc), author_name=author_name)


# ---------------------------------------------------------------------------
# MCP server wiring
# ---------------------------------------------------------------------------

server: Server = Server(SERVER_NAME)

TOOL_DISPATCH: dict[str, Callable[..., dict]] = {
    "search_trial_publications":  tool_search_trial_publications,
    "search_publications":        tool_search_publications,
    "get_publication_details":    tool_get_publication_details,
    "search_author_publications": tool_search_author_publications,
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Declare PubMed tools exposed to the agent."""
    return [
        Tool(
            name="search_trial_publications",
            description=(
                "Search PubMed for papers that mention a specific NCT ID. "
                "Useful for cross-validating AACT results against published "
                "literature (check F)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "nct_id": {"type": "string", "description": "NCT ID."},
                    "max_results": {
                        "type": "integer",
                        "description": "Max publications (default 10, cap 50).",
                        "default": 10,
                    },
                },
                "required": ["nct_id"],
            },
        ),
        Tool(
            name="search_publications",
            description=(
                "Search PubMed by free-text query (keywords, drug name, "
                "condition, etc.). Returns summary records."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "PubMed search term.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max publications (default 10, cap 50).",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_publication_details",
            description=(
                "Return full abstract, MeSH terms, and publication types "
                "for a single PMID."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pmid": {"type": "string", "description": "PubMed ID."},
                },
                "required": ["pmid"],
            },
        ),
        Tool(
            name="search_author_publications",
            description=(
                "Return recent papers by an author within the last N years. "
                "Use to check whether a trial PI published results elsewhere."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "author_name": {
                        "type": "string",
                        "description": "Author name, e.g. 'Smith J'.",
                    },
                    "years": {
                        "type": "integer",
                        "description": "Year window size (default 3).",
                        "default": 3,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max publications (default 10, cap 50).",
                        "default": 10,
                    },
                },
                "required": ["author_name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch tool invocation; always return a single TextContent."""
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        payload = _error(name, f"unknown tool {name!r}")
    else:
        try:
            payload = fn(**(arguments or {}))
        except TypeError as exc:
            log.warning("%s called with bad arguments %s: %s",
                        name, arguments, exc)
            payload = _error(name, f"invalid arguments: {exc}")
        except Exception as exc:  # noqa: BLE001 — never crash the server
            log.exception("%s crashed unexpectedly: %s", name, exc)
            payload = _error(name, str(exc))
    return [TextContent(type="text", text=json.dumps(payload, default=str))]


async def _amain() -> None:
    """Serve MCP over stdio until the client disconnects."""
    log.info("Starting PubMed MCP server (stdio). Cache at %s", CACHE_DIR)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main() -> None:
    """Entry point: ``python -m mcp_servers.pubmed_server``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
