"""
agents/browser_agent.py
Groq-backed Web Research Agent for LocalMind-RAG (Phase 11).

Provides:
- WebSearchProvider abstraction (default: DuckDuckGoSearchProvider)
- Safe URL fetching with SSRF prevention (rejecting private/loopback addresses)
- Groq-based factual evidence extraction from retrieved web content
- Structured WebEvidence chunk construction for P3 Researcher
"""

import ipaddress
import logging
import os
import re
import urllib.parse
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ============================================================================
# 1. Data Models
# ============================================================================

@dataclass
class WebEvidence:
    """Structured evidence retrieved and verified from a web source."""
    title: str
    url: str
    snippet: str
    content: str
    query: str
    source_type: str = "web"
    domain: str = field(default="")

    def __post_init__(self):
        if not self.domain and self.url:
            try:
                parsed = urllib.parse.urlparse(self.url)
                self.domain = parsed.netloc or ""
            except Exception:
                self.domain = ""


# ============================================================================
# 2. Web Search Provider Abstraction
# ============================================================================

class WebSearchProvider(ABC):
    """Abstract interface for external search providers."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Execute web search for query.
        Returns a list of dicts: [{"title": str, "url": str, "snippet": str}].
        """
        pass


class DuckDuckGoSearchProvider(WebSearchProvider):
    """DuckDuckGo HTML search provider using httpx and BeautifulSoup."""

    def __init__(self, timeout: float = 6.0):
        self.timeout = timeout
        self.endpoint = "https://html.duckduckgo.com/html/"

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        headers = {
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        data = {"q": query}

        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                resp = client.post(self.endpoint, data=data, headers=headers)
                if resp.status_code != 200:
                    logger.warning(f"[BrowserAgent] DDG search returned status {resp.status_code}")
                    return []

                soup = BeautifulSoup(resp.text, "html.parser")
                results: List[Dict[str, str]] = []

                for result in soup.find_all("div", class_="result"):
                    title_elem = result.find("a", class_="result__a")
                    snippet_elem = result.find("a", class_="result__snippet")

                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    raw_url = title_elem.get("href", "")

                    # Clean DuckDuckGo redirect wrapper if present
                    parsed_url = self._clean_ddg_url(raw_url)
                    if not parsed_url or not _is_safe_public_url(parsed_url):
                        continue

                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                    results.append({
                        "title": title,
                        "url": parsed_url,
                        "snippet": snippet,
                    })

                    if len(results) >= max_results:
                        break

                return results

        except Exception as e:
            logger.warning(f"[BrowserAgent] DDG search failed: {e}")
            return []

    def _clean_ddg_url(self, raw_url: str) -> str:
        """Extract destination URL from DuckDuckGo redirect URL."""
        if not raw_url:
            return ""
        if "uddg=" in raw_url:
            match = re.search(r"uddg=([^&]+)", raw_url)
            if match:
                return urllib.parse.unquote(match.group(1))
        if raw_url.startswith("//"):
            return "https:" + raw_url
        return raw_url


# ============================================================================
# 3. Security & Safe Page Fetching
# ============================================================================

def _is_safe_public_url(url: str) -> bool:
    """
    Validate that the URL is public http/https and does not target
    private IP ranges, localhost, or metadata endpoints (SSRF prevention).
    """
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Block localhost / link-local / cloud metadata
        blocked_hosts = {
            "localhost",
            "127.0.0.1",
            "0.0.0.0",
            "::1",
            "169.254.169.254",  # AWS/GCP metadata
            "metadata.google.internal",
        }
        if hostname.lower() in blocked_hosts or hostname.endswith(".local"):
            return False

        # If hostname is an IP, verify it is globally reachable
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except ValueError:
            # Not an IP string, hostname is domain name
            pass

        return True
    except Exception:
        return False


def fetch_page_content(url: str, timeout: float = 6.0, max_chars: int = 2500) -> Optional[str]:
    """
    Safely retrieve page content from a public URL.
    Strips scripts, styles, and non-content tags.
    """
    if not _is_safe_public_url(url):
        logger.warning(f"[BrowserAgent] Rejected unsafe or non-public URL: {url}")
        return None

    headers = {"User-Agent": DEFAULT_USER_AGENT}
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code != 200:
                logger.debug(f"[BrowserAgent] Fetch {url} returned HTTP {resp.status_code}")
                return None

            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
                tag.decompose()

            text = " ".join(soup.stripped_strings)
            return text[:max_chars].strip() if text else None
    except Exception as e:
        logger.debug(f"[BrowserAgent] Failed to fetch {url}: {e}")
        return None


# ============================================================================
# 4. Groq-Based Evidence Extraction
# ============================================================================

def extract_evidence_with_groq(
    query: str,
    title: str,
    url: str,
    raw_content: str,
    snippet: str,
    api_key: str,
    model: str,
) -> Optional[str]:
    """
    Call Groq to extract concise, strictly grounded evidence answering the research query
    from the retrieved external content. Groq extracts from actual content; it does not invent facts.
    """
    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        evidence_text = raw_content if raw_content and len(raw_content) > 50 else snippet
        if not evidence_text:
            return None

        prompt = f"""\
You are an objective web research evidence extractor.
Given the user's research query and text retrieved from an external webpage, extract key factual evidence directly relevant to the query.

Rules:
1. Extract only facts directly stated in the retrieved text.
2. Do not add outside knowledge or extrapolate.
3. Be concise and factual (2-4 clear sentences or bullet points).
4. Do not output conversational filler or greeting.

Research Query: {query}
Source Title: {title}
Source URL: {url}
Retrieved Webpage Content:
\"\"\"{evidence_text[:2000]}\"\"\"

Factual Evidence:"""

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise evidence extraction assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=250,
        )

        extracted = response.choices[0].message.content
        return extracted.strip() if extracted else snippet
    except Exception as e:
        logger.warning(f"[BrowserAgent] Groq evidence extraction failed: {e}")
        # Graceful fallback: return the raw snippet rather than failing the source
        return snippet or raw_content[:300]


# ============================================================================
# 5. High-Level Research Orchestrator
# ============================================================================

def research(
    query: str,
    provider: Optional[WebSearchProvider] = None,
    max_sources: int = 3,
) -> List[Dict[str, Any]]:
    """
    Execute external web research for the P3 Researcher agent.

    Workflow:
    1. Verify GROQ_API_KEY and GROQ_WEB_MODEL configuration.
    2. Search the web using provider (default: DuckDuckGoSearchProvider).
    3. Safely fetch top page excerpts.
    4. Extract grounded evidence using Groq.
    5. Return standardized chunk dictionaries with complete provenance.

    If Groq configuration is missing or search fails, returns [] without crashing.
    """
    # 1. Configuration check
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.warning("[BrowserAgent] GROQ_API_KEY not configured. Web research is unavailable.")
        return []

    groq_model = os.getenv("GROQ_WEB_MODEL") or os.getenv("GROQ_MODEL")
    if not groq_model:
        logger.warning("[BrowserAgent] Neither GROQ_WEB_MODEL nor GROQ_MODEL is configured. Web research is unavailable.")
        return []

    # 2. Search provider
    search_provider = provider or DuckDuckGoSearchProvider()
    search_results = search_provider.search(query, max_results=max_sources)
    if not search_results:
        logger.info(f"[BrowserAgent] No search results found for: '{query[:60]}'")
        return []

    logger.info(f"[BrowserAgent] Found {len(search_results)} search results for: '{query[:60]}'")

    # 3. Retrieve content & extract evidence
    evidence_chunks: List[Dict[str, Any]] = []

    for item in search_results:
        title = item.get("title") or "Web Source"
        url = item.get("url") or ""
        snippet = item.get("snippet") or ""

        # Fetch actual page content
        page_content = fetch_page_content(url) if url else None

        # Extract grounded evidence via Groq
        evidence_text = extract_evidence_with_groq(
            query=query,
            title=title,
            url=url,
            raw_content=page_content or "",
            snippet=snippet,
            api_key=groq_api_key,
            model=groq_model,
        )

        if not evidence_text:
            continue

        domain = ""
        if url:
            try:
                domain = urllib.parse.urlparse(url).netloc
            except Exception:
                domain = ""

        chunk = {
            "node_id": f"web_{uuid.uuid4().hex[:8]}",
            "text": evidence_text,
            "score": 0.85,
            "metadata": {
                "filename": title,
                "title": title,
                "url": url,
                "domain": domain,
                "source_type": "web",
                "snippet": snippet or evidence_text[:200],
                "doc_category": "web",
                "query": query,
            },
            "source": "web_search",
        }
        evidence_chunks.append(chunk)

    logger.info(f"[BrowserAgent] Produced {len(evidence_chunks)} structured web evidence chunks.")
    return evidence_chunks
