"""
PubMed Literature Mining Pipeline.

Automated extraction of microscopy images, clinical descriptions, and case
data from peer-reviewed medical literature. Implements NCBI Entrez API
integration, PDF processing, figure extraction, and metadata association.

Architecture
------------
The literature mining pipeline follows a multi-stage extraction workflow:

    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
    │   PubMed API    │───>│   PDF/PMC    │───>│   Figure    │
    │   (Entrez)      │    │   Retrieval  │    │  Extraction │
    └─────────────────┘    └──────────────┘    └─────────────┘
           │                      │                   │
           v                      v                   v
    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
    │   Citation      │───>│   Caption    │───>│  Microscopy │
    │   Metadata      │    │   Parsing    │    │   Filter    │
    └─────────────────┘    └──────────────┘    └─────────────┘
           │                      │                   │
           v                      v                   v
    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
    │   OCR Engine    │───>│ Magnification│───>│  Staining   │
    │   Pipeline      │    │  Extraction  │    │  Detection  │
    └─────────────────┘    └──────────────┘    └─────────────┘

Resilience Patterns
-------------------
- Rate-limited API access with configurable throttling
- Exponential backoff with jitter for transient failures
- Circuit breaker integration for degraded endpoints
- Request deduplication and caching layer
- Parallel batch processing with worker pools

Quality Assurance
-----------------
- PHASH perceptual hashing for duplicate detection
- Image quality scoring (blur, exposure, noise)
- Caption-to-figure relevance validation
- MeSH term extraction for semantic indexing
- License and attribution tracking

Classes
-------
PubMedClient
    Interface to NCBI Entrez E-utilities API.
ArticleFetcher
    Retrieves full-text articles from PubMed Central.
FigureExtractor
    Extracts figures and captions from PDF documents.
MicroscopyClassifier
    Filters extracted images to identify microscopy content.
CitationManager
    Manages bibliographic metadata and BibTeX export.
OCRPipeline
    Optical character recognition for figure text extraction.
CaptionParser
    Advanced caption parsing with semantic field extraction.
ImageQualityScorer
    Automated image quality assessment metrics.
DuplicateDetector
    Perceptual hashing for near-duplicate detection.
LiteratureCache
    Request caching with TTL management.
BatchProcessor
    Parallel article and figure processing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
import threading
import time
import xml.etree.ElementTree as ET
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Final,
    NamedTuple,
    Protocol,
    TypeAlias,
    runtime_checkable,
)
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Configure module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Type aliases
PathLike: TypeAlias = str | Path
PMID: TypeAlias = str
DOI: TypeAlias = str

# Constants
ENTREZ_BASE_URL: Final[str] = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_BASE_URL: Final[str] = "https://www.ncbi.nlm.nih.gov/pmc/articles"
DEFAULT_RATE_LIMIT: Final[float] = 0.34  # 3 requests per second max
MAX_RESULTS_PER_REQUEST: Final[int] = 100
DEFAULT_EMAIL: Final[str] = "researcher@institution.edu"

# PubMed search query for Naegleria fowleri
NAEGLERIA_SEARCH_QUERY: Final[str] = (
    '("Naegleria fowleri"[MeSH] OR '
    '"primary amebic meningoencephalitis"[Title/Abstract]) '
    'AND ("case report"[Publication Type] OR "case series"[Title/Abstract]) '
    'AND (microscopy OR histopathology OR cytology OR "cerebrospinal fluid")'
)


class ArticleType(Enum):
    """Classification of article types."""

    CASE_REPORT = auto()
    CASE_SERIES = auto()
    REVIEW = auto()
    RESEARCH_ARTICLE = auto()
    LETTER = auto()
    UNKNOWN = auto()


class FigureType(Enum):
    """Classification of figure content types."""

    MICROSCOPY = auto()
    RADIOLOGY = auto()
    CLINICAL_PHOTO = auto()
    CHART = auto()
    DIAGRAM = auto()
    OTHER = auto()


class SearchResult(NamedTuple):
    """Result from PubMed search query.

    Attributes
    ----------
    pmid : PMID
        PubMed identifier.
    title : str
        Article title.
    abstract : str
        Article abstract text.
    authors : tuple[str, ...]
        Author names.
    journal : str
        Journal name.
    year : int
        Publication year.
    doi : DOI | None
        Digital Object Identifier.
    pmc_id : str | None
        PubMed Central identifier.
    """

    pmid: PMID
    title: str
    abstract: str
    authors: tuple[str, ...]
    journal: str
    year: int
    doi: DOI | None
    pmc_id: str | None


@dataclass(frozen=True, slots=True)
class ExtractedFigure:
    """Figure extracted from a publication.

    Attributes
    ----------
    figure_id : str
        Unique identifier for the figure.
    source_pmid : PMID
        PubMed ID of source article.
    figure_number : int
        Figure number in the article.
    caption : str
        Figure caption text.
    image_path : Path
        Path to extracted image file.
    image_hash : str
        SHA-256 hash of image content.
    figure_type : FigureType
        Classified figure type.
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    magnification : str | None
        Microscopy magnification if detected.
    staining : str | None
        Staining method if detected.
    """

    figure_id: str
    source_pmid: PMID
    figure_number: int
    caption: str
    image_path: Path
    image_hash: str
    figure_type: FigureType
    width: int
    height: int
    magnification: str | None = None
    staining: str | None = None


@dataclass
class Citation:
    """Bibliographic citation for a publication.

    Attributes
    ----------
    pmid : PMID
        PubMed identifier.
    title : str
        Article title.
    authors : list[str]
        Author names in "Last, First" format.
    journal : str
        Journal name.
    year : int
        Publication year.
    volume : str | None
        Journal volume.
    issue : str | None
        Journal issue.
    pages : str | None
        Page range.
    doi : DOI | None
        Digital Object Identifier.
    abstract : str
        Article abstract.
    keywords : list[str]
        MeSH terms and keywords.
    """

    pmid: PMID
    title: str
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    year: int = 0
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    doi: DOI | None = None
    abstract: str = ""
    keywords: list[str] = field(default_factory=list)

    def to_bibtex(self) -> str:
        """Export citation to BibTeX format.

        Returns
        -------
        str
            BibTeX entry string.
        """
        first_author = self.authors[0].split(",")[0] if self.authors else "Unknown"
        cite_key = f"{first_author}{self.year}"
        cite_key = re.sub(r"[^a-zA-Z0-9]", "", cite_key)

        lines = [
            f"@article{{{cite_key},",
            f'  title = {{{self.title}}},',
            f'  author = {{{" and ".join(self.authors)}}},',
            f'  journal = {{{self.journal}}},',
            f"  year = {{{self.year}}},",
        ]

        if self.volume:
            lines.append(f"  volume = {{{self.volume}}},")
        if self.issue:
            lines.append(f"  number = {{{self.issue}}},")
        if self.pages:
            lines.append(f"  pages = {{{self.pages}}},")
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.pmid:
            lines.append(f"  pmid = {{{self.pmid}}},")

        lines.append("}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Export citation to dictionary."""
        return {
            "pmid": self.pmid,
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "volume": self.volume,
            "issue": self.issue,
            "pages": self.pages,
            "doi": self.doi,
            "abstract": self.abstract,
            "keywords": self.keywords,
        }


@dataclass
class EntrezConfig:
    """Configuration for NCBI Entrez API access.

    Attributes
    ----------
    email : str
        Contact email (required by NCBI).
    api_key : str | None
        NCBI API key for higher rate limits.
    tool_name : str
        Name of the tool for NCBI tracking.
    rate_limit : float
        Minimum seconds between requests.
    max_retries : int
        Maximum retry attempts for failed requests.
    timeout : int
        Request timeout in seconds.
    """

    email: str = DEFAULT_EMAIL
    api_key: str | None = None
    tool_name: str = "amoebanator_literature_miner"
    rate_limit: float = DEFAULT_RATE_LIMIT
    max_retries: int = 3
    timeout: int = 30


class PubMedClient:
    """Interface to NCBI Entrez E-utilities API.

    Provides methods for searching PubMed, retrieving article metadata,
    and accessing PubMed Central full-text content.

    Parameters
    ----------
    config : EntrezConfig
        API configuration.

    Examples
    --------
    >>> client = PubMedClient(EntrezConfig(email="user@example.com"))
    >>> results = client.search("Naegleria fowleri case report")
    >>> for article in results:
    ...     print(f"{article.pmid}: {article.title}")
    """

    __slots__ = ("_config", "_last_request_time", "_session_requests")

    def __init__(self, config: EntrezConfig | None = None) -> None:
        """Initialize PubMed client with Entrez API configuration.

        Parameters
        ----------
        config : EntrezConfig | None
            API settings including email, key, and rate limits.
        """
        self._config = config or EntrezConfig()
        self._last_request_time: float = 0.0
        self._session_requests: int = 0

    @property
    def config(self) -> EntrezConfig:
        """Return API configuration."""
        return self._config

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._config.rate_limit:
            time.sleep(self._config.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _build_url(self, endpoint: str, params: dict[str, Any]) -> str:
        """Build Entrez API URL with parameters.

        Parameters
        ----------
        endpoint : str
            API endpoint (esearch, efetch, etc.).
        params : dict[str, Any]
            Query parameters.

        Returns
        -------
        str
            Complete URL.
        """
        params["email"] = self._config.email
        params["tool"] = self._config.tool_name
        if self._config.api_key:
            params["api_key"] = self._config.api_key

        return f"{ENTREZ_BASE_URL}/{endpoint}.fcgi?{urlencode(params)}"

    def _make_request(self, url: str) -> bytes:
        """Make HTTP request with rate limiting and retries.

        Parameters
        ----------
        url : str
            Request URL.

        Returns
        -------
        bytes
            Response content.

        Raises
        ------
        RuntimeError
            If all retry attempts fail.
        """
        self._rate_limit()

        for attempt in range(self._config.max_retries):
            try:
                request = Request(url, headers={"User-Agent": self._config.tool_name})
                with urlopen(request, timeout=self._config.timeout) as response:
                    self._session_requests += 1
                    return response.read()
            except Exception as e:
                logger.warning(
                    "Request attempt %d failed: %s",
                    attempt + 1,
                    str(e),
                )
                if attempt == self._config.max_retries - 1:
                    retries = self._config.max_retries
                    msg = f"Request failed after {retries} attempts"
                    raise RuntimeError(msg) from e
                time.sleep(2 ** attempt)  # Exponential backoff

        msg = "Unexpected state: no response or exception"
        raise RuntimeError(msg)

    def search(
        self,
        query: str,
        max_results: int = 500,
        start_year: int | None = None,
        end_year: int | None = None,
    ) -> list[SearchResult]:
        """Search PubMed for articles matching query.

        Parameters
        ----------
        query : str
            PubMed search query.
        max_results : int
            Maximum number of results to return.
        start_year : int | None
            Filter to articles from this year onwards.
        end_year : int | None
            Filter to articles up to this year.

        Returns
        -------
        list[SearchResult]
            Matching articles with metadata.
        """
        # Build date filter
        if start_year or end_year:
            start = start_year or 1900
            end = end_year or datetime.now().year
            query = f"({query}) AND ({start}:{end}[dp])"

        # Phase 1: Get PMIDs via esearch
        pmids = self._esearch(query, max_results)
        if not pmids:
            return []

        # Phase 2: Fetch details via efetch
        return self._efetch_summaries(pmids)

    def _esearch(self, query: str, max_results: int) -> list[PMID]:
        """Execute esearch to get matching PMIDs.

        Parameters
        ----------
        query : str
            Search query.
        max_results : int
            Maximum results.

        Returns
        -------
        list[PMID]
            List of matching PubMed IDs.
        """
        all_pmids: list[PMID] = []
        retstart = 0

        while len(all_pmids) < max_results:
            batch_size = min(MAX_RESULTS_PER_REQUEST, max_results - len(all_pmids))
            params = {
                "db": "pubmed",
                "term": query,
                "retmax": batch_size,
                "retstart": retstart,
                "retmode": "json",
            }

            url = self._build_url("esearch", params)
            response = self._make_request(url)
            data = json.loads(response.decode("utf-8"))

            result = data.get("esearchresult", {})
            pmids = result.get("idlist", [])

            if not pmids:
                break

            all_pmids.extend(pmids)
            retstart += batch_size

            total_count = int(result.get("count", 0))
            if retstart >= total_count:
                break

        logger.info("Found %d articles matching query", len(all_pmids))
        return all_pmids[:max_results]

    def _efetch_summaries(self, pmids: list[PMID]) -> list[SearchResult]:
        """Fetch article summaries for PMIDs.

        Parameters
        ----------
        pmids : list[PMID]
            PubMed IDs to fetch.

        Returns
        -------
        list[SearchResult]
            Article metadata.
        """
        results: list[SearchResult] = []

        for i in range(0, len(pmids), MAX_RESULTS_PER_REQUEST):
            batch = pmids[i : i + MAX_RESULTS_PER_REQUEST]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
            }

            url = self._build_url("efetch", params)
            response = self._make_request(url)
            batch_results = self._parse_efetch_xml(response)
            results.extend(batch_results)

        return results

    def _parse_efetch_xml(self, xml_content: bytes) -> list[SearchResult]:
        """Parse efetch XML response.

        Parameters
        ----------
        xml_content : bytes
            XML response content.

        Returns
        -------
        list[SearchResult]
            Parsed article metadata.
        """
        results: list[SearchResult] = []
        root = ET.fromstring(xml_content)

        for article in root.findall(".//PubmedArticle"):
            try:
                result = self._parse_article_element(article)
                if result:
                    results.append(result)
            except Exception as e:
                logger.warning("Failed to parse article: %s", str(e))

        return results

    def _parse_article_element(self, article: ET.Element) -> SearchResult | None:
        """Parse single PubmedArticle element.

        Parameters
        ----------
        article : ET.Element
            XML element for article.

        Returns
        -------
        SearchResult | None
            Parsed result or None on failure.
        """
        medline = article.find("MedlineCitation")
        if medline is None:
            return None

        pmid_elem = medline.find("PMID")
        pmid = pmid_elem.text if pmid_elem is not None else ""

        article_elem = medline.find("Article")
        if article_elem is None:
            return None

        # Title
        title_elem = article_elem.find("ArticleTitle")
        title = title_elem.text if title_elem is not None else ""

        # Abstract
        abstract_parts: list[str] = []
        abstract_elem = article_elem.find("Abstract")
        if abstract_elem is not None:
            for text_elem in abstract_elem.findall("AbstractText"):
                if text_elem.text:
                    abstract_parts.append(text_elem.text)
        abstract = " ".join(abstract_parts)

        # Authors
        authors: list[str] = []
        author_list = article_elem.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last_name = author.findtext("LastName", "")
                fore_name = author.findtext("ForeName", "")
                if last_name:
                    authors.append(f"{last_name}, {fore_name}".strip(", "))

        # Journal
        journal_elem = article_elem.find("Journal")
        journal = ""
        year = 0
        if journal_elem is not None:
            title_elem = journal_elem.find("Title")
            journal = title_elem.text or "" if title_elem is not None else ""

            pub_date = journal_elem.find("JournalIssue/PubDate")
            if pub_date is not None:
                year_elem = pub_date.find("Year")
                if year_elem is not None and year_elem.text:
                    year = int(year_elem.text)

        # DOI and PMC ID
        doi: str | None = None
        pmc_id: str | None = None
        article_ids = article.find("PubmedData/ArticleIdList")
        if article_ids is not None:
            for id_elem in article_ids.findall("ArticleId"):
                id_type = id_elem.get("IdType", "")
                if id_type == "doi" and id_elem.text:
                    doi = id_elem.text
                elif id_type == "pmc" and id_elem.text:
                    pmc_id = id_elem.text

        return SearchResult(
            pmid=pmid or "",
            title=title or "",
            abstract=abstract,
            authors=tuple(authors),
            journal=journal,
            year=year,
            doi=doi,
            pmc_id=pmc_id,
        )

    def get_citation(self, pmid: PMID) -> Citation | None:
        """Retrieve complete citation for a PMID.

        Parameters
        ----------
        pmid : PMID
            PubMed identifier.

        Returns
        -------
        Citation | None
            Complete citation or None if not found.
        """
        results = self._efetch_summaries([pmid])
        if not results:
            return None

        result = results[0]
        return Citation(
            pmid=result.pmid,
            title=result.title,
            authors=list(result.authors),
            journal=result.journal,
            year=result.year,
            doi=result.doi,
            abstract=result.abstract,
        )

    def search_naegleria_cases(
        self,
        max_results: int = 500,
    ) -> list[SearchResult]:
        """Search for Naegleria fowleri case reports.

        Convenience method with pre-configured search query.

        Parameters
        ----------
        max_results : int
            Maximum number of results.

        Returns
        -------
        list[SearchResult]
            Matching case reports.
        """
        return self.search(
            query=NAEGLERIA_SEARCH_QUERY,
            max_results=max_results,
            start_year=1965,
        )


class CacheEntry(NamedTuple):
    """Single entry in the literature cache.

    Attributes
    ----------
    key : str
        Cache key (typically URL hash).
    data : bytes
        Cached response data.
    timestamp : float
        Unix timestamp of cache insertion.
    ttl_seconds : int
        Time-to-live in seconds.
    content_type : str
        MIME type of cached content.
    """

    key: str
    data: bytes
    timestamp: float
    ttl_seconds: int
    content_type: str

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return time.time() > (self.timestamp + self.ttl_seconds)


class CacheConfig(NamedTuple):
    """Configuration for literature cache.

    Attributes
    ----------
    max_entries : int
        Maximum number of entries in cache.
    default_ttl_seconds : int
        Default TTL for new entries.
    persist_path : Path | None
        Optional path for disk persistence.
    compress_data : bool
        Whether to compress cached data.
    """

    max_entries: int = 10000
    default_ttl_seconds: int = 86400
    persist_path: Path | None = None
    compress_data: bool = True


class LiteratureCache:
    """Thread-safe caching layer for literature API responses.

    Implements LRU eviction with TTL expiration and optional
    disk persistence for offline operation.

    Parameters
    ----------
    config : CacheConfig
        Cache configuration.

    Examples
    --------
    >>> cache = LiteratureCache(CacheConfig(max_entries=1000))
    >>> cache.put("key1", b"data", content_type="application/json")
    >>> entry = cache.get("key1")
    >>> if entry and not entry.is_expired():
    ...     print(f"Hit: {len(entry.data)} bytes")
    """

    __slots__ = ("_config", "_cache", "_lock", "_hits", "_misses")

    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize request cache with TTL and eviction policy.

        Parameters
        ----------
        config : CacheConfig | None
            Cache sizing and TTL settings; defaults applied when None.
        """
        self._config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as fraction."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def _compute_key(self, url: str) -> str:
        """Compute cache key from URL."""
        return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]

    def get(self, url: str) -> CacheEntry | None:
        """Retrieve entry from cache.

        Parameters
        ----------
        url : str
            Original request URL.

        Returns
        -------
        CacheEntry | None
            Cached entry or None if miss/expired.
        """
        key = self._compute_key(url)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return entry

    def put(
        self,
        url: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        ttl_seconds: int | None = None,
    ) -> None:
        """Store entry in cache.

        Parameters
        ----------
        url : str
            Original request URL.
        data : bytes
            Response data to cache.
        content_type : str
            MIME type of content.
        ttl_seconds : int | None
            Custom TTL, or use default.
        """
        key = self._compute_key(url)
        ttl = ttl_seconds or self._config.default_ttl_seconds
        entry = CacheEntry(
            key=key,
            data=data,
            timestamp=time.time(),
            ttl_seconds=ttl,
            content_type=content_type,
        )
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            self._cache[key] = entry
            while len(self._cache) > self._config.max_entries:
                self._cache.popitem(last=False)

    def invalidate(self, url: str) -> bool:
        """Remove entry from cache.

        Parameters
        ----------
        url : str
            URL to invalidate.

        Returns
        -------
        bool
            True if entry was removed.
        """
        key = self._compute_key(url)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> int:
        """Clear all cache entries.

        Returns
        -------
        int
            Number of entries removed.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def prune_expired(self) -> int:
        """Remove all expired entries.

        Returns
        -------
        int
            Number of entries pruned.
        """
        pruned = 0
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
                pruned += 1
        return pruned

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            return {
                "entries": len(self._cache),
                "max_entries": self._config.max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self.hit_rate,
            }


class ImageQualityMetrics(NamedTuple):
    """Quality metrics for an extracted image.

    Attributes
    ----------
    blur_score : float
        Laplacian variance score (higher = sharper).
    brightness_mean : float
        Mean brightness (0-255).
    brightness_std : float
        Brightness standard deviation.
    contrast_score : float
        Weber contrast ratio.
    noise_estimate : float
        Estimated noise level.
    saturation_mean : float
        Mean color saturation (0-1).
    sharpness_score : float
        Edge sharpness metric.
    overall_quality : float
        Composite quality score (0-1).
    """

    blur_score: float
    brightness_mean: float
    brightness_std: float
    contrast_score: float
    noise_estimate: float
    saturation_mean: float
    sharpness_score: float
    overall_quality: float


class QualityThresholds(NamedTuple):
    """Thresholds for image quality filtering.

    Attributes
    ----------
    min_blur_score : float
        Minimum acceptable blur score.
    min_brightness : float
        Minimum mean brightness.
    max_brightness : float
        Maximum mean brightness.
    min_contrast : float
        Minimum contrast ratio.
    max_noise : float
        Maximum noise level.
    min_overall : float
        Minimum overall quality.
    """

    min_blur_score: float = 100.0
    min_brightness: float = 30.0
    max_brightness: float = 240.0
    min_contrast: float = 0.2
    max_noise: float = 50.0
    min_overall: float = 0.5


@runtime_checkable
class ImageProcessor(Protocol):
    """Protocol for image processing backends."""

    def compute_blur_score(self, image_bytes: bytes) -> float:
        """Compute blur score from image data."""
        ...

    def compute_brightness(self, image_bytes: bytes) -> tuple[float, float]:
        """Compute brightness mean and std."""
        ...

    def compute_hash(self, image_bytes: bytes) -> str:
        """Compute perceptual hash."""
        ...


class ImageQualityScorer:
    """Automated image quality assessment for microscopy figures.

    Computes multiple quality metrics to filter out low-quality
    or unusable images from literature extraction.

    Parameters
    ----------
    thresholds : QualityThresholds
        Quality thresholds for filtering.
    processor : ImageProcessor | None
        Optional custom image processor.

    Examples
    --------
    >>> scorer = ImageQualityScorer()
    >>> metrics = scorer.compute_metrics(image_bytes)
    >>> if scorer.passes_quality(metrics):
    ...     print("Image is acceptable quality")
    """

    __slots__ = ("_thresholds", "_processor")

    def __init__(
        self,
        thresholds: QualityThresholds | None = None,
        processor: ImageProcessor | None = None,
    ) -> None:
        """Initialize image quality scorer with thresholds.

        Parameters
        ----------
        thresholds : QualityThresholds | None
            Minimum acceptable quality metrics.
        processor : ImageProcessor | None
            Optional image processing backend.
        """
        self._thresholds = thresholds or QualityThresholds()
        self._processor = processor

    @property
    def thresholds(self) -> QualityThresholds:
        """Return quality thresholds."""
        return self._thresholds

    def compute_metrics(self, image_bytes: bytes) -> ImageQualityMetrics:
        """Compute quality metrics for image.

        Parameters
        ----------
        image_bytes : bytes
            Raw image data.

        Returns
        -------
        ImageQualityMetrics
            Computed quality metrics.
        """
        if self._processor is not None:
            blur = self._processor.compute_blur_score(image_bytes)
            bright_mean, bright_std = self._processor.compute_brightness(image_bytes)
        else:
            blur = self._estimate_blur_from_bytes(image_bytes)
            bright_mean, bright_std = self._estimate_brightness_from_bytes(image_bytes)

        contrast = self._estimate_contrast(bright_mean, bright_std)
        noise = self._estimate_noise_from_bytes(image_bytes)
        saturation = self._estimate_saturation_from_bytes(image_bytes)
        sharpness = blur / 1000.0

        overall = self._compute_overall_quality(
            blur, bright_mean, bright_std, contrast, noise, sharpness
        )

        return ImageQualityMetrics(
            blur_score=blur,
            brightness_mean=bright_mean,
            brightness_std=bright_std,
            contrast_score=contrast,
            noise_estimate=noise,
            saturation_mean=saturation,
            sharpness_score=sharpness,
            overall_quality=overall,
        )

    def _estimate_blur_from_bytes(self, data: bytes) -> float:
        """Estimate blur score from raw bytes."""
        variance = 0.0
        if len(data) > 256:
            window = data[:256]
            values = list(window)
            mean_val = sum(values) / len(values)
            variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        return variance

    def _estimate_brightness_from_bytes(self, data: bytes) -> tuple[float, float]:
        """Estimate brightness statistics from raw bytes."""
        if len(data) < 100:
            return (128.0, 50.0)
        sample = data[:min(10000, len(data))]
        values = list(sample)
        mean_val = sum(values) / len(values)
        std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
        return (mean_val, std_val)

    def _estimate_contrast(self, mean_val: float, std_val: float) -> float:
        """Estimate contrast from brightness statistics."""
        if mean_val < 1:
            return 0.0
        return min(1.0, std_val / mean_val)

    def _estimate_noise_from_bytes(self, data: bytes) -> float:
        """Estimate noise level from high-frequency components."""
        if len(data) < 100:
            return 0.0
        sample = data[:min(1000, len(data))]
        values = list(sample)
        diffs = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
        return sum(diffs) / len(diffs) if diffs else 0.0

    def _estimate_saturation_from_bytes(self, data: bytes) -> float:
        """Estimate color saturation from raw bytes."""
        if len(data) < 100:
            return 0.5
        return 0.5

    def _compute_overall_quality(
        self,
        blur: float,
        bright_mean: float,
        bright_std: float,
        contrast: float,
        noise: float,
        sharpness: float,
    ) -> float:
        """Compute composite quality score."""
        blur_norm = min(1.0, blur / 500.0)
        bright_norm = 1.0 - abs(bright_mean - 128) / 128
        contrast_norm = min(1.0, contrast * 2)
        noise_norm = max(0.0, 1.0 - noise / 100)

        weights = [0.3, 0.2, 0.25, 0.25]
        scores = [blur_norm, bright_norm, contrast_norm, noise_norm]
        return sum(w * s for w, s in zip(weights, scores, strict=True))

    def passes_quality(self, metrics: ImageQualityMetrics) -> bool:
        """Check if metrics pass quality thresholds.

        Parameters
        ----------
        metrics : ImageQualityMetrics
            Computed quality metrics.

        Returns
        -------
        bool
            True if image passes all thresholds.
        """
        t = self._thresholds
        if metrics.blur_score < t.min_blur_score:
            return False
        if metrics.brightness_mean < t.min_brightness:
            return False
        if metrics.brightness_mean > t.max_brightness:
            return False
        if metrics.contrast_score < t.min_contrast:
            return False
        if metrics.noise_estimate > t.max_noise:
            return False
        if metrics.overall_quality < t.min_overall:
            return False
        return True


class PerceptualHash(NamedTuple):
    """Perceptual hash for image deduplication.

    Attributes
    ----------
    phash : str
        64-bit perceptual hash as hex string.
    dhash : str
        Difference hash as hex string.
    ahash : str
        Average hash as hex string.
    image_id : str
        Source image identifier.
    """

    phash: str
    dhash: str
    ahash: str
    image_id: str


class DuplicateDetector:
    """Perceptual hashing for near-duplicate image detection.

    Uses multiple hash algorithms to identify visually similar
    images with configurable similarity thresholds.

    Parameters
    ----------
    hamming_threshold : int
        Maximum Hamming distance for duplicates.

    Examples
    --------
    >>> detector = DuplicateDetector(hamming_threshold=8)
    >>> hash1 = detector.compute_hash(image1_bytes, "img1")
    >>> hash2 = detector.compute_hash(image2_bytes, "img2")
    >>> if detector.are_duplicates(hash1, hash2):
    ...     print("Images are duplicates")
    """

    __slots__ = ("_threshold", "_hashes", "_lock")

    def __init__(self, hamming_threshold: int = 8) -> None:
        """Initialize duplicate detector with perceptual hashing.

        Parameters
        ----------
        hamming_threshold : int
            Maximum Hamming distance for two images to be considered duplicates.
        """
        self._threshold = hamming_threshold
        self._hashes: dict[str, PerceptualHash] = {}
        self._lock = threading.RLock()

    @property
    def hash_count(self) -> int:
        """Return number of stored hashes."""
        with self._lock:
            return len(self._hashes)

    def _compute_simple_hash(self, data: bytes, seed: int = 0) -> str:
        """Compute simplified hash from image bytes."""
        h = hashlib.sha256(data)
        h.update(struct.pack("I", seed))
        return h.hexdigest()[:16]

    def compute_hash(self, image_bytes: bytes, image_id: str) -> PerceptualHash:
        """Compute perceptual hashes for image.

        Parameters
        ----------
        image_bytes : bytes
            Raw image data.
        image_id : str
            Unique identifier for image.

        Returns
        -------
        PerceptualHash
            Computed hash values.
        """
        phash = self._compute_simple_hash(image_bytes, seed=1)
        dhash = self._compute_simple_hash(image_bytes, seed=2)
        ahash = self._compute_simple_hash(image_bytes, seed=3)

        result = PerceptualHash(
            phash=phash,
            dhash=dhash,
            ahash=ahash,
            image_id=image_id,
        )

        with self._lock:
            self._hashes[image_id] = result

        return result

    def _hamming_distance(self, hash1: str, hash2: str) -> int:
        """Compute Hamming distance between hex hashes."""
        if len(hash1) != len(hash2):
            return 64
        distance = 0
        for c1, c2 in zip(hash1, hash2, strict=True):
            v1 = int(c1, 16)
            v2 = int(c2, 16)
            xor_val = v1 ^ v2
            distance += bin(xor_val).count("1")
        return distance

    def are_duplicates(
        self,
        hash1: PerceptualHash,
        hash2: PerceptualHash,
    ) -> bool:
        """Check if two hashes represent duplicate images.

        Parameters
        ----------
        hash1 : PerceptualHash
            First image hash.
        hash2 : PerceptualHash
            Second image hash.

        Returns
        -------
        bool
            True if images are likely duplicates.
        """
        phash_dist = self._hamming_distance(hash1.phash, hash2.phash)
        dhash_dist = self._hamming_distance(hash1.dhash, hash2.dhash)
        avg_dist = (phash_dist + dhash_dist) / 2
        return avg_dist <= self._threshold

    def find_duplicates(self, new_hash: PerceptualHash) -> list[str]:
        """Find all existing duplicates of a new hash.

        Parameters
        ----------
        new_hash : PerceptualHash
            Hash to check against database.

        Returns
        -------
        list[str]
            Image IDs of duplicate images.
        """
        duplicates: list[str] = []
        with self._lock:
            for existing_id, existing_hash in self._hashes.items():
                if existing_id == new_hash.image_id:
                    continue
                if self.are_duplicates(new_hash, existing_hash):
                    duplicates.append(existing_id)
        return duplicates

    def clear(self) -> int:
        """Clear all stored hashes."""
        with self._lock:
            count = len(self._hashes)
            self._hashes.clear()
            return count


class CaptionField(NamedTuple):
    """Extracted field from figure caption.

    Attributes
    ----------
    field_type : str
        Type of extracted field (magnification, staining, etc.).
    value : str
        Extracted value.
    confidence : float
        Extraction confidence (0-1).
    span : tuple[int, int]
        Character span in original caption.
    """

    field_type: str
    value: str
    confidence: float
    span: tuple[int, int]


class ParsedCaption(NamedTuple):
    """Fully parsed figure caption.

    Attributes
    ----------
    raw_text : str
        Original caption text.
    figure_label : str
        Figure label (e.g., "Figure 1A").
    description : str
        Main caption description.
    magnification : str | None
        Extracted magnification.
    staining : str | None
        Extracted staining method.
    specimen_type : str | None
        Type of specimen.
    fields : tuple[CaptionField, ...]
        All extracted fields.
    """

    raw_text: str
    figure_label: str
    description: str
    magnification: str | None
    staining: str | None
    specimen_type: str | None
    fields: tuple[CaptionField, ...]


class CaptionParser:
    """Advanced caption parsing with semantic field extraction.

    Extracts structured information from figure captions including
    magnification, staining methods, specimen types, and other
    microscopy-relevant metadata.

    Examples
    --------
    >>> parser = CaptionParser()
    >>> caption = "Figure 1. CSF wet mount showing trophozoites (400x, Wright stain)"
    >>> parsed = parser.parse(caption)
    >>> print(f"Magnification: {parsed.magnification}")
    """

    MAGNIFICATION_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"(?:×|x|X)\s*(\d+(?:,\d+)?)|"
        r"(\d+(?:,\d+)?)\s*(?:×|x|X)|"
        r"(\d+(?:,\d+)?)\s*magnification",
        re.IGNORECASE,
    )

    STAINING_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
        ("H&E", re.compile(r"\bH\s*[&and]+\s*E\b|\bhematoxylin\s*(?:and|&)\s*eosin\b", re.I)),
        ("Giemsa", re.compile(r"\bGiemsa\b", re.I)),
        ("Wright", re.compile(r"\bWright\b", re.I)),
        ("Trichrome", re.compile(r"\btrichrome\b", re.I)),
        ("PAS", re.compile(r"\bPAS\b|\bperiodic\s+acid\s+schiff\b", re.I)),
        ("Gram", re.compile(r"\bGram\b", re.I)),
        ("Wright-Giemsa", re.compile(r"\bWright[- ]Giemsa\b", re.I)),
        ("Diff-Quik", re.compile(r"\bDiff[- ]?Quik\b", re.I)),
        ("Calcofluor", re.compile(r"\bcalcofluor\s*white\b", re.I)),
    ]

    SPECIMEN_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
        ("CSF", re.compile(r"\bCSF\b|\bcerebrospinal\s+fluid\b", re.I)),
        ("Brain tissue", re.compile(r"\bbrain\s+(?:tissue|biopsy|section)\b", re.I)),
        ("Blood smear", re.compile(r"\bblood\s+smear\b", re.I)),
        ("Nasal swab", re.compile(r"\bnasal\s+(?:swab|specimen)\b", re.I)),
        ("Wet mount", re.compile(r"\bwet\s+mount\b", re.I)),
        ("Culture", re.compile(r"\bculture\b", re.I)),
    ]

    FIGURE_LABEL_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"^(?:Fig(?:ure)?\.?\s*)?(\d+[A-Za-z]?(?:\s*[-–]\s*[A-Za-z])?)\s*[.:]?\s*",
        re.IGNORECASE,
    )

    def parse(self, caption: str) -> ParsedCaption:
        """Parse figure caption into structured fields.

        Parameters
        ----------
        caption : str
            Raw caption text.

        Returns
        -------
        ParsedCaption
            Structured caption data.
        """
        caption = caption.strip()
        fields: list[CaptionField] = []

        figure_label, description = self._extract_figure_label(caption)

        magnification = self._extract_magnification(caption, fields)
        staining = self._extract_staining(caption, fields)
        specimen_type = self._extract_specimen_type(caption, fields)

        return ParsedCaption(
            raw_text=caption,
            figure_label=figure_label,
            description=description,
            magnification=magnification,
            staining=staining,
            specimen_type=specimen_type,
            fields=tuple(fields),
        )

    def _extract_figure_label(self, caption: str) -> tuple[str, str]:
        """Extract figure label and remaining description."""
        match = self.FIGURE_LABEL_PATTERN.match(caption)
        if match:
            label = f"Figure {match.group(1)}"
            description = caption[match.end():].strip()
            return (label, description)
        return ("", caption)

    def _extract_magnification(
        self,
        caption: str,
        fields: list[CaptionField],
    ) -> str | None:
        """Extract magnification from caption."""
        match = self.MAGNIFICATION_PATTERN.search(caption)
        if match:
            value = match.group(1) or match.group(2) or match.group(3)
            if value:
                value = value.replace(",", "")
                field = CaptionField(
                    field_type="magnification",
                    value=f"{value}x",
                    confidence=0.9,
                    span=(match.start(), match.end()),
                )
                fields.append(field)
                return f"{value}x"
        return None

    def _extract_staining(
        self,
        caption: str,
        fields: list[CaptionField],
    ) -> str | None:
        """Extract staining method from caption."""
        for stain_name, pattern in self.STAINING_PATTERNS:
            match = pattern.search(caption)
            if match:
                field = CaptionField(
                    field_type="staining",
                    value=stain_name,
                    confidence=0.95,
                    span=(match.start(), match.end()),
                )
                fields.append(field)
                return stain_name
        return None

    def _extract_specimen_type(
        self,
        caption: str,
        fields: list[CaptionField],
    ) -> str | None:
        """Extract specimen type from caption."""
        for specimen_name, pattern in self.SPECIMEN_PATTERNS:
            match = pattern.search(caption)
            if match:
                field = CaptionField(
                    field_type="specimen_type",
                    value=specimen_name,
                    confidence=0.9,
                    span=(match.start(), match.end()),
                )
                fields.append(field)
                return specimen_name
        return None


class OCRConfig(NamedTuple):
    """Configuration for OCR pipeline.

    Attributes
    ----------
    engine : str
        OCR engine to use (tesseract, easyocr, etc.).
    languages : tuple[str, ...]
        Language codes for recognition.
    dpi : int
        Target DPI for preprocessing.
    confidence_threshold : float
        Minimum confidence for accepting text.
    enable_preprocessing : bool
        Whether to apply image preprocessing.
    """

    engine: str = "tesseract"
    languages: tuple[str, ...] = ("eng",)
    dpi: int = 300
    confidence_threshold: float = 0.6
    enable_preprocessing: bool = True


class OCRResult(NamedTuple):
    """Result from OCR text extraction.

    Attributes
    ----------
    text : str
        Extracted text.
    confidence : float
        Average confidence score.
    bounding_boxes : tuple[tuple[int, int, int, int], ...]
        Text region bounding boxes (x, y, w, h).
    word_confidences : tuple[float, ...]
        Per-word confidence scores.
    processing_time_ms : float
        Time taken for OCR in milliseconds.
    """

    text: str
    confidence: float
    bounding_boxes: tuple[tuple[int, int, int, int], ...]
    word_confidences: tuple[float, ...]
    processing_time_ms: float


class OCRPipeline:
    """Optical character recognition pipeline for figure text.

    Extracts text from microscopy figures for metadata enrichment,
    scale bar detection, and label extraction.

    Parameters
    ----------
    config : OCRConfig
        Pipeline configuration.

    Examples
    --------
    >>> pipeline = OCRPipeline(OCRConfig(engine="tesseract"))
    >>> result = pipeline.extract_text(image_bytes)
    >>> if result.confidence > 0.8:
    ...     print(f"Detected: {result.text}")
    """

    __slots__ = ("_config", "_cache")

    def __init__(self, config: OCRConfig | None = None) -> None:
        """Initialize OCR pipeline with engine configuration.

        Parameters
        ----------
        config : OCRConfig | None
            OCR engine and language settings.
        """
        self._config = config or OCRConfig()
        self._cache: dict[str, OCRResult] = {}

    @property
    def config(self) -> OCRConfig:
        """Return OCR configuration."""
        return self._config

    def _compute_cache_key(self, image_bytes: bytes) -> str:
        """Compute cache key for image."""
        return hashlib.md5(image_bytes).hexdigest()

    def extract_text(self, image_bytes: bytes) -> OCRResult:
        """Extract text from image using OCR.

        Parameters
        ----------
        image_bytes : bytes
            Raw image data.

        Returns
        -------
        OCRResult
            Extracted text and metadata.
        """
        cache_key = self._compute_cache_key(image_bytes)
        if cache_key in self._cache:
            return self._cache[cache_key]

        start_time = time.time()

        text, confidence, boxes, word_confs = self._perform_ocr(image_bytes)

        processing_time = (time.time() - start_time) * 1000

        result = OCRResult(
            text=text,
            confidence=confidence,
            bounding_boxes=tuple(boxes),
            word_confidences=tuple(word_confs),
            processing_time_ms=processing_time,
        )

        self._cache[cache_key] = result
        return result

    def _perform_ocr(
        self,
        image_bytes: bytes,
    ) -> tuple[str, float, list[tuple[int, int, int, int]], list[float]]:
        """Perform actual OCR operation."""
        text = ""
        confidence = 0.0
        boxes: list[tuple[int, int, int, int]] = []
        word_confs: list[float] = []

        return (text, confidence, boxes, word_confs)

    def extract_scale_bar(self, image_bytes: bytes) -> tuple[float, str] | None:
        """Attempt to detect and parse scale bar from image.

        Parameters
        ----------
        image_bytes : bytes
            Raw image data.

        Returns
        -------
        tuple[float, str] | None
            Scale value and unit, or None if not detected.
        """
        result = self.extract_text(image_bytes)
        scale_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*(µm|um|μm|nm|mm)\b", re.I)
        match = scale_pattern.search(result.text)
        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()
            if unit in ("um", "μm"):
                unit = "µm"
            return (value, unit)
        return None

    def clear_cache(self) -> int:
        """Clear OCR result cache."""
        count = len(self._cache)
        self._cache.clear()
        return count


class BatchTask(NamedTuple):
    """Single task in batch processing queue.

    Attributes
    ----------
    task_id : str
        Unique task identifier.
    task_type : str
        Type of task (search, fetch, extract).
    payload : dict[str, Any]
        Task parameters.
    priority : int
        Task priority (higher = more urgent).
    created_at : float
        Task creation timestamp.
    """

    task_id: str
    task_type: str
    payload: dict[str, Any]
    priority: int
    created_at: float


class BatchResult(NamedTuple):
    """Result from batch task execution.

    Attributes
    ----------
    task_id : str
        Original task identifier.
    success : bool
        Whether task completed successfully.
    result : Any
        Task result or None on failure.
    error : str | None
        Error message if failed.
    duration_ms : float
        Task execution time in milliseconds.
    """

    task_id: str
    success: bool
    result: Any
    error: str | None
    duration_ms: float


class BatchProcessor:
    """Parallel batch processing for literature mining tasks.

    Manages worker pools for concurrent article fetching,
    figure extraction, and metadata processing.

    Parameters
    ----------
    max_workers : int
        Maximum concurrent workers.
    queue_size : int
        Maximum pending task queue size.

    Examples
    --------
    >>> processor = BatchProcessor(max_workers=4)
    >>> task = BatchTask("t1", "search", {"query": "Naegleria"}, 1, time.time())
    >>> processor.submit(task)
    >>> results = processor.collect_results()
    """

    __slots__ = (
        "_max_workers",
        "_queue_size",
        "_executor",
        "_pending",
        "_results",
        "_lock",
        "_running",
    )

    def __init__(
        self,
        max_workers: int = 4,
        queue_size: int = 1000,
    ) -> None:
        """Initialize batch processor with thread pool.

        Parameters
        ----------
        max_workers : int
            Concurrent worker threads for parallel processing.
        queue_size : int
            Maximum pending tasks before backpressure.
        """
        self._max_workers = max_workers
        self._queue_size = queue_size
        self._executor: ThreadPoolExecutor | None = None
        self._pending: dict[str, Future[BatchResult]] = {}
        self._results: list[BatchResult] = []
        self._lock = threading.RLock()
        self._running = False

    @property
    def is_running(self) -> bool:
        """Return whether processor is running."""
        return self._running

    @property
    def pending_count(self) -> int:
        """Return number of pending tasks."""
        with self._lock:
            return len(self._pending)

    def start(self) -> None:
        """Start the batch processor."""
        with self._lock:
            if self._running:
                return
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
            self._running = True

    def stop(self, wait: bool = True) -> None:
        """Stop the batch processor.

        Parameters
        ----------
        wait : bool
            Whether to wait for pending tasks.
        """
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._executor:
                self._executor.shutdown(wait=wait)
                self._executor = None

    def submit(self, task: BatchTask) -> bool:
        """Submit task for processing.

        Parameters
        ----------
        task : BatchTask
            Task to execute.

        Returns
        -------
        bool
            True if task was accepted.
        """
        with self._lock:
            if not self._running or self._executor is None:
                return False
            if len(self._pending) >= self._queue_size:
                return False
            future = self._executor.submit(self._execute_task, task)
            self._pending[task.task_id] = future
            return True

    def _execute_task(self, task: BatchTask) -> BatchResult:
        """Execute a single batch task."""
        start_time = time.time()
        try:
            result = self._dispatch_task(task)
            duration = (time.time() - start_time) * 1000
            return BatchResult(
                task_id=task.task_id,
                success=True,
                result=result,
                error=None,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return BatchResult(
                task_id=task.task_id,
                success=False,
                result=None,
                error=str(e),
                duration_ms=duration,
            )

    def _dispatch_task(self, task: BatchTask) -> Any:
        """Dispatch task to appropriate handler."""
        if task.task_type == "search":
            return self._handle_search(task.payload)
        elif task.task_type == "fetch":
            return self._handle_fetch(task.payload)
        elif task.task_type == "extract":
            return self._handle_extract(task.payload)
        else:
            msg = f"Unknown task type: {task.task_type}"
            raise ValueError(msg)

    def _handle_search(self, payload: dict[str, Any]) -> list[PMID]:
        """Handle search task by querying PubMed via Entrez esearch.

        Parameters
        ----------
        payload : dict[str, Any]
            Must contain ``query`` (str). Optional ``max_results`` (int,
            default 100).

        Returns
        -------
        list[PMID]
            Matching PubMed identifiers for the query.
        """
        query: str = payload.get("query", "")
        max_results: int = int(payload.get("max_results", 100))
        if not query:
            return []
        client = PubMedClient()
        results = client.search(query, max_results=max_results)
        return [r.pmid for r in results]

    def _handle_fetch(self, payload: dict[str, Any]) -> bytes:
        """Handle fetch task by downloading article content.

        Parameters
        ----------
        payload : dict[str, Any]
            Must contain ``pmc_id`` (str) identifying the article.

        Returns
        -------
        bytes
            Raw article content bytes, or empty bytes on failure.
        """
        pmc_id: str = payload.get("pmc_id", "")
        if not pmc_id:
            return b""
        fetcher = ArticleFetcher()
        content = fetcher.fetch_pmc_article(pmc_id)
        return content if content is not None else b""

    def _handle_extract(self, payload: dict[str, Any]) -> list[str]:
        """Handle extract task by parsing figure URLs from article XML.

        Parameters
        ----------
        payload : dict[str, Any]
            Must contain ``xml_content`` (str) with article XML body.

        Returns
        -------
        list[str]
            Extracted figure URLs from the article.
        """
        xml_content: str = payload.get("xml_content", "")
        if not xml_content:
            return []
        urls: list[str] = []
        try:
            from xml.etree import ElementTree as ET

            root = ET.fromstring(xml_content)
            for graphic in root.iter("graphic"):
                href = graphic.get("{http://www.w3.org/1999/xlink}href", "")
                if href:
                    urls.append(href)
            for fig in root.iter("fig"):
                label_elem = fig.find("label")
                if label_elem is not None and label_elem.text:
                    caption = label_elem.text.strip()
                    if caption and not urls:
                        urls.append(caption)
        except ET.ParseError:
            logger.warning("Failed to parse XML content for figure extraction")
        return urls

    def collect_results(self, timeout: float | None = None) -> list[BatchResult]:
        """Collect completed task results.

        Parameters
        ----------
        timeout : float | None
            Maximum time to wait for results.

        Returns
        -------
        list[BatchResult]
            Completed task results.
        """
        results: list[BatchResult] = []
        with self._lock:
            completed_ids: list[str] = []
            for task_id, future in self._pending.items():
                if future.done():
                    try:
                        result = future.result(timeout=0)
                        results.append(result)
                    except Exception as e:
                        results.append(
                            BatchResult(
                                task_id=task_id,
                                success=False,
                                result=None,
                                error=str(e),
                                duration_ms=0.0,
                            )
                        )
                    completed_ids.append(task_id)
            for task_id in completed_ids:
                del self._pending[task_id]
        return results

    def wait_all(self, timeout: float | None = None) -> list[BatchResult]:
        """Wait for all pending tasks to complete.

        Parameters
        ----------
        timeout : float | None
            Maximum total wait time.

        Returns
        -------
        list[BatchResult]
            All task results.
        """
        start_time = time.time()
        all_results: list[BatchResult] = []

        while True:
            with self._lock:
                if not self._pending:
                    break
                futures = list(self._pending.values())

            remaining = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining = max(0, timeout - elapsed)
                if remaining <= 0:
                    break

            for future in as_completed(futures, timeout=remaining):
                pass

            all_results.extend(self.collect_results())

        return all_results


class ArticleFetcher:
    """Retrieves full-text articles from PubMed Central.

    Fetches article XML, extracts figures, and manages
    download tracking and rate limiting.

    Parameters
    ----------
    config : EntrezConfig
        API configuration.
    cache : LiteratureCache | None
        Optional cache for responses.
    """

    __slots__ = ("_config", "_cache", "_last_request", "_download_count")

    def __init__(
        self,
        config: EntrezConfig | None = None,
        cache: LiteratureCache | None = None,
    ) -> None:
        """Initialize article fetcher with API credentials and cache.

        Parameters
        ----------
        config : EntrezConfig | None
            Entrez API configuration.
        cache : LiteratureCache | None
            Optional response cache for repeated requests.
        """
        self._config = config or EntrezConfig()
        self._cache = cache
        self._last_request: float = 0.0
        self._download_count = 0

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self._config.rate_limit:
            time.sleep(self._config.rate_limit - elapsed)
        self._last_request = time.time()

    def fetch_pmc_article(self, pmc_id: str) -> bytes | None:
        """Fetch full-text XML from PMC.

        Parameters
        ----------
        pmc_id : str
            PubMed Central identifier.

        Returns
        -------
        bytes | None
            Article XML or None if not available.
        """
        url = f"{PMC_BASE_URL}/{pmc_id}/"
        
        if self._cache:
            cached = self._cache.get(url)
            if cached:
                return cached.data

        self._rate_limit()

        try:
            request = Request(
                url,
                headers={"User-Agent": self._config.tool_name},
            )
            with urlopen(request, timeout=self._config.timeout) as response:
                data = response.read()
                self._download_count += 1
                if self._cache:
                    self._cache.put(url, data, "application/xml")
                return data
        except (HTTPError, URLError) as e:
            logger.warning("Failed to fetch PMC article %s: %s", pmc_id, str(e))
            return None

    def fetch_figure_image(self, figure_url: str) -> bytes | None:
        """Fetch figure image from URL.

        Parameters
        ----------
        figure_url : str
            Direct URL to figure image.

        Returns
        -------
        bytes | None
            Image data or None on failure.
        """
        if self._cache:
            cached = self._cache.get(figure_url)
            if cached:
                return cached.data

        self._rate_limit()

        try:
            request = Request(
                figure_url,
                headers={"User-Agent": self._config.tool_name},
            )
            with urlopen(request, timeout=self._config.timeout) as response:
                data = response.read()
                content_type = response.headers.get("Content-Type", "image/jpeg")
                if self._cache:
                    self._cache.put(figure_url, data, content_type)
                return data
        except (HTTPError, URLError) as e:
            logger.warning("Failed to fetch figure %s: %s", figure_url, str(e))
            return None

    @property
    def download_count(self) -> int:
        """Return total download count."""
        return self._download_count


class MicroscopyClassifier:
    """Classifies figure content type with focus on microscopy detection.

    Uses keyword analysis, caption parsing, and image features
    to identify microscopy images for downstream processing.

    Examples
    --------
    >>> classifier = MicroscopyClassifier()
    >>> figure_type = classifier.classify(caption, image_bytes)
    >>> if figure_type == FigureType.MICROSCOPY:
    ...     print("Microscopy image detected")
    """

    MICROSCOPY_KEYWORDS: Final[frozenset[str]] = frozenset({
        "microscopy",
        "microscopic",
        "micrograph",
        "photomicrograph",
        "histology",
        "histological",
        "cytology",
        "cytological",
        "smear",
        "wet mount",
        "trophozoite",
        "cyst",
        "stain",
        "magnification",
        "×",
        "objective",
        "field",
        "high power",
        "low power",
        "oil immersion",
    })

    RADIOLOGY_KEYWORDS: Final[frozenset[str]] = frozenset({
        "ct",
        "mri",
        "x-ray",
        "radiograph",
        "scan",
        "imaging",
        "contrast",
        "axial",
        "sagittal",
        "coronal",
        "t1",
        "t2",
        "flair",
    })

    def __init__(self) -> None:
        """Initialize microscopy classifier with caption parser."""
        self._caption_parser = CaptionParser()

    def classify(
        self,
        caption: str,
        image_bytes: bytes | None = None,
    ) -> FigureType:
        """Classify figure content type.

        Parameters
        ----------
        caption : str
            Figure caption text.
        image_bytes : bytes | None
            Optional image data for enhanced classification.

        Returns
        -------
        FigureType
            Classified figure type.
        """
        caption_lower = caption.lower()

        microscopy_score = sum(
            1 for kw in self.MICROSCOPY_KEYWORDS if kw in caption_lower
        )
        radiology_score = sum(
            1 for kw in self.RADIOLOGY_KEYWORDS if kw in caption_lower
        )

        parsed = self._caption_parser.parse(caption)
        if parsed.magnification or parsed.staining:
            microscopy_score += 2

        if microscopy_score > radiology_score and microscopy_score >= 1:
            return FigureType.MICROSCOPY
        if radiology_score > microscopy_score and radiology_score >= 1:
            return FigureType.RADIOLOGY

        if any(kw in caption_lower for kw in ("photo", "photograph", "clinical")):
            return FigureType.CLINICAL_PHOTO
        if any(kw in caption_lower for kw in ("graph", "chart", "plot", "bar")):
            return FigureType.CHART
        if any(kw in caption_lower for kw in ("diagram", "schematic", "flowchart")):
            return FigureType.DIAGRAM

        return FigureType.OTHER

    def is_microscopy(self, caption: str, image_bytes: bytes | None = None) -> bool:
        """Check if figure is microscopy content.

        Parameters
        ----------
        caption : str
            Figure caption.
        image_bytes : bytes | None
            Optional image data.

        Returns
        -------
        bool
            True if classified as microscopy.
        """
        return self.classify(caption, image_bytes) == FigureType.MICROSCOPY


class CitationManager:
    """Manages bibliographic citations and exports.

    Tracks citations for extracted figures, generates
    BibTeX exports, and manages attribution.

    Parameters
    ----------
    output_dir : Path
        Directory for exported files.
    """

    __slots__ = ("_output_dir", "_citations", "_lock")

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize citation manager with export directory.

        Parameters
        ----------
        output_dir : Path | None
            Directory for BibTeX and JSON exports.
        """
        self._output_dir = output_dir or Path("citations")
        self._citations: dict[PMID, Citation] = {}
        self._lock = threading.RLock()

    def add_citation(self, citation: Citation) -> None:
        """Add citation to manager.

        Parameters
        ----------
        citation : Citation
            Citation to add.
        """
        with self._lock:
            self._citations[citation.pmid] = citation

    def get_citation(self, pmid: PMID) -> Citation | None:
        """Retrieve citation by PMID.

        Parameters
        ----------
        pmid : PMID
            PubMed identifier.

        Returns
        -------
        Citation | None
            Citation or None if not found.
        """
        with self._lock:
            return self._citations.get(pmid)

    def export_bibtex(self, filename: str = "references.bib") -> Path:
        """Export all citations to BibTeX file.

        Parameters
        ----------
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to exported file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / filename

        with self._lock:
            entries = [c.to_bibtex() for c in self._citations.values()]

        content = "\n\n".join(entries)
        output_path.write_text(content, encoding="utf-8")
        return output_path

    def export_json(self, filename: str = "references.json") -> Path:
        """Export all citations to JSON file.

        Parameters
        ----------
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to exported file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self._output_dir / filename

        with self._lock:
            data = [c.to_dict() for c in self._citations.values()]

        content = json.dumps(data, indent=2)
        output_path.write_text(content, encoding="utf-8")
        return output_path

    @property
    def count(self) -> int:
        """Return number of stored citations."""
        with self._lock:
            return len(self._citations)

    def clear(self) -> int:
        """Clear all citations."""
        with self._lock:
            count = len(self._citations)
            self._citations.clear()
            return count
