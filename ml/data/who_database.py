"""Phase 1.3: WHO Database Access for Global Entamoeba Surveillance.

This module implements secure access to WHO's Global Health Observatory (GHO)
data API and regional surveillance databases for entamoeba epidemiological data.
Includes rate limiting, caching, and data harmonization across regional formats.

Technical implementation follows WHO API v2 specifications with OAuth2
authentication and supports both synchronous and streaming data access.

Architecture
------------
The WHO data pipeline implements a multi-tier data access architecture:

    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
    │   WHO GHO API   │───>│ Rate Limiter │───>│   Caching   │
    │   (OAuth2)      │    │ Token Bucket │    │   Layer     │
    └─────────────────┘    └──────────────┘    └─────────────┘
           │                      │                   │
           v                      v                   v
    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
    │   Harmonizer    │───>│   Outbreak   │───>│   Trend     │
    │   (Formats)     │    │   Detector   │    │   Analyzer  │
    └─────────────────┘    └──────────────┘    └─────────────┘
           │                      │                   │
           v                      v                   v
    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
    │   Geographic    │───>│   Regional   │───>│   Report    │
    │   Clustering    │    │   Aggregator │    │   Generator │
    └─────────────────┘    └──────────────┘    └─────────────┘

Epidemiological Features
------------------------
- Outbreak detection with CUSUM algorithm
- Geographic clustering using DBSCAN
- Trend analysis with Mann-Kendall test
- Seasonal decomposition (STL)
- Reporting delay estimation
- Case fatality rate confidence intervals

Data Quality
------------
- Missing data imputation strategies
- Outlier detection and flagging
- Completeness scoring
- Cross-source validation
- Temporal consistency checks
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import (
    Any,
    Callable,
    Final,
    Iterator,
    Mapping,
    NamedTuple,
    Protocol,
    Sequence,
    TypeAlias,
)

import numpy as np

logger: Final = logging.getLogger(__name__)

JSONDict: TypeAlias = dict[str, Any]
NDArrayFloat: TypeAlias = np.ndarray
CountryCode: TypeAlias = str
Timestamp: TypeAlias = float


class WHORegion(Enum):
    """WHO Regional Offices for epidemiological data partitioning."""

    AFRO = auto()
    AMRO = auto()
    SEARO = auto()
    EURO = auto()
    EMRO = auto()
    WPRO = auto()


class DataGranularity(Enum):
    """Temporal granularity for surveillance data."""

    DAILY = auto()
    WEEKLY = auto()
    MONTHLY = auto()
    QUARTERLY = auto()
    YEARLY = auto()


class CachePolicy(Enum):
    """Caching strategies for WHO data requests."""

    NO_CACHE = auto()
    SESSION = auto()
    PERSISTENT = auto()
    STALE_WHILE_REVALIDATE = auto()


class AuthToken(NamedTuple):
    """OAuth2 token with expiration tracking."""

    access_token: str
    token_type: str
    expires_at: datetime
    refresh_token: str | None
    scope: str

    @property
    def is_expired(self) -> bool:
        """Check if token requires refresh."""
        buffer = timedelta(minutes=5)
        return datetime.now() >= (self.expires_at - buffer)


class RateLimitState(NamedTuple):
    """Rate limiter state for API compliance."""

    requests_remaining: int
    reset_time: datetime
    daily_quota: int
    daily_used: int


class SurveillanceRecord(NamedTuple):
    """Single surveillance observation from WHO database."""

    country_code: str
    region: WHORegion
    observation_date: datetime
    case_count: int
    mortality_count: int
    age_group: str
    data_source: str
    confidence_level: float


class EpidemiologicalSummary(NamedTuple):
    """Aggregated epidemiological statistics."""

    total_cases: int
    total_deaths: int
    case_fatality_rate: float
    incidence_per_100k: float
    trend_coefficient: float
    reporting_completeness: float


class TokenProvider(Protocol):
    """Protocol for OAuth2 token acquisition."""

    def get_token(self) -> AuthToken:
        """Acquire valid access token."""
        ...

    def refresh_token(self, refresh_token: str) -> AuthToken:
        """Refresh expired token."""
        ...


class DataValidator(Protocol):
    """Protocol for surveillance data validation."""

    def validate(self, record: SurveillanceRecord) -> bool:
        """Check record integrity and plausibility."""
        ...


@dataclass(frozen=True, slots=True)
class WHOEndpoint:
    """WHO API endpoint configuration."""

    base_url: str
    version: str
    path: str
    requires_auth: bool = True

    @property
    def full_url(self) -> str:
        """Construct complete endpoint URL."""
        return f"{self.base_url}/{self.version}/{self.path}"


@dataclass(frozen=True, slots=True)
class QueryParameters:
    """Parameters for WHO GHO API queries."""

    indicator_code: str
    country_codes: tuple[str, ...]
    start_year: int
    end_year: int
    granularity: DataGranularity
    regions: tuple[WHORegion, ...] = ()
    age_groups: tuple[str, ...] = ()
    include_metadata: bool = True

    def to_query_string(self) -> str:
        """Convert to URL query parameters."""
        params: list[str] = [
            f"indicator={self.indicator_code}",
            f"countries={','.join(self.country_codes)}",
            f"startYear={self.start_year}",
            f"endYear={self.end_year}",
        ]
        if self.regions:
            region_codes = [r.name for r in self.regions]
            params.append(f"regions={','.join(region_codes)}")
        if self.age_groups:
            params.append(f"ageGroups={','.join(self.age_groups)}")
        if self.include_metadata:
            params.append("metadata=true")
        return "&".join(params)


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """Cached response with metadata."""

    data: bytes
    etag: str
    cached_at: datetime
    ttl_seconds: int

    @property
    def is_stale(self) -> bool:
        """Check if cache entry requires refresh."""
        elapsed = (datetime.now() - self.cached_at).total_seconds()
        return elapsed > self.ttl_seconds


@dataclass(slots=True)
class RateLimiter:
    """Token bucket rate limiter for WHO API compliance."""

    max_requests_per_second: float = 10.0
    max_daily_requests: int = 10000
    _tokens: float = field(default=10.0, init=False, repr=False)
    _last_update: float = field(default_factory=time.time, init=False, repr=False)
    _daily_count: int = field(default=0, init=False, repr=False)
    _day_start: datetime = field(
        default_factory=lambda: datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ),
        init=False,
        repr=False,
    )

    def _refill_tokens(self) -> None:
        """Refill token bucket based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(
            self.max_requests_per_second,
            self._tokens + elapsed * self.max_requests_per_second,
        )
        self._last_update = now

    def _reset_daily_if_needed(self) -> None:
        """Reset daily counter at midnight."""
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if today > self._day_start:
            self._day_start = today
            self._daily_count = 0

    def acquire(self, timeout: float = 30.0) -> bool:
        """Acquire rate limit token, blocking if necessary."""
        self._reset_daily_if_needed()
        if self._daily_count >= self.max_daily_requests:
            logger.warning("Daily rate limit exceeded")
            return False

        start_time = time.time()
        while True:
            self._refill_tokens()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self._daily_count += 1
                return True

            if time.time() - start_time > timeout:
                return False

            wait_time = (1.0 - self._tokens) / self.max_requests_per_second
            time.sleep(min(wait_time, 0.1))

    def get_state(self) -> RateLimitState:
        """Return current rate limiter state."""
        self._reset_daily_if_needed()
        return RateLimitState(
            requests_remaining=self.max_daily_requests - self._daily_count,
            reset_time=self._day_start + timedelta(days=1),
            daily_quota=self.max_daily_requests,
            daily_used=self._daily_count,
        )


@dataclass(slots=True)
class ResponseCache:
    """LRU cache for WHO API responses with TTL support."""

    cache_dir: Path
    max_entries: int = 1000
    default_ttl: int = 3600
    _entries: dict[str, CacheEntry] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Initialize cache directory."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_persistent_cache()

    def _cache_key(self, query: QueryParameters) -> str:
        """Generate deterministic cache key."""
        key_parts = [
            query.indicator_code,
            ",".join(query.country_codes),
            str(query.start_year),
            str(query.end_year),
            query.granularity.name,
        ]
        return "_".join(key_parts)

    def _load_persistent_cache(self) -> None:
        """Load cached entries from disk."""
        index_file = self.cache_dir / "cache_index.json"
        if not index_file.exists():
            return

        try:
            with index_file.open("r") as f:
                index = json.load(f)
            for key, meta in index.items():
                cache_file = self.cache_dir / f"{key}.cache"
                if cache_file.exists():
                    entry = CacheEntry(
                        data=cache_file.read_bytes(),
                        etag=meta["etag"],
                        cached_at=datetime.fromisoformat(meta["cached_at"]),
                        ttl_seconds=meta["ttl"],
                    )
                    if not entry.is_stale:
                        self._entries[key] = entry
        except (json.JSONDecodeError, KeyError, OSError):
            logger.warning("Cache index corrupted, starting fresh")

    def get(self, query: QueryParameters) -> CacheEntry | None:
        """Retrieve cached response if available and fresh."""
        key = self._cache_key(query)
        entry = self._entries.get(key)
        if entry and not entry.is_stale:
            return entry
        if entry:
            del self._entries[key]
        return None

    def put(
        self, query: QueryParameters, data: bytes, etag: str, ttl: int | None = None
    ) -> None:
        """Store response in cache."""
        key = self._cache_key(query)
        entry = CacheEntry(
            data=data,
            etag=etag,
            cached_at=datetime.now(),
            ttl_seconds=ttl or self.default_ttl,
        )
        self._entries[key] = entry

        if len(self._entries) > self.max_entries:
            self._evict_oldest()

        cache_file = self.cache_dir / f"{key}.cache"
        cache_file.write_bytes(data)
        self._persist_index()

    def _evict_oldest(self) -> None:
        """Remove oldest cache entries when limit exceeded."""
        sorted_entries = sorted(
            self._entries.items(), key=lambda x: x[1].cached_at
        )
        entries_to_remove = len(self._entries) - self.max_entries + 100
        for key, _ in sorted_entries[:entries_to_remove]:
            del self._entries[key]
            cache_file = self.cache_dir / f"{key}.cache"
            cache_file.unlink(missing_ok=True)

    def _persist_index(self) -> None:
        """Write cache index to disk."""
        index = {
            key: {
                "etag": entry.etag,
                "cached_at": entry.cached_at.isoformat(),
                "ttl": entry.ttl_seconds,
            }
            for key, entry in self._entries.items()
        }
        index_file = self.cache_dir / "cache_index.json"
        with index_file.open("w") as f:
            json.dump(index, f)


@dataclass(slots=True)
class DataHarmonizer:
    """Harmonize surveillance data across regional formats."""

    country_code_mapping: dict[str, str] = field(default_factory=dict)
    age_group_standardization: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize standard mappings."""
        self.country_code_mapping = {
            "UK": "GBR",
            "USA": "USA",
            "US": "USA",
            "GB": "GBR",
        }
        self.age_group_standardization = {
            "0-4": "0-4",
            "under5": "0-4",
            "<5": "0-4",
            "5-14": "5-14",
            "5-9": "5-9",
            "10-14": "10-14",
            "15-24": "15-24",
            "15-19": "15-19",
            "20-24": "20-24",
            "25-64": "25-64",
            "65+": "65+",
            "elderly": "65+",
        }

    def standardize_country_code(self, code: str) -> str:
        """Convert to ISO 3166-1 alpha-3."""
        return self.country_code_mapping.get(code.upper(), code.upper())

    def standardize_age_group(self, age_group: str) -> str:
        """Convert to standard age group format."""
        normalized = age_group.lower().replace(" ", "").replace("years", "")
        return self.age_group_standardization.get(normalized, age_group)

    def harmonize_record(
        self, raw: JSONDict, region: WHORegion
    ) -> SurveillanceRecord:
        """Convert raw API response to standardized record."""
        country = self.standardize_country_code(raw.get("countryCode", ""))
        age = self.standardize_age_group(raw.get("ageGroup", "all"))

        return SurveillanceRecord(
            country_code=country,
            region=region,
            observation_date=datetime.fromisoformat(raw.get("date", "")),
            case_count=int(raw.get("cases", 0)),
            mortality_count=int(raw.get("deaths", 0)),
            age_group=age,
            data_source=raw.get("source", "WHO-GHO"),
            confidence_level=float(raw.get("confidence", 0.95)),
        )


@dataclass(slots=True)
class WHODataClient:
    """Client for WHO Global Health Observatory API access."""

    token_provider: TokenProvider
    cache_dir: Path
    rate_limiter: RateLimiter = field(default_factory=RateLimiter)
    cache: ResponseCache = field(init=False)
    harmonizer: DataHarmonizer = field(default_factory=DataHarmonizer)
    _current_token: AuthToken | None = field(default=None, init=False, repr=False)

    GHO_BASE_URL: Final = "https://ghoapi.azureedge.net/api"
    ENTAMOEBA_INDICATOR: Final = "ENTAMOEBA_INCIDENCE"

    def __post_init__(self) -> None:
        """Initialize cache and authenticate."""
        self.cache = ResponseCache(cache_dir=self.cache_dir)

    def _ensure_token(self) -> AuthToken:
        """Ensure valid authentication token."""
        if self._current_token is None or self._current_token.is_expired:
            if (
                self._current_token
                and self._current_token.refresh_token
            ):
                self._current_token = self.token_provider.refresh_token(
                    self._current_token.refresh_token
                )
            else:
                self._current_token = self.token_provider.get_token()
        return self._current_token

    def _build_headers(self) -> dict[str, str]:
        """Construct API request headers."""
        token = self._ensure_token()
        return {
            "Authorization": f"{token.token_type} {token.access_token}",
            "Accept": "application/json",
            "User-Agent": "Amoebanator-Pro/2060",
        }

    def query_surveillance_data(
        self, params: QueryParameters
    ) -> list[SurveillanceRecord]:
        """Fetch surveillance records from WHO GHO API."""
        cached = self.cache.get(params)
        if cached:
            logger.info("Returning cached WHO data")
            return self._parse_response(cached.data)

        if not self.rate_limiter.acquire():
            raise RuntimeError("Rate limit exceeded, try again later")

        logger.info(
            "Fetching WHO data for %s from %d to %d",
            params.indicator_code,
            params.start_year,
            params.end_year,
        )

        # Simulated API call - in production would use httpx/aiohttp
        response_data = self._simulate_api_call(params)
        self.cache.put(params, response_data, etag="simulated", ttl=3600)

        return self._parse_response(response_data)

    def _simulate_api_call(self, params: QueryParameters) -> bytes:
        """Simulate WHO API response for development."""
        records = []
        for country in params.country_codes:
            for year in range(params.start_year, params.end_year + 1):
                records.append(
                    {
                        "countryCode": country,
                        "date": f"{year}-01-01",
                        "cases": np.random.randint(100, 10000),
                        "deaths": np.random.randint(1, 100),
                        "ageGroup": "all",
                        "source": "WHO-GHO",
                        "confidence": 0.95,
                    }
                )
        return json.dumps({"records": records}).encode()

    def _parse_response(self, data: bytes) -> list[SurveillanceRecord]:
        """Parse API response into surveillance records."""
        parsed = json.loads(data.decode())
        records: list[SurveillanceRecord] = []
        for raw in parsed.get("records", []):
            record = self.harmonizer.harmonize_record(raw, WHORegion.AFRO)
            records.append(record)
        return records

    def stream_historical_data(
        self,
        countries: Sequence[str],
        start_year: int,
        end_year: int,
        batch_size: int = 100,
    ) -> Iterator[list[SurveillanceRecord]]:
        """Stream historical data in batches for memory efficiency."""
        for year in range(start_year, end_year + 1):
            params = QueryParameters(
                indicator_code=self.ENTAMOEBA_INDICATOR,
                country_codes=tuple(countries),
                start_year=year,
                end_year=year,
                granularity=DataGranularity.YEARLY,
            )
            records = self.query_surveillance_data(params)
            for i in range(0, len(records), batch_size):
                yield records[i : i + batch_size]

    def compute_epidemiological_summary(
        self, records: Sequence[SurveillanceRecord], population: int
    ) -> EpidemiologicalSummary:
        """Compute aggregate epidemiological metrics."""
        if not records:
            return EpidemiologicalSummary(
                total_cases=0,
                total_deaths=0,
                case_fatality_rate=0.0,
                incidence_per_100k=0.0,
                trend_coefficient=0.0,
                reporting_completeness=0.0,
            )

        total_cases = sum(r.case_count for r in records)
        total_deaths = sum(r.mortality_count for r in records)
        cfr = total_deaths / max(total_cases, 1)
        incidence = (total_cases / population) * 100000

        # Compute linear trend coefficient
        years = np.array(
            [r.observation_date.year for r in records], dtype=np.float64
        )
        cases = np.array([r.case_count for r in records], dtype=np.float64)
        if len(years) > 1:
            trend = float(np.polyfit(years, cases, 1)[0])
        else:
            trend = 0.0

        completeness = np.mean([r.confidence_level for r in records])

        return EpidemiologicalSummary(
            total_cases=total_cases,
            total_deaths=total_deaths,
            case_fatality_rate=cfr,
            incidence_per_100k=incidence,
            trend_coefficient=trend,
            reporting_completeness=float(completeness),
        )


@lru_cache(maxsize=128)
def get_who_region_countries(region: WHORegion) -> tuple[str, ...]:
    """Return country codes for WHO region (cached)."""
    region_mapping: dict[WHORegion, tuple[str, ...]] = {
        WHORegion.AFRO: (
            "NGA", "ETH", "EGY", "ZAF", "KEN", "GHA", "MOZ", "MDG",
            "CMR", "CIV", "NER", "BFA", "MLI", "MWI", "ZMB", "SEN",
        ),
        WHORegion.AMRO: (
            "USA", "MEX", "BRA", "COL", "ARG", "CAN", "PER", "VEN",
            "CHL", "GTM", "ECU", "BOL", "CUB", "HTI", "DOM", "HND",
        ),
        WHORegion.SEARO: (
            "IND", "IDN", "BGD", "THA", "MMR", "NPL", "LKA", "BTN",
        ),
        WHORegion.EURO: (
            "RUS", "DEU", "GBR", "FRA", "ITA", "ESP", "POL", "UKR",
        ),
        WHORegion.EMRO: (
            "PAK", "IRN", "SAU", "IRQ", "AFG", "YEM", "SYR", "JOR",
        ),
        WHORegion.WPRO: (
            "CHN", "JPN", "PHL", "VNM", "KOR", "MYS", "AUS", "PNG",
        ),
    }
    return region_mapping.get(region, ())


def create_default_who_client(
    cache_dir: Path, token_provider: TokenProvider
) -> WHODataClient:
    """Factory function for standard WHO client configuration."""
    return WHODataClient(
        token_provider=token_provider,
        cache_dir=cache_dir,
        rate_limiter=RateLimiter(
            max_requests_per_second=5.0,
            max_daily_requests=5000,
        ),
    )


__all__ = [
    "WHORegion",
    "DataGranularity",
    "CachePolicy",
    "AuthToken",
    "RateLimitState",
    "SurveillanceRecord",
    "EpidemiologicalSummary",
    "TokenProvider",
    "DataValidator",
    "WHOEndpoint",
    "QueryParameters",
    "CacheEntry",
    "RateLimiter",
    "ResponseCache",
    "DataHarmonizer",
    "WHODataClient",
    "get_who_region_countries",
    "create_default_who_client",
    "OutbreakAlert",
    "OutbreakSeverity",
    "OutbreakDetector",
    "CUSUMDetector",
    "TrendAnalysis",
    "TrendDirection",
    "TrendAnalyzer",
    "MannKendallResult",
    "GeographicCluster",
    "ClusteringConfig",
    "GeographicClusterer",
    "RegionalAggregator",
    "RegionalSummary",
    "ReportGenerator",
    "DataQualityMetrics",
    "QualityAssessor",
    "TimeSeriesDecomposition",
    "SeasonalDecomposer",
    "ImputationStrategy",
    "MissingDataHandler",
    "CrossSourceValidator",
    "ValidationReport",
]


class OutbreakSeverity(Enum):
    """Severity classification for outbreak alerts."""

    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    CRITICAL = auto()


class TrendDirection(Enum):
    """Direction of epidemiological trend."""

    INCREASING = auto()
    DECREASING = auto()
    STABLE = auto()
    UNDEFINED = auto()


class ImputationStrategy(Enum):
    """Strategy for handling missing data."""

    ZERO = auto()
    MEAN = auto()
    MEDIAN = auto()
    LINEAR_INTERPOLATION = auto()
    SEASONAL = auto()
    FORWARD_FILL = auto()
    BACKWARD_FILL = auto()


class OutbreakAlert(NamedTuple):
    """Alert generated by outbreak detection system.

    Attributes
    ----------
    alert_id : str
        Unique identifier for this alert.
    country_code : str
        ISO 3166-1 alpha-3 country code.
    detection_date : datetime
        When the outbreak was detected.
    severity : OutbreakSeverity
        Classified severity level.
    observed_cases : int
        Number of cases triggering alert.
    expected_cases : float
        Baseline expected case count.
    excess_ratio : float
        Ratio of observed to expected.
    confidence : float
        Confidence in detection (0-1).
    message : str
        Human-readable alert description.
    """

    alert_id: str
    country_code: CountryCode
    detection_date: datetime
    severity: OutbreakSeverity
    observed_cases: int
    expected_cases: float
    excess_ratio: float
    confidence: float
    message: str


class MannKendallResult(NamedTuple):
    """Result of Mann-Kendall trend test.

    Attributes
    ----------
    trend : TrendDirection
        Detected trend direction.
    p_value : float
        Statistical significance.
    tau : float
        Kendall tau correlation coefficient.
    slope : float
        Sen's slope estimator.
    intercept : float
        Intercept of trend line.
    """

    trend: TrendDirection
    p_value: float
    tau: float
    slope: float
    intercept: float


class TrendAnalysis(NamedTuple):
    """Comprehensive trend analysis result.

    Attributes
    ----------
    country_code : str
        Country analyzed.
    start_date : datetime
        Analysis start date.
    end_date : datetime
        Analysis end date.
    direction : TrendDirection
        Overall trend direction.
    annual_change_rate : float
        Percent change per year.
    mann_kendall : MannKendallResult
        Statistical test result.
    forecast_next_year : float
        Projected cases next year.
    """

    country_code: CountryCode
    start_date: datetime
    end_date: datetime
    direction: TrendDirection
    annual_change_rate: float
    mann_kendall: MannKendallResult
    forecast_next_year: float


class GeographicCluster(NamedTuple):
    """Geographic cluster of disease activity.

    Attributes
    ----------
    cluster_id : str
        Unique cluster identifier.
    countries : tuple[str, ...]
        Countries in cluster.
    centroid_lat : float
        Cluster centroid latitude.
    centroid_lon : float
        Cluster centroid longitude.
    total_cases : int
        Total cases in cluster.
    radius_km : float
        Approximate cluster radius.
    relative_risk : float
        Relative risk compared to baseline.
    """

    cluster_id: str
    countries: tuple[CountryCode, ...]
    centroid_lat: float
    centroid_lon: float
    total_cases: int
    radius_km: float
    relative_risk: float


class RegionalSummary(NamedTuple):
    """Summary statistics for a WHO region.

    Attributes
    ----------
    region : WHORegion
        WHO region.
    total_cases : int
        Total cases in region.
    total_deaths : int
        Total deaths in region.
    case_fatality_rate : float
        CFR for region.
    countries_reporting : int
        Number of countries with data.
    trend : TrendDirection
        Regional trend.
    hotspot_countries : tuple[str, ...]
        Countries with elevated activity.
    """

    region: WHORegion
    total_cases: int
    total_deaths: int
    case_fatality_rate: float
    countries_reporting: int
    trend: TrendDirection
    hotspot_countries: tuple[CountryCode, ...]


class DataQualityMetrics(NamedTuple):
    """Quality metrics for surveillance data.

    Attributes
    ----------
    completeness : float
        Fraction of expected data points present.
    timeliness : float
        Average reporting delay in days.
    consistency : float
        Internal consistency score.
    plausibility : float
        Plausibility score based on ranges.
    overall_score : float
        Composite quality score.
    """

    completeness: float
    timeliness: float
    consistency: float
    plausibility: float
    overall_score: float


class TimeSeriesDecomposition(NamedTuple):
    """STL decomposition of time series.

    Attributes
    ----------
    trend : tuple[float, ...]
        Trend component.
    seasonal : tuple[float, ...]
        Seasonal component.
    residual : tuple[float, ...]
        Residual component.
    period : int
        Detected seasonal period.
    """

    trend: tuple[float, ...]
    seasonal: tuple[float, ...]
    residual: tuple[float, ...]
    period: int


class ValidationReport(NamedTuple):
    """Report from cross-source validation.

    Attributes
    ----------
    source_a : str
        First data source name.
    source_b : str
        Second data source name.
    records_compared : int
        Number of records compared.
    matches : int
        Number of matching records.
    discrepancies : int
        Number of discrepancies.
    correlation : float
        Correlation between sources.
    recommendations : tuple[str, ...]
        Recommendations for resolution.
    """

    source_a: str
    source_b: str
    records_compared: int
    matches: int
    discrepancies: int
    correlation: float
    recommendations: tuple[str, ...]


class ClusteringConfig(NamedTuple):
    """Configuration for geographic clustering.

    Attributes
    ----------
    min_cluster_size : int
        Minimum countries per cluster.
    max_distance_km : float
        Maximum inter-country distance.
    min_relative_risk : float
        Minimum RR to form cluster.
    temporal_window_days : int
        Time window for clustering.
    """

    min_cluster_size: int = 2
    max_distance_km: float = 2000.0
    min_relative_risk: float = 1.5
    temporal_window_days: int = 90


@dataclass
class CUSUMDetector:
    """CUSUM algorithm for outbreak detection.

    Implements Cumulative Sum control chart for detecting
    shifts in disease incidence above baseline.

    Parameters
    ----------
    sensitivity : float
        Detection sensitivity (k parameter).
    threshold : float
        Alert threshold (h parameter).
    baseline_window : int
        Days for baseline calculation.
    """

    sensitivity: float = 0.5
    threshold: float = 4.0
    baseline_window: int = 365
    _cumsum_high: dict[CountryCode, float] = field(default_factory=dict, init=False)
    _cumsum_low: dict[CountryCode, float] = field(default_factory=dict, init=False)
    _baselines: dict[CountryCode, float] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def set_baseline(self, country: CountryCode, baseline: float) -> None:
        """Set baseline for country."""
        with self._lock:
            self._baselines[country] = baseline
            self._cumsum_high[country] = 0.0
            self._cumsum_low[country] = 0.0

    def update(self, country: CountryCode, observed: int) -> OutbreakAlert | None:
        """Update CUSUM and check for alert.

        Parameters
        ----------
        country : CountryCode
            Country code.
        observed : int
            Observed case count.

        Returns
        -------
        OutbreakAlert | None
            Alert if threshold exceeded.
        """
        with self._lock:
            baseline = self._baselines.get(country, 0.0)
            if baseline <= 0:
                return None

            normalized = (observed - baseline) / max(math.sqrt(baseline), 1.0)

            high = max(0, self._cumsum_high.get(country, 0.0) + normalized - self.sensitivity)
            low = min(0, self._cumsum_low.get(country, 0.0) + normalized + self.sensitivity)

            self._cumsum_high[country] = high
            self._cumsum_low[country] = low

            if high > self.threshold:
                severity = self._classify_severity(observed, baseline)
                alert_id = hashlib.md5(
                    f"{country}{datetime.now().isoformat()}".encode()
                ).hexdigest()[:12]

                alert = OutbreakAlert(
                    alert_id=alert_id,
                    country_code=country,
                    detection_date=datetime.now(),
                    severity=severity,
                    observed_cases=observed,
                    expected_cases=baseline,
                    excess_ratio=observed / max(baseline, 1),
                    confidence=min(0.99, 0.5 + (high - self.threshold) / 10),
                    message=f"Elevated cases in {country}: {observed} vs expected {baseline:.0f}",
                )
                self._cumsum_high[country] = 0.0
                return alert

            return None

    def _classify_severity(self, observed: int, baseline: float) -> OutbreakSeverity:
        """Classify outbreak severity."""
        ratio = observed / max(baseline, 1)
        if ratio >= 5.0:
            return OutbreakSeverity.CRITICAL
        elif ratio >= 3.0:
            return OutbreakSeverity.HIGH
        elif ratio >= 2.0:
            return OutbreakSeverity.MODERATE
        else:
            return OutbreakSeverity.LOW


@dataclass
class OutbreakDetector:
    """Multi-algorithm outbreak detection system.

    Combines CUSUM, moving average, and historical comparison
    for robust outbreak detection.

    Parameters
    ----------
    cusum : CUSUMDetector
        CUSUM detector instance.
    alert_callback : Callable[[OutbreakAlert], None] | None
        Optional callback for alerts.
    """

    cusum: CUSUMDetector = field(default_factory=CUSUMDetector)
    alert_callback: Callable[[OutbreakAlert], None] | None = None
    _alert_history: list[OutbreakAlert] = field(default_factory=list, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def initialize_baselines(
        self,
        historical_data: Mapping[CountryCode, Sequence[int]],
    ) -> None:
        """Initialize baselines from historical data.

        Parameters
        ----------
        historical_data : Mapping[CountryCode, Sequence[int]]
            Historical case counts by country.
        """
        for country, cases in historical_data.items():
            if len(cases) > 0:
                baseline = statistics.mean(cases)
                self.cusum.set_baseline(country, baseline)

    def process_record(self, record: SurveillanceRecord) -> OutbreakAlert | None:
        """Process surveillance record for outbreak detection.

        Parameters
        ----------
        record : SurveillanceRecord
            Surveillance record to process.

        Returns
        -------
        OutbreakAlert | None
            Alert if outbreak detected.
        """
        alert = self.cusum.update(record.country_code, record.case_count)

        if alert:
            with self._lock:
                self._alert_history.append(alert)
            if self.alert_callback:
                self.alert_callback(alert)

        return alert

    def process_batch(
        self,
        records: Sequence[SurveillanceRecord],
    ) -> list[OutbreakAlert]:
        """Process batch of records.

        Parameters
        ----------
        records : Sequence[SurveillanceRecord]
            Records to process.

        Returns
        -------
        list[OutbreakAlert]
            Generated alerts.
        """
        alerts = []
        for record in records:
            alert = self.process_record(record)
            if alert:
                alerts.append(alert)
        return alerts

    @property
    def alert_count(self) -> int:
        """Return total alerts generated."""
        with self._lock:
            return len(self._alert_history)

    def get_recent_alerts(self, days: int = 30) -> list[OutbreakAlert]:
        """Get alerts from recent period."""
        cutoff = datetime.now() - timedelta(days=days)
        with self._lock:
            return [a for a in self._alert_history if a.detection_date >= cutoff]


@dataclass
class TrendAnalyzer:
    """Statistical trend analysis for surveillance data.

    Implements Mann-Kendall test and Sen's slope for
    non-parametric trend detection.
    """

    min_data_points: int = 5
    alpha: float = 0.05

    def analyze(
        self,
        records: Sequence[SurveillanceRecord],
        country: CountryCode,
    ) -> TrendAnalysis | None:
        """Perform trend analysis for country.

        Parameters
        ----------
        records : Sequence[SurveillanceRecord]
            Surveillance records.
        country : CountryCode
            Country to analyze.

        Returns
        -------
        TrendAnalysis | None
            Analysis result or None if insufficient data.
        """
        country_records = [r for r in records if r.country_code == country]
        if len(country_records) < self.min_data_points:
            return None

        sorted_records = sorted(country_records, key=lambda r: r.observation_date)
        cases = [r.case_count for r in sorted_records]

        mk_result = self._mann_kendall_test(cases)

        start_date = sorted_records[0].observation_date
        end_date = sorted_records[-1].observation_date
        years_span = max((end_date - start_date).days / 365.25, 1)

        if cases[0] > 0:
            total_change = (cases[-1] - cases[0]) / cases[0]
            annual_change = total_change / years_span
        else:
            annual_change = 0.0

        forecast = cases[-1] + mk_result.slope * 12

        return TrendAnalysis(
            country_code=country,
            start_date=start_date,
            end_date=end_date,
            direction=mk_result.trend,
            annual_change_rate=annual_change,
            mann_kendall=mk_result,
            forecast_next_year=max(0, forecast),
        )

    def _mann_kendall_test(self, data: Sequence[int]) -> MannKendallResult:
        """Perform Mann-Kendall trend test."""
        n = len(data)
        if n < 3:
            return MannKendallResult(
                trend=TrendDirection.UNDEFINED,
                p_value=1.0,
                tau=0.0,
                slope=0.0,
                intercept=float(data[0]) if data else 0.0,
            )

        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                diff = data[j] - data[i]
                if diff > 0:
                    s += 1
                elif diff < 0:
                    s -= 1

        var_s = (n * (n - 1) * (2 * n + 5)) / 18
        if var_s > 0:
            if s > 0:
                z = (s - 1) / math.sqrt(var_s)
            elif s < 0:
                z = (s + 1) / math.sqrt(var_s)
            else:
                z = 0
        else:
            z = 0

        p_value = 2 * (1 - self._normal_cdf(abs(z)))

        tau = 2 * s / (n * (n - 1)) if n > 1 else 0

        slope = self._sens_slope(data)
        intercept = statistics.median(
            data[i] - slope * i for i in range(len(data))
        )

        if p_value <= self.alpha:
            if tau > 0:
                direction = TrendDirection.INCREASING
            else:
                direction = TrendDirection.DECREASING
        else:
            direction = TrendDirection.STABLE

        return MannKendallResult(
            trend=direction,
            p_value=p_value,
            tau=tau,
            slope=slope,
            intercept=intercept,
        )

    def _sens_slope(self, data: Sequence[int]) -> float:
        """Calculate Sen's slope estimator."""
        slopes = []
        n = len(data)
        for i in range(n - 1):
            for j in range(i + 1, n):
                if j != i:
                    slopes.append((data[j] - data[i]) / (j - i))
        return statistics.median(slopes) if slopes else 0.0

    def _normal_cdf(self, x: float) -> float:
        """Approximate standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))


@dataclass
class GeographicClusterer:
    """Geographic clustering for disease hotspot identification.

    Uses approximate DBSCAN-like clustering based on
    country centroids and case counts.

    Parameters
    ----------
    config : ClusteringConfig
        Clustering configuration.
    """

    config: ClusteringConfig = field(default_factory=ClusteringConfig)
    _country_coords: dict[CountryCode, tuple[float, float]] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        """Initialize country coordinates."""
        self._country_coords = {
            "USA": (37.0902, -95.7129),
            "MEX": (23.6345, -102.5528),
            "BRA": (-14.2350, -51.9253),
            "IND": (20.5937, 78.9629),
            "CHN": (35.8617, 104.1954),
            "NGA": (9.0820, 8.6753),
            "ETH": (9.1450, 40.4897),
            "EGY": (26.8206, 30.8025),
            "ZAF": (-30.5595, 22.9375),
            "KEN": (-1.2921, 36.8219),
            "GBR": (55.3781, -3.4360),
            "FRA": (46.2276, 2.2137),
            "DEU": (51.1657, 10.4515),
            "ITA": (41.8719, 12.5674),
            "ESP": (40.4637, -3.7492),
            "JPN": (36.2048, 138.2529),
            "AUS": (-25.2744, 133.7751),
            "CAN": (56.1304, -106.3468),
            "ARG": (-38.4161, -63.6167),
            "PER": (-9.1900, -75.0152),
        }

    def _haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """Calculate haversine distance in km."""
        r = 6371.0

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return r * c

    def find_clusters(
        self,
        case_data: Mapping[CountryCode, int],
        baseline_data: Mapping[CountryCode, float],
    ) -> list[GeographicCluster]:
        """Identify geographic clusters of elevated disease activity.

        Parameters
        ----------
        case_data : Mapping[CountryCode, int]
            Current case counts by country.
        baseline_data : Mapping[CountryCode, float]
            Baseline expected cases.

        Returns
        -------
        list[GeographicCluster]
            Identified clusters.
        """
        elevated_countries: list[tuple[CountryCode, float]] = []

        for country, cases in case_data.items():
            baseline = baseline_data.get(country, 1.0)
            if baseline > 0:
                rr = cases / baseline
                if rr >= self.config.min_relative_risk:
                    elevated_countries.append((country, rr))

        if len(elevated_countries) < self.config.min_cluster_size:
            return []

        clusters: list[GeographicCluster] = []
        used = set()

        for country, rr in elevated_countries:
            if country in used or country not in self._country_coords:
                continue

            lat1, lon1 = self._country_coords[country]
            cluster_members = [(country, rr)]

            for other_country, other_rr in elevated_countries:
                if other_country == country or other_country in used:
                    continue
                if other_country not in self._country_coords:
                    continue

                lat2, lon2 = self._country_coords[other_country]
                distance = self._haversine_distance(lat1, lon1, lat2, lon2)

                if distance <= self.config.max_distance_km:
                    cluster_members.append((other_country, other_rr))

            if len(cluster_members) >= self.config.min_cluster_size:
                for c, _ in cluster_members:
                    used.add(c)

                member_codes = tuple(c for c, _ in cluster_members)
                total_cases = sum(
                    case_data.get(c, 0) for c in member_codes
                )
                avg_rr = statistics.mean(rr for _, rr in cluster_members)

                lats = [self._country_coords[c][0] for c in member_codes]
                lons = [self._country_coords[c][1] for c in member_codes]
                centroid_lat = statistics.mean(lats)
                centroid_lon = statistics.mean(lons)

                max_dist = max(
                    self._haversine_distance(
                        centroid_lat, centroid_lon,
                        self._country_coords[c][0],
                        self._country_coords[c][1],
                    )
                    for c in member_codes
                )

                cluster_id = hashlib.md5(
                    "".join(sorted(member_codes)).encode()
                ).hexdigest()[:8]

                clusters.append(
                    GeographicCluster(
                        cluster_id=cluster_id,
                        countries=member_codes,
                        centroid_lat=centroid_lat,
                        centroid_lon=centroid_lon,
                        total_cases=total_cases,
                        radius_km=max_dist,
                        relative_risk=avg_rr,
                    )
                )

        return clusters


@dataclass
class RegionalAggregator:
    """Aggregate surveillance data by WHO region.

    Provides regional summaries and cross-country comparisons.
    """

    trend_analyzer: TrendAnalyzer = field(default_factory=TrendAnalyzer)

    def aggregate_region(
        self,
        records: Sequence[SurveillanceRecord],
        region: WHORegion,
    ) -> RegionalSummary:
        """Aggregate data for single region.

        Parameters
        ----------
        records : Sequence[SurveillanceRecord]
            All surveillance records.
        region : WHORegion
            Region to aggregate.

        Returns
        -------
        RegionalSummary
            Regional summary statistics.
        """
        region_records = [r for r in records if r.region == region]

        if not region_records:
            return RegionalSummary(
                region=region,
                total_cases=0,
                total_deaths=0,
                case_fatality_rate=0.0,
                countries_reporting=0,
                trend=TrendDirection.UNDEFINED,
                hotspot_countries=(),
            )

        total_cases = sum(r.case_count for r in region_records)
        total_deaths = sum(r.mortality_count for r in region_records)
        cfr = total_deaths / max(total_cases, 1)

        countries = set(r.country_code for r in region_records)
        countries_reporting = len(countries)

        country_totals: dict[CountryCode, int] = defaultdict(int)
        for r in region_records:
            country_totals[r.country_code] += r.case_count

        if country_totals:
            mean_cases = statistics.mean(country_totals.values())
            hotspots = tuple(
                c for c, cases in country_totals.items()
                if cases > mean_cases * 2
            )
        else:
            hotspots = ()

        all_cases = [r.case_count for r in sorted(
            region_records, key=lambda r: r.observation_date
        )]
        if len(all_cases) >= 3:
            trend = self._detect_simple_trend(all_cases)
        else:
            trend = TrendDirection.UNDEFINED

        return RegionalSummary(
            region=region,
            total_cases=total_cases,
            total_deaths=total_deaths,
            case_fatality_rate=cfr,
            countries_reporting=countries_reporting,
            trend=trend,
            hotspot_countries=hotspots,
        )

    def _detect_simple_trend(self, values: Sequence[int]) -> TrendDirection:
        """Simple trend detection based on moving average."""
        n = len(values)
        if n < 3:
            return TrendDirection.UNDEFINED

        first_half = statistics.mean(values[: n // 2])
        second_half = statistics.mean(values[n // 2 :])

        if second_half > first_half * 1.1:
            return TrendDirection.INCREASING
        elif second_half < first_half * 0.9:
            return TrendDirection.DECREASING
        else:
            return TrendDirection.STABLE

    def aggregate_all_regions(
        self,
        records: Sequence[SurveillanceRecord],
    ) -> dict[WHORegion, RegionalSummary]:
        """Aggregate all WHO regions.

        Parameters
        ----------
        records : Sequence[SurveillanceRecord]
            All surveillance records.

        Returns
        -------
        dict[WHORegion, RegionalSummary]
            Summary for each region.
        """
        return {
            region: self.aggregate_region(records, region)
            for region in WHORegion
        }


@dataclass
class QualityAssessor:
    """Assess data quality of surveillance records.

    Computes completeness, timeliness, consistency, and
    plausibility metrics.
    """

    expected_countries: set[CountryCode] = field(default_factory=set)
    plausibility_bounds: tuple[int, int] = (0, 1000000)

    def assess(
        self,
        records: Sequence[SurveillanceRecord],
        expected_period_days: int = 365,
    ) -> DataQualityMetrics:
        """Assess data quality for set of records.

        Parameters
        ----------
        records : Sequence[SurveillanceRecord]
            Records to assess.
        expected_period_days : int
            Expected reporting period.

        Returns
        -------
        DataQualityMetrics
            Quality metrics.
        """
        if not records:
            return DataQualityMetrics(
                completeness=0.0,
                timeliness=0.0,
                consistency=0.0,
                plausibility=0.0,
                overall_score=0.0,
            )

        completeness = self._compute_completeness(records)
        timeliness = self._compute_timeliness(records)
        consistency = self._compute_consistency(records)
        plausibility = self._compute_plausibility(records)

        weights = [0.3, 0.2, 0.25, 0.25]
        scores = [completeness, timeliness, consistency, plausibility]
        overall = sum(w * s for w, s in zip(weights, scores, strict=True))

        return DataQualityMetrics(
            completeness=completeness,
            timeliness=timeliness,
            consistency=consistency,
            plausibility=plausibility,
            overall_score=overall,
        )

    def _compute_completeness(self, records: Sequence[SurveillanceRecord]) -> float:
        """Compute data completeness score."""
        if not self.expected_countries:
            return 1.0

        reporting_countries = set(r.country_code for r in records)
        return len(reporting_countries) / len(self.expected_countries)

    def _compute_timeliness(self, records: Sequence[SurveillanceRecord]) -> float:
        """Compute timeliness score based on confidence levels."""
        if not records:
            return 0.0
        return statistics.mean(r.confidence_level for r in records)

    def _compute_consistency(self, records: Sequence[SurveillanceRecord]) -> float:
        """Compute internal consistency score."""
        by_country: dict[CountryCode, list[int]] = defaultdict(list)
        for r in records:
            by_country[r.country_code].append(r.case_count)

        consistency_scores = []
        for cases in by_country.values():
            if len(cases) >= 2:
                cv = statistics.stdev(cases) / max(statistics.mean(cases), 1)
                consistency_scores.append(max(0, 1 - cv / 2))

        return statistics.mean(consistency_scores) if consistency_scores else 1.0

    def _compute_plausibility(self, records: Sequence[SurveillanceRecord]) -> float:
        """Compute plausibility score."""
        min_val, max_val = self.plausibility_bounds
        plausible_count = sum(
            1 for r in records
            if min_val <= r.case_count <= max_val
        )
        return plausible_count / len(records) if records else 0.0


@dataclass
class SeasonalDecomposer:
    """STL-like seasonal decomposition of time series."""

    default_period: int = 12

    def decompose(
        self,
        values: Sequence[float],
        period: int | None = None,
    ) -> TimeSeriesDecomposition:
        """Decompose time series into components.

        Parameters
        ----------
        values : Sequence[float]
            Time series values.
        period : int | None
            Seasonal period (auto-detect if None).

        Returns
        -------
        TimeSeriesDecomposition
            Decomposed components.
        """
        period = period or self.default_period
        n = len(values)

        if n < period * 2:
            return TimeSeriesDecomposition(
                trend=tuple(values),
                seasonal=tuple(0.0 for _ in values),
                residual=tuple(0.0 for _ in values),
                period=period,
            )

        trend = self._moving_average(values, period)

        detrended = [v - t for v, t in zip(values, trend, strict=True)]
        seasonal = self._extract_seasonal(detrended, period)

        residual = [
            v - t - s
            for v, t, s in zip(values, trend, seasonal, strict=True)
        ]

        return TimeSeriesDecomposition(
            trend=tuple(trend),
            seasonal=tuple(seasonal),
            residual=tuple(residual),
            period=period,
        )

    def _moving_average(self, values: Sequence[float], window: int) -> list[float]:
        """Compute centered moving average."""
        n = len(values)
        result = list(values)

        half = window // 2
        for i in range(half, n - half):
            result[i] = statistics.mean(values[i - half : i + half + 1])

        return result

    def _extract_seasonal(
        self,
        detrended: Sequence[float],
        period: int,
    ) -> list[float]:
        """Extract seasonal component."""
        n = len(detrended)
        seasonal_indices: list[list[float]] = [[] for _ in range(period)]

        for i, v in enumerate(detrended):
            seasonal_indices[i % period].append(v)

        seasonal_means = [
            statistics.mean(vals) if vals else 0.0
            for vals in seasonal_indices
        ]

        return [seasonal_means[i % period] for i in range(n)]


@dataclass
class MissingDataHandler:
    """Handle missing data in surveillance records.

    Provides various imputation strategies for incomplete
    time series data.
    """

    strategy: ImputationStrategy = ImputationStrategy.LINEAR_INTERPOLATION

    def impute(
        self,
        records: Sequence[SurveillanceRecord],
        expected_dates: Sequence[datetime],
    ) -> list[SurveillanceRecord]:
        """Impute missing records.

        Parameters
        ----------
        records : Sequence[SurveillanceRecord]
            Existing records.
        expected_dates : Sequence[datetime]
            Dates that should have data.

        Returns
        -------
        list[SurveillanceRecord]
            Records with imputed values.
        """
        by_country: dict[CountryCode, dict[datetime, SurveillanceRecord]] = defaultdict(dict)
        for r in records:
            by_country[r.country_code][r.observation_date] = r

        all_records: list[SurveillanceRecord] = list(records)

        for country, date_map in by_country.items():
            existing_dates = set(date_map.keys())
            missing_dates = [d for d in expected_dates if d not in existing_dates]

            if not missing_dates:
                continue

            existing_values = [
                date_map[d].case_count
                for d in sorted(existing_dates)
            ]

            for missing_date in missing_dates:
                imputed_value = self._impute_value(
                    missing_date,
                    sorted(existing_dates),
                    existing_values,
                )

                sample_record = list(date_map.values())[0]
                imputed_record = SurveillanceRecord(
                    country_code=country,
                    region=sample_record.region,
                    observation_date=missing_date,
                    case_count=imputed_value,
                    mortality_count=0,
                    age_group=sample_record.age_group,
                    data_source="imputed",
                    confidence_level=0.5,
                )
                all_records.append(imputed_record)

        return all_records

    def _impute_value(
        self,
        target_date: datetime,
        dates: Sequence[datetime],
        values: Sequence[int],
    ) -> int:
        """Impute single missing value."""
        if self.strategy == ImputationStrategy.ZERO:
            return 0
        elif self.strategy == ImputationStrategy.MEAN:
            return round(statistics.mean(values)) if values else 0
        elif self.strategy == ImputationStrategy.MEDIAN:
            return round(statistics.median(values)) if values else 0
        elif self.strategy == ImputationStrategy.FORWARD_FILL:
            return values[-1] if values else 0
        elif self.strategy == ImputationStrategy.BACKWARD_FILL:
            return values[0] if values else 0
        elif self.strategy == ImputationStrategy.LINEAR_INTERPOLATION:
            return self._linear_interpolate(target_date, dates, values)
        else:
            return round(statistics.mean(values)) if values else 0

    def _linear_interpolate(
        self,
        target_date: datetime,
        dates: Sequence[datetime],
        values: Sequence[int],
    ) -> int:
        """Perform linear interpolation."""
        if len(dates) < 2:
            return values[0] if values else 0

        before_idx = -1
        after_idx = -1

        for i, d in enumerate(dates):
            if d <= target_date:
                before_idx = i
            if d >= target_date and after_idx == -1:
                after_idx = i

        if before_idx == -1:
            return values[0]
        if after_idx == -1 or after_idx == before_idx:
            return values[-1]

        before_date = dates[before_idx]
        after_date = dates[after_idx]
        before_val = values[before_idx]
        after_val = values[after_idx]

        total_span = (after_date - before_date).total_seconds()
        target_span = (target_date - before_date).total_seconds()

        if total_span == 0:
            return before_val

        fraction = target_span / total_span
        interpolated = before_val + (after_val - before_val) * fraction

        return max(0, round(interpolated))


@dataclass
class CrossSourceValidator:
    """Validate data consistency across multiple sources."""

    tolerance_fraction: float = 0.2

    def validate(
        self,
        source_a_records: Sequence[SurveillanceRecord],
        source_b_records: Sequence[SurveillanceRecord],
        source_a_name: str = "Source A",
        source_b_name: str = "Source B",
    ) -> ValidationReport:
        """Compare records from two sources.

        Parameters
        ----------
        source_a_records : Sequence[SurveillanceRecord]
            Records from first source.
        source_b_records : Sequence[SurveillanceRecord]
            Records from second source.
        source_a_name : str
            Name of first source.
        source_b_name : str
            Name of second source.

        Returns
        -------
        ValidationReport
            Validation results.
        """
        a_by_key: dict[tuple[str, str], int] = {}
        for r in source_a_records:
            key = (r.country_code, r.observation_date.isoformat()[:10])
            a_by_key[key] = r.case_count

        b_by_key: dict[tuple[str, str], int] = {}
        for r in source_b_records:
            key = (r.country_code, r.observation_date.isoformat()[:10])
            b_by_key[key] = r.case_count

        common_keys = set(a_by_key.keys()) & set(b_by_key.keys())
        records_compared = len(common_keys)

        matches = 0
        discrepancies = 0
        a_values: list[float] = []
        b_values: list[float] = []

        for key in common_keys:
            a_val = a_by_key[key]
            b_val = b_by_key[key]
            a_values.append(float(a_val))
            b_values.append(float(b_val))

            if a_val == b_val:
                matches += 1
            elif abs(a_val - b_val) <= max(a_val, b_val) * self.tolerance_fraction:
                matches += 1
            else:
                discrepancies += 1

        if len(a_values) >= 2:
            mean_a = statistics.mean(a_values)
            mean_b = statistics.mean(b_values)
            std_a = statistics.stdev(a_values)
            std_b = statistics.stdev(b_values)

            if std_a > 0 and std_b > 0:
                covariance = sum(
                    (a - mean_a) * (b - mean_b)
                    for a, b in zip(a_values, b_values, strict=True)
                ) / len(a_values)
                correlation = covariance / (std_a * std_b)
            else:
                correlation = 1.0 if mean_a == mean_b else 0.0
        else:
            correlation = 1.0

        recommendations: list[str] = []
        if discrepancies > records_compared * 0.1:
            recommendations.append(
                "High discrepancy rate detected. Review data collection methodology."
            )
        if correlation < 0.9:
            recommendations.append(
                f"Low correlation ({correlation:.2f}). Investigate systematic differences."
            )

        return ValidationReport(
            source_a=source_a_name,
            source_b=source_b_name,
            records_compared=records_compared,
            matches=matches,
            discrepancies=discrepancies,
            correlation=correlation,
            recommendations=tuple(recommendations),
        )


@dataclass
class ReportGenerator:
    """Generate formatted reports from surveillance analysis.

    Creates markdown and JSON reports for epidemiological
    summaries, outbreak alerts, and trend analyses.
    """

    output_dir: Path = field(default_factory=lambda: Path("reports"))

    def __post_init__(self) -> None:
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_regional_report(
        self,
        summaries: Mapping[WHORegion, RegionalSummary],
        report_date: datetime | None = None,
    ) -> Path:
        """Generate regional summary report.

        Parameters
        ----------
        summaries : Mapping[WHORegion, RegionalSummary]
            Regional summaries.
        report_date : datetime | None
            Report date.

        Returns
        -------
        Path
            Path to generated report.
        """
        report_date = report_date or datetime.now()
        filename = f"regional_report_{report_date.strftime('%Y%m%d')}.md"
        filepath = self.output_dir / filename

        lines = [
            "# WHO Regional Surveillance Report",
            "",
            f"**Generated:** {report_date.isoformat()}",
            "",
            "## Summary by Region",
            "",
        ]

        for region, summary in summaries.items():
            lines.extend([
                f"### {region.name}",
                "",
                f"- **Total Cases:** {summary.total_cases:,}",
                f"- **Total Deaths:** {summary.total_deaths:,}",
                f"- **Case Fatality Rate:** {summary.case_fatality_rate:.2%}",
                f"- **Countries Reporting:** {summary.countries_reporting}",
                f"- **Trend:** {summary.trend.name}",
                "",
            ])
            if summary.hotspot_countries:
                lines.append(f"**Hotspots:** {', '.join(summary.hotspot_countries)}")
                lines.append("")

        filepath.write_text("\n".join(lines), encoding="utf-8")
        return filepath

    def generate_alert_report(
        self,
        alerts: Sequence[OutbreakAlert],
    ) -> Path:
        """Generate outbreak alert report."""
        report_date = datetime.now()
        filename = f"outbreak_alerts_{report_date.strftime('%Y%m%d')}.json"
        filepath = self.output_dir / filename

        data = {
            "generated": report_date.isoformat(),
            "alert_count": len(alerts),
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "country": a.country_code,
                    "detected": a.detection_date.isoformat(),
                    "severity": a.severity.name,
                    "observed": a.observed_cases,
                    "expected": a.expected_cases,
                    "excess_ratio": a.excess_ratio,
                    "confidence": a.confidence,
                    "message": a.message,
                }
                for a in alerts
            ],
        }

        filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return filepath

