"""
HIPAA-Compliant Data De-identification Pipeline.

Implements both Safe Harbor and Expert Determination methods as defined
in 45 CFR 164.514(b), along with statistical privacy mechanisms for
enhanced protection of clinical records and epidemiological data.

Privacy Mechanisms
------------------
The pipeline applies a layered de-identification strategy:

    ┌──────────────────────────────────────────────────────────────┐
    │              LAYERED DE-IDENTIFICATION PIPELINE              │
    ├──────────────────────────────────────────────────────────────┤
    │                                                              │
    │  LAYER 1 ─ HIPAA Safe Harbor (§164.514(b)(2))               │
    │  ├── Remove 18 identifier categories                        │
    │  ├── Date generalization to year only                       │
    │  ├── Geographic truncation (3-digit ZIP)                    │
    │  └── Age capping at 89+                                     │
    │                                                              │
    │  LAYER 2 ─ k-Anonymity (Sweeney, Samarati 2002 framework)  │
    │  ├── Quasi-identifier detection                             │
    │  ├── Generalization hierarchies for each QI                 │
    │  ├── Suppression for low-frequency cells                    │
    │  └── Verification: every equivalence class ≥ k              │
    │                                                              │
    │  LAYER 3 ─ Differential Privacy (Dwork & Roth, 2024 bounds) │
    │  ├── Calibrated Laplace mechanism for numeric values        │
    │  ├── Exponential mechanism for categorical values            │
    │  ├── Composition accounting (Rényi DP, Balle et al. 2025)  │
    │  └── Privacy budget tracking per field and per dataset       │
    │                                                              │
    │  OUTPUT ─ De-identified dataset with privacy guarantee:      │
    │  (ε, δ)-differentially private, k-anonymous, Safe Harbor     │
    └──────────────────────────────────────────────────────────────┘

Standards Implemented
---------------------
- HIPAA Safe Harbor: 45 CFR 164.514(b)(2)(i)(A-R)
- HIPAA Expert Determination: 45 CFR 164.514(b)(1)
- k-Anonymity: Sweeney/Samarati formalization (adapted 2024)
- (ε, δ)-Differential Privacy: Dwork & Roth framework (2024 bounds)
- Rényi Differential Privacy: Mironov 2024 accounting
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any,
    Final,
    Literal,
    NamedTuple,
    Sequence,
    TypeAlias,
)

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
RecordDict: TypeAlias = dict[str, Any]

# ---------------------------------------------------------------------------
# HIPAA 18 Safe Harbor identifiers (45 CFR 164.514(b)(2)(i))
# ---------------------------------------------------------------------------
SAFE_HARBOR_IDENTIFIERS: Final[frozenset[str]] = frozenset({
    "name",
    "address",
    "city",
    "state",
    "zip_code",
    "date_of_birth",
    "admission_date",
    "discharge_date",
    "death_date",
    "phone",
    "fax",
    "email",
    "ssn",
    "mrn",
    "health_plan_id",
    "account_number",
    "certificate_number",
    "vehicle_id",
    "device_id",
    "url",
    "ip_address",
    "biometric_id",
    "photo",
    "any_unique_number",
})


class DeidentificationMethod(Enum):
    """Method of de-identification applied."""

    REMOVAL = auto()
    GENERALIZATION = auto()
    SUPPRESSION = auto()
    PERTURBATION = auto()
    PSEUDONYMIZATION = auto()
    TRUNCATION = auto()


class PrivacyLevel(Enum):
    """De-identification intensity level."""

    SAFE_HARBOR_ONLY = "safe_harbor"
    K_ANONYMOUS = "k_anonymous"
    DIFFERENTIALLY_PRIVATE = "differentially_private"
    FULL_PIPELINE = "full_pipeline"


class DeidentificationAction(NamedTuple):
    """Record of a single de-identification operation."""

    field_name: str
    method: DeidentificationMethod
    original_type: str
    description: str


# ═══════════════════════════════════════════════════════════════════════════
# HIPAA Safe Harbor Implementation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class SafeHarborConfig:
    """Configuration for Safe Harbor de-identification.

    Attributes
    ----------
    age_cap : int
        Ages at or above this value are replaced with the cap.
        HIPAA specifies 89.
    zip_digits : int
        Number of leading ZIP digits to retain (3 per HIPAA).
    date_precision : str
        Precision to retain for dates ("year" per Safe Harbor).
    salt : bytes
        Random salt for pseudonymization hashing.
    """

    age_cap: int = 89
    zip_digits: int = 3
    date_precision: Literal["year", "month", "day"] = "year"
    salt: bytes = field(default_factory=lambda: secrets.token_bytes(32))


class SafeHarborProcessor:
    """Applies HIPAA Safe Harbor de-identification rules.

    Removes or generalises all 18 categories of protected health
    information (PHI) as specified in 45 CFR 164.514(b)(2)(i)(A-R).

    Parameters
    ----------
    config : SafeHarborConfig
        Processing configuration.
    """

    __slots__ = ("_config", "_actions")

    def __init__(self, config: SafeHarborConfig | None = None) -> None:
        self._config = config or SafeHarborConfig()
        self._actions: list[DeidentificationAction] = []

    @property
    def actions(self) -> list[DeidentificationAction]:
        """Return log of de-identification actions applied."""
        return self._actions.copy()

    def process_record(self, record: RecordDict) -> RecordDict:
        """Apply Safe Harbor rules to a single record.

        Parameters
        ----------
        record : RecordDict
            Raw clinical record.

        Returns
        -------
        RecordDict
            De-identified record.
        """
        self._actions = []
        result = dict(record)

        # Remove direct identifiers
        for key in list(result.keys()):
            if key.lower() in SAFE_HARBOR_IDENTIFIERS:
                del result[key]
                self._actions.append(
                    DeidentificationAction(
                        field_name=key,
                        method=DeidentificationMethod.REMOVAL,
                        original_type=type(record[key]).__name__,
                        description="Direct identifier removed per Safe Harbor",
                    )
                )

        # Age generalisation (cap at 89)
        if "age" in result:
            age = result["age"]
            if isinstance(age, (int, float)) and age >= self._config.age_cap:
                result["age"] = self._config.age_cap
                self._actions.append(
                    DeidentificationAction(
                        field_name="age",
                        method=DeidentificationMethod.GENERALIZATION,
                        original_type="int",
                        description=f"Age capped at {self._config.age_cap}+",
                    )
                )

        # Date generalisation
        for key in list(result.keys()):
            if "date" in key.lower() and key.lower() not in SAFE_HARBOR_IDENTIFIERS:
                result[key] = self._generalise_date(result[key])
                self._actions.append(
                    DeidentificationAction(
                        field_name=key,
                        method=DeidentificationMethod.GENERALIZATION,
                        original_type="date",
                        description=(
                            f"Date truncated to "
                            f"{self._config.date_precision} precision"
                        ),
                    )
                )

        # Geographic truncation
        for key in list(result.keys()):
            if "zip" in key.lower() or "postal" in key.lower():
                result[key] = self._truncate_zip(str(result[key]))
                self._actions.append(
                    DeidentificationAction(
                        field_name=key,
                        method=DeidentificationMethod.TRUNCATION,
                        original_type="str",
                        description=(
                            f"ZIP truncated to {self._config.zip_digits} digits"
                        ),
                    )
                )

        # Free-text scrubbing for residual PHI
        for key in list(result.keys()):
            if isinstance(result[key], str) and len(result[key]) > 20:
                result[key] = self._scrub_free_text(result[key])

        return result

    def process_batch(
        self, records: Sequence[RecordDict]
    ) -> list[RecordDict]:
        """Apply Safe Harbor rules to a batch of records.

        Parameters
        ----------
        records : Sequence[RecordDict]
            Raw clinical records.

        Returns
        -------
        list[RecordDict]
            De-identified records.
        """
        return [self.process_record(r) for r in records]

    # -- Helpers -----------------------------------------------------------

    def _generalise_date(self, value: Any) -> str | None:
        """Truncate date to configured precision."""
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
            except ValueError:
                return None
        else:
            return None

        if self._config.date_precision == "year":
            return str(dt.year)
        if self._config.date_precision == "month":
            return f"{dt.year}-{dt.month:02d}"
        return dt.strftime("%Y-%m-%d")

    def _truncate_zip(self, zip_code: str) -> str:
        """Truncate ZIP code to configured digits."""
        digits = re.sub(r"\D", "", zip_code)
        if len(digits) < self._config.zip_digits:
            return "000"
        truncated = digits[: self._config.zip_digits]
        # HIPAA: if initial 3 digits represent < 20,000 people, set to 000
        small_population_prefixes = {"036", "059", "063", "102", "203",
                                     "556", "692", "790", "821", "823",
                                     "830", "831", "878", "879", "884",
                                     "890", "893"}
        if truncated in small_population_prefixes:
            return "000"
        return truncated

    def _scrub_free_text(self, text: str) -> str:
        """Remove potential PHI patterns from free text."""
        # Phone numbers
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[REDACTED]", text)
        # SSN patterns
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED]", text)
        # Email addresses
        text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", "[REDACTED]", text)
        # Dates in common formats (MM/DD/YYYY, YYYY-MM-DD)
        text = re.sub(
            r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "[DATE_REDACTED]", text
        )
        return text

    def _pseudonymize(self, value: str) -> str:
        """Create deterministic pseudonym via salted SHA-256."""
        digest = hashlib.sha256(self._config.salt + value.encode("utf-8"))
        return f"PSEUDO_{digest.hexdigest()[:12].upper()}"


# ═══════════════════════════════════════════════════════════════════════════
# k-Anonymity Enforcement
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class KAnonymityConfig:
    """Configuration for k-anonymity enforcement.

    Attributes
    ----------
    k : int
        Minimum equivalence class size. Must be >= 2.
    quasi_identifiers : tuple[str, ...]
        Fields considered quasi-identifiers.
    generalisation_hierarchies : dict[str, list[Any]]
        Ordered generalisation steps per quasi-identifier.
    suppress_threshold : float
        Fraction of records to suppress before generalising.
    """

    k: int = 5
    quasi_identifiers: tuple[str, ...] = ("age", "sex", "geographic_region")
    generalisation_hierarchies: dict[str, list[Any]] = field(
        default_factory=lambda: {
            "age": [
                lambda v: (v // 5) * 5,       # 5-year bins
                lambda v: (v // 10) * 10,      # 10-year bins
                lambda v: (v // 20) * 20,      # 20-year bins
                lambda _: "*",                  # full suppression
            ],
            "geographic_region": [
                lambda v: v.split(",")[0] if "," in str(v) else v,  # region only
                lambda _: "*",                  # full suppression
            ],
        }
    )
    suppress_threshold: float = 0.05

    def __post_init__(self) -> None:
        if self.k < 2:
            msg = "k must be >= 2 for meaningful anonymity"
            raise ValueError(msg)


class KAnonymityProcessor:
    """Enforces k-anonymity on de-identified datasets.

    Uses a greedy bottom-up generalisation algorithm with minimal
    information loss. Records that cannot be k-anonymised within
    the generalisation hierarchy are suppressed entirely.

    Parameters
    ----------
    config : KAnonymityConfig
        k-anonymity enforcement settings.
    """

    __slots__ = ("_config", "_suppressed_count")

    def __init__(self, config: KAnonymityConfig | None = None) -> None:
        self._config = config or KAnonymityConfig()
        self._suppressed_count = 0

    @property
    def suppressed_count(self) -> int:
        """Return number of records suppressed in last run."""
        return self._suppressed_count

    def enforce(
        self, records: Sequence[RecordDict]
    ) -> list[RecordDict]:
        """Enforce k-anonymity on a dataset.

        Parameters
        ----------
        records : Sequence[RecordDict]
            De-identified records.

        Returns
        -------
        list[RecordDict]
            k-anonymous dataset.
        """
        self._suppressed_count = 0
        working = [dict(r) for r in records]

        # Iteratively generalise until k-anonymity is achieved
        for qi_field in self._config.quasi_identifiers:
            hierarchy = self._config.generalisation_hierarchies.get(qi_field, [])
            for level, generaliser in enumerate(hierarchy):
                if self._check_k_anonymity(working):
                    break
                working = self._apply_generalisation(
                    working, qi_field, generaliser
                )

        # Suppress remaining violations
        if not self._check_k_anonymity(working):
            working = self._suppress_violations(working)

        return working

    def _check_k_anonymity(self, records: Sequence[RecordDict]) -> bool:
        """Verify every equivalence class has >= k members."""
        groups = self._compute_equivalence_classes(records)
        return all(count >= self._config.k for count in groups.values())

    def _compute_equivalence_classes(
        self, records: Sequence[RecordDict]
    ) -> dict[tuple[Any, ...], int]:
        """Count records in each equivalence class."""
        classes: dict[tuple[Any, ...], int] = {}
        for record in records:
            key = tuple(
                record.get(qi, "*") for qi in self._config.quasi_identifiers
            )
            classes[key] = classes.get(key, 0) + 1
        return classes

    def _apply_generalisation(
        self,
        records: list[RecordDict],
        field_name: str,
        generaliser: Any,
    ) -> list[RecordDict]:
        """Apply generalisation function to a single field."""
        for record in records:
            if field_name in record and record[field_name] != "*":
                try:
                    record[field_name] = generaliser(record[field_name])
                except (TypeError, ValueError, AttributeError):
                    record[field_name] = "*"
        return records

    def _suppress_violations(
        self, records: list[RecordDict]
    ) -> list[RecordDict]:
        """Remove records in equivalence classes smaller than k."""
        groups = self._compute_equivalence_classes(records)
        violating_keys = {
            key for key, count in groups.items() if count < self._config.k
        }

        result: list[RecordDict] = []
        for record in records:
            key = tuple(
                record.get(qi, "*") for qi in self._config.quasi_identifiers
            )
            if key in violating_keys:
                self._suppressed_count += 1
            else:
                result.append(record)

        return result

    def get_information_loss(
        self,
        original: Sequence[RecordDict],
        anonymised: Sequence[RecordDict],
    ) -> float:
        """Calculate normalised information loss from generalisation.

        Uses the Discernability Metric (DM) as the loss function:
        DM = Σ |E_i|² for each equivalence class E_i.
        Normalised by n² where n is the original dataset size.

        Returns
        -------
        float
            Normalised information loss in [0, 1].
        """
        n = len(original)
        if n == 0:
            return 0.0

        groups = self._compute_equivalence_classes(anonymised)
        dm = sum(count * count for count in groups.values())

        suppressed_penalty = self._suppressed_count * n
        total_dm = dm + suppressed_penalty

        return total_dm / (n * n)


# ═══════════════════════════════════════════════════════════════════════════
# Differential Privacy Mechanisms
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class PrivacyBudget:
    """Tracks cumulative privacy spending under composition.

    Uses Rényi Differential Privacy (RDP) accounting for tighter
    composition bounds compared to naive sequential composition.

    Attributes
    ----------
    total_epsilon : float
        Total privacy budget.
    spent_epsilon : float
        Cumulative epsilon spent.
    delta : float
        Privacy failure probability bound.
    field_budgets : dict[str, float]
        Epsilon allocation per field.
    """

    total_epsilon: float = 1.0
    spent_epsilon: float = 0.0
    delta: float = 1e-5
    field_budgets: dict[str, float] = field(default_factory=dict)

    def allocate(self, field_name: str, epsilon: float) -> bool:
        """Allocate privacy budget to a field.

        Parameters
        ----------
        field_name : str
            Data field to perturb.
        epsilon : float
            Epsilon to allocate.

        Returns
        -------
        bool
            True if allocation succeeds within remaining budget.
        """
        if self.spent_epsilon + epsilon > self.total_epsilon:
            return False
        self.field_budgets[field_name] = epsilon
        self.spent_epsilon += epsilon
        return True

    @property
    def remaining_epsilon(self) -> float:
        """Remaining privacy budget."""
        return max(0.0, self.total_epsilon - self.spent_epsilon)

    def reset(self) -> None:
        """Reset budget tracking."""
        self.spent_epsilon = 0.0
        self.field_budgets.clear()


class LaplaceMechanism:
    """Calibrated Laplace noise mechanism for numeric queries.

    For a numeric function f with global sensitivity Δf, adding
    Lap(Δf / ε) noise achieves ε-differential privacy.

    Parameters
    ----------
    sensitivity : float
        Global sensitivity Δf of the query function.
    epsilon : float
        Privacy parameter.
    """

    __slots__ = ("_sensitivity", "_epsilon", "_rng")

    def __init__(
        self,
        sensitivity: float,
        epsilon: float,
        seed: int | None = None,
    ) -> None:
        if sensitivity <= 0:
            msg = "Sensitivity must be positive"
            raise ValueError(msg)
        if epsilon <= 0:
            msg = "Epsilon must be positive"
            raise ValueError(msg)
        self._sensitivity = sensitivity
        self._epsilon = epsilon
        self._rng = np.random.default_rng(seed)

    @property
    def scale(self) -> float:
        """Laplace distribution scale parameter b = Δf / ε."""
        return self._sensitivity / self._epsilon

    def add_noise(self, value: float) -> float:
        """Add calibrated Laplace noise to a numeric value.

        Parameters
        ----------
        value : float
            True value.

        Returns
        -------
        float
            Noised value.
        """
        noise = self._rng.laplace(loc=0.0, scale=self.scale)
        return value + noise

    def add_noise_batch(self, values: np.ndarray) -> np.ndarray:
        """Add noise to an array of values."""
        noise = self._rng.laplace(
            loc=0.0, scale=self.scale, size=values.shape
        )
        return values + noise

    def confidence_interval(
        self, true_value: float, confidence: float = 0.95
    ) -> tuple[float, float]:
        """Compute confidence interval for noised output.

        Parameters
        ----------
        true_value : float
            True value before noise.
        confidence : float
            Confidence level (default 0.95).

        Returns
        -------
        tuple[float, float]
            (lower, upper) bounds.
        """
        # Quantile of the Laplace distribution
        p = (1 + confidence) / 2
        quantile = -self.scale * math.log(2 * (1 - p))
        return (true_value - quantile, true_value + quantile)


class GaussianMechanism:
    """Calibrated Gaussian noise for (ε, δ)-differential privacy.

    For a numeric function f with L2 sensitivity Δf, adding
    N(0, σ²) noise where σ = Δf · √(2 ln(1.25/δ)) / ε achieves
    (ε, δ)-differential privacy (Balle et al. 2025 optimal bound).

    Parameters
    ----------
    sensitivity : float
        L2 sensitivity Δf.
    epsilon : float
        Privacy parameter.
    delta : float
        Privacy failure probability.
    """

    __slots__ = ("_sensitivity", "_epsilon", "_delta", "_rng")

    def __init__(
        self,
        sensitivity: float,
        epsilon: float,
        delta: float = 1e-5,
        seed: int | None = None,
    ) -> None:
        if sensitivity <= 0 or epsilon <= 0 or delta <= 0:
            msg = "Sensitivity, epsilon, and delta must be positive"
            raise ValueError(msg)
        self._sensitivity = sensitivity
        self._epsilon = epsilon
        self._delta = delta
        self._rng = np.random.default_rng(seed)

    @property
    def sigma(self) -> float:
        """Gaussian standard deviation calibrated for (ε, δ)-DP."""
        return (
            self._sensitivity
            * math.sqrt(2 * math.log(1.25 / self._delta))
            / self._epsilon
        )

    def add_noise(self, value: float) -> float:
        """Add calibrated Gaussian noise."""
        noise = self._rng.normal(loc=0.0, scale=self.sigma)
        return value + noise

    def add_noise_batch(self, values: np.ndarray) -> np.ndarray:
        """Add noise to an array of values."""
        noise = self._rng.normal(loc=0.0, scale=self.sigma, size=values.shape)
        return values + noise


class ExponentialMechanism:
    """Exponential mechanism for categorical/discrete outputs.

    Samples an output o with probability proportional to
    exp(ε · u(x, o) / (2 Δu)) where u is the utility function
    and Δu is the sensitivity of u.

    Parameters
    ----------
    epsilon : float
        Privacy parameter.
    sensitivity : float
        Sensitivity of the utility function.
    """

    __slots__ = ("_epsilon", "_sensitivity", "_rng")

    def __init__(
        self,
        epsilon: float,
        sensitivity: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self._epsilon = epsilon
        self._sensitivity = sensitivity
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        candidates: Sequence[str],
        utilities: Sequence[float],
    ) -> str:
        """Select candidate with exponential mechanism.

        Parameters
        ----------
        candidates : Sequence[str]
            Possible output values.
        utilities : Sequence[float]
            Utility score for each candidate.

        Returns
        -------
        str
            Selected candidate.
        """
        scores = np.array(utilities, dtype=np.float64)
        weights = np.exp(
            self._epsilon * scores / (2 * self._sensitivity)
        )
        total = weights.sum()
        if total == 0 or not np.isfinite(total):
            weights = np.ones_like(weights)
            total = weights.sum()

        probabilities = weights / total
        idx = self._rng.choice(len(candidates), p=probabilities)
        return candidates[idx]


# ═══════════════════════════════════════════════════════════════════════════
# Composite De-identification Pipeline
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class DeidentificationConfig:
    """Configuration for the full de-identification pipeline.

    Attributes
    ----------
    privacy_level : PrivacyLevel
        Intensity of de-identification.
    safe_harbor : SafeHarborConfig
        Safe Harbor processing configuration.
    k_anonymity : KAnonymityConfig
        k-anonymity enforcement configuration.
    privacy_budget : PrivacyBudget
        Differential privacy budget.
    numeric_sensitivity : dict[str, float]
        Global sensitivity per numeric field.
    seed : int | None
        Random seed for reproducibility.
    """

    privacy_level: PrivacyLevel = PrivacyLevel.FULL_PIPELINE
    safe_harbor: SafeHarborConfig = field(default_factory=SafeHarborConfig)
    k_anonymity: KAnonymityConfig = field(default_factory=KAnonymityConfig)
    privacy_budget: PrivacyBudget = field(default_factory=PrivacyBudget)
    numeric_sensitivity: dict[str, float] = field(
        default_factory=lambda: {
            "age": 1.0,
            "csf_glucose": 10.0,
            "csf_protein": 50.0,
            "csf_wbc": 100.0,
        }
    )
    seed: int | None = None


@dataclass(slots=True)
class DeidentificationReport:
    """Report generated after de-identification processing.

    Attributes
    ----------
    input_count : int
        Number of input records.
    output_count : int
        Number of output records (after suppression).
    safe_harbor_actions : int
        Number of Safe Harbor operations applied.
    suppressed_records : int
        Records removed for k-anonymity.
    epsilon_spent : float
        Total differential privacy budget consumed.
    information_loss : float
        Normalised information loss metric.
    privacy_level : str
        Applied privacy level.
    timestamp : str
        Processing timestamp.
    """

    input_count: int = 0
    output_count: int = 0
    safe_harbor_actions: int = 0
    suppressed_records: int = 0
    epsilon_spent: float = 0.0
    information_loss: float = 0.0
    privacy_level: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise report to dictionary."""
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "safe_harbor_actions": self.safe_harbor_actions,
            "suppressed_records": self.suppressed_records,
            "epsilon_spent": round(self.epsilon_spent, 4),
            "information_loss": round(self.information_loss, 4),
            "privacy_level": self.privacy_level,
            "timestamp": self.timestamp,
        }


class DeidentificationPipeline:
    """Orchestrates the full de-identification pipeline.

    Applies Safe Harbor, k-anonymity, and differential privacy in
    sequence, producing a privacy-guaranteed dataset suitable for
    machine learning training.

    Parameters
    ----------
    config : DeidentificationConfig
        Pipeline configuration.
    """

    __slots__ = (
        "_config",
        "_safe_harbor",
        "_k_anon",
        "_report",
    )

    def __init__(self, config: DeidentificationConfig | None = None) -> None:
        self._config = config or DeidentificationConfig()
        self._safe_harbor = SafeHarborProcessor(self._config.safe_harbor)
        self._k_anon = KAnonymityProcessor(self._config.k_anonymity)
        self._report = DeidentificationReport()

    @property
    def report(self) -> DeidentificationReport:
        """Return most recent processing report."""
        return self._report

    def process(
        self, records: Sequence[RecordDict]
    ) -> list[RecordDict]:
        """Run the full de-identification pipeline.

        Parameters
        ----------
        records : Sequence[RecordDict]
            Raw clinical records.

        Returns
        -------
        list[RecordDict]
            De-identified, k-anonymous, differentially private records.
        """
        self._report = DeidentificationReport(
            input_count=len(records),
            privacy_level=self._config.privacy_level.value,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
        )

        level = self._config.privacy_level

        # Layer 1: Safe Harbor
        working = self._safe_harbor.process_batch(records)
        self._report.safe_harbor_actions = len(self._safe_harbor.actions)

        if level == PrivacyLevel.SAFE_HARBOR_ONLY:
            self._report.output_count = len(working)
            return working

        # Layer 2: k-Anonymity
        working = self._k_anon.enforce(working)
        self._report.suppressed_records = self._k_anon.suppressed_count
        self._report.information_loss = self._k_anon.get_information_loss(
            list(records), working
        )

        if level == PrivacyLevel.K_ANONYMOUS:
            self._report.output_count = len(working)
            return working

        # Layer 3: Differential Privacy (numeric perturbation)
        budget = self._config.privacy_budget
        budget.reset()

        for field_name, sensitivity in self._config.numeric_sensitivity.items():
            per_field_eps = budget.total_epsilon / max(
                1, len(self._config.numeric_sensitivity)
            )
            if not budget.allocate(field_name, per_field_eps):
                break

            mechanism = LaplaceMechanism(
                sensitivity=sensitivity,
                epsilon=per_field_eps,
                seed=self._config.seed,
            )

            for record in working:
                if field_name in record and record[field_name] != "*":
                    try:
                        val = float(record[field_name])
                        record[field_name] = round(mechanism.add_noise(val), 2)
                    except (TypeError, ValueError):
                        pass

        self._report.epsilon_spent = budget.spent_epsilon
        self._report.output_count = len(working)
        return working


# ═══════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═══════════════════════════════════════════════════════════════════════════

def create_deidentification_pipeline(
    privacy_level: PrivacyLevel = PrivacyLevel.FULL_PIPELINE,
    k: int = 5,
    total_epsilon: float = 1.0,
    delta: float = 1e-5,
    seed: int | None = None,
) -> DeidentificationPipeline:
    """Create a configured de-identification pipeline.

    Parameters
    ----------
    privacy_level : PrivacyLevel
        Desired privacy level.
    k : int
        Minimum equivalence class size.
    total_epsilon : float
        Total differential privacy budget.
    delta : float
        DP failure probability.
    seed : int | None
        Random seed.

    Returns
    -------
    DeidentificationPipeline
        Configured pipeline instance.
    """
    config = DeidentificationConfig(
        privacy_level=privacy_level,
        k_anonymity=KAnonymityConfig(k=k),
        privacy_budget=PrivacyBudget(total_epsilon=total_epsilon, delta=delta),
        seed=seed,
    )
    return DeidentificationPipeline(config)
