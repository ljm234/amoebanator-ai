"""
Clinical Records Parser and Validator.

Provides parsers for structured clinical data in CSV and JSON formats,
with comprehensive validation for medical record integrity. Implements
schema enforcement, missing value handling, and type coercion for
downstream machine learning consumption.

Data Model
----------
Clinical records follow a standardized schema:

    ┌─────────────────────────────────────────────────────────────┐
    │                    CLINICAL RECORD SCHEMA                   │
    ├─────────────────────────────────────────────────────────────┤
    │  Patient Demographics                                       │
    │  ├── age: int (0-120)                                      │
    │  ├── sex: Literal["M", "F", "U"]                           │
    │  └── weight_kg: float (optional)                           │
    │                                                             │
    │  Symptom Data                                               │
    │  ├── onset_date: datetime                                  │
    │  ├── symptoms: list[str]                                   │
    │  └── duration_hours: int                                   │
    │                                                             │
    │  Laboratory Values                                          │
    │  ├── csf_glucose: float (mg/dL)                           │
    │  ├── csf_protein: float (mg/dL)                           │
    │  ├── csf_wbc: int (cells/μL)                              │
    │  └── csf_rbc: int (cells/μL)                              │
    │                                                             │
    │  Diagnostic Tests                                           │
    │  ├── microscopy: int (0=neg, 1=pos)                        │
    │  ├── pcr: int (0=neg, 1=pos)                               │
    │  └── culture: int (0=neg, 1=pos)                           │
    │                                                             │
    │  Epidemiology                                               │
    │  ├── exposure_type: str                                    │
    │  ├── water_source: str                                     │
    │  └── geographic_region: str                                │
    │                                                             │
    │  Outcome                                                    │
    │  ├── diagnosis: str                                        │
    │  ├── outcome: Literal["survived", "deceased", "unknown"]   │
    │  └── days_to_outcome: int                                  │
    └─────────────────────────────────────────────────────────────┘

Classes
-------
ClinicalRecordParser
    Primary interface for parsing clinical record files.
CSVRecordParser
    Specialized parser for CSV format records.
JSONRecordParser
    Specialized parser for JSON format records.
RecordValidator
    Validates records against clinical schema.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Final,
    Literal,
    NamedTuple,
    Sequence,
    TypeAlias,
)

import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Type aliases
PathLike: TypeAlias = str | Path
SexType: TypeAlias = Literal["M", "F", "U"]
OutcomeType: TypeAlias = Literal["survived", "deceased", "unknown"]

# Constants
REQUIRED_FIELDS: Final[frozenset[str]] = frozenset({
    "age",
    "csf_wbc",
    "microscopy",
})

OPTIONAL_FIELDS: Final[frozenset[str]] = frozenset({
    "sex",
    "weight_kg",
    "csf_glucose",
    "csf_protein",
    "csf_rbc",
    "pcr",
    "culture",
    "symptoms",
    "onset_date",
    "duration_hours",
    "exposure_type",
    "water_source",
    "geographic_region",
    "diagnosis",
    "outcome",
    "days_to_outcome",
})

# Valid ranges for numeric fields (clinical plausibility)
VALID_RANGES: Final[dict[str, tuple[float, float]]] = {
    "age": (0, 120),
    "weight_kg": (0.5, 500),
    "csf_glucose": (0, 500),
    "csf_protein": (0, 5000),
    "csf_wbc": (0, 100000),
    "csf_rbc": (0, 1000000),
    "duration_hours": (0, 8760),  # Up to 1 year
    "days_to_outcome": (0, 365),
}

# Symptom vocabulary for PAM
KNOWN_SYMPTOMS: Final[frozenset[str]] = frozenset({
    "headache",
    "fever",
    "nausea",
    "vomiting",
    "stiff_neck",
    "altered_mental_status",
    "seizures",
    "photophobia",
    "hallucinations",
    "ataxia",
    "cranial_nerve_palsy",
    "coma",
    "frontal_headache",
    "meningismus",
    "lethargy",
})


class ValidationSeverity(Enum):
    """Severity level for validation issues."""

    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class ValidationResult(NamedTuple):
    """Result of a single validation check.

    Attributes
    ----------
    field_name : str
        Name of the validated field.
    is_valid : bool
        Whether the field passed validation.
    severity : ValidationSeverity
        Severity of any issue found.
    message : str
        Description of the validation result.
    original_value : Any
        Original value before coercion.
    corrected_value : Any | None
        Corrected value if applicable.
    """

    field_name: str
    is_valid: bool
    severity: ValidationSeverity
    message: str
    original_value: Any
    corrected_value: Any | None = None


@dataclass(frozen=True, slots=True)
class ClinicalRecord:
    """Immutable representation of a clinical case record.

    Attributes
    ----------
    record_id : str
        Unique identifier for this record.
    age : int
        Patient age in years.
    sex : SexType
        Patient sex (M/F/U).
    csf_glucose : float | None
        CSF glucose in mg/dL.
    csf_protein : float | None
        CSF protein in mg/dL.
    csf_wbc : int
        CSF white blood cell count.
    csf_rbc : int | None
        CSF red blood cell count.
    microscopy : int
        Microscopy result (0=neg, 1=pos).
    pcr : int | None
        PCR result (0=neg, 1=pos).
    culture : int | None
        Culture result (0=neg, 1=pos).
    symptoms : tuple[str, ...]
        Tuple of symptom strings.
    onset_date : datetime | None
        Date/time of symptom onset.
    duration_hours : int | None
        Duration of symptoms in hours.
    exposure_type : str | None
        Type of water exposure.
    water_source : str | None
        Source of water exposure.
    geographic_region : str | None
        Geographic region of exposure.
    diagnosis : str | None
        Final diagnosis.
    outcome : OutcomeType
        Patient outcome.
    days_to_outcome : int | None
        Days from onset to outcome.
    """

    record_id: str
    age: int
    sex: SexType = "U"
    csf_glucose: float | None = None
    csf_protein: float | None = None
    csf_wbc: int = 0
    csf_rbc: int | None = None
    microscopy: int = 0
    pcr: int | None = None
    culture: int | None = None
    symptoms: tuple[str, ...] = ()
    onset_date: datetime | None = None
    duration_hours: int | None = None
    exposure_type: str | None = None
    water_source: str | None = None
    geographic_region: str | None = None
    diagnosis: str | None = None
    outcome: OutcomeType = "unknown"
    days_to_outcome: int | None = None


@dataclass
class ParserConfig:
    """Configuration for clinical record parsing.

    Attributes
    ----------
    strict_mode : bool
        Raise exceptions on validation errors.
    coerce_types : bool
        Attempt to coerce values to expected types.
    fill_missing : bool
        Fill missing optional fields with defaults.
    validate_ranges : bool
        Validate numeric fields against plausible ranges.
    normalize_symptoms : bool
        Normalize symptom strings to vocabulary.
    date_format : str
        Expected format for date strings.
    missing_values : frozenset[str]
        Strings that represent missing values.
    """

    strict_mode: bool = True
    coerce_types: bool = True
    fill_missing: bool = True
    validate_ranges: bool = True
    normalize_symptoms: bool = True
    date_format: str = "%Y-%m-%d"
    missing_values: frozenset[str] = field(
        default_factory=lambda: frozenset({"", "NA", "N/A", "null", "None", "nan"})
    )


class RecordValidator:
    """Validate clinical records against schema and clinical plausibility.

    Parameters
    ----------
    config : ParserConfig
        Parser configuration.

    Examples
    --------
    >>> validator = RecordValidator(ParserConfig())
    >>> results = validator.validate_record(raw_data)
    >>> if all(r.is_valid for r in results):
    ...     process_record(raw_data)
    """

    __slots__ = ("_config", "_validation_log")

    def __init__(self, config: ParserConfig | None = None) -> None:
        """Initialize record validator with schema configuration.

        Parameters
        ----------
        config : ParserConfig | None
            Validation rules and strictness settings.
        """
        self._config = config or ParserConfig()
        self._validation_log: list[ValidationResult] = []

    @property
    def validation_log(self) -> list[ValidationResult]:
        """Return validation results from most recent validation."""
        return self._validation_log.copy()

    def validate_record(
        self,
        data: dict[str, Any],
    ) -> list[ValidationResult]:
        """Validate a single clinical record.

        Parameters
        ----------
        data : dict[str, Any]
            Raw record data as dictionary.

        Returns
        -------
        list[ValidationResult]
            Validation results for each field.
        """
        self._validation_log = []

        # Check required fields
        for field_name in REQUIRED_FIELDS:
            if field_name not in data:
                self._validation_log.append(
                    ValidationResult(
                        field_name=field_name,
                        is_valid=False,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Required field '{field_name}' is missing",
                        original_value=None,
                    )
                )
            elif self._is_missing(data[field_name]):
                self._validation_log.append(
                    ValidationResult(
                        field_name=field_name,
                        is_valid=False,
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Required field '{field_name}' has missing value",
                        original_value=data[field_name],
                    )
                )

        # Validate numeric fields
        for field_name, (low, high) in VALID_RANGES.items():
            if field_name in data and not self._is_missing(data[field_name]):
                result = self._validate_numeric_range(
                    field_name, data[field_name], low, high
                )
                self._validation_log.append(result)

        # Validate binary fields
        for field_name in ("microscopy", "pcr", "culture"):
            if field_name in data and not self._is_missing(data[field_name]):
                result = self._validate_binary(field_name, data[field_name])
                self._validation_log.append(result)

        # Validate sex field
        if "sex" in data and not self._is_missing(data["sex"]):
            result = self._validate_sex(data["sex"])
            self._validation_log.append(result)

        # Validate symptoms
        if "symptoms" in data and not self._is_missing(data["symptoms"]):
            result = self._validate_symptoms(data["symptoms"])
            self._validation_log.append(result)

        # Validate outcome
        if "outcome" in data and not self._is_missing(data["outcome"]):
            result = self._validate_outcome(data["outcome"])
            self._validation_log.append(result)

        return self._validation_log

    def _is_missing(self, value: Any) -> bool:
        """Check if a value represents a missing value.

        Parameters
        ----------
        value : Any
            Value to check.

        Returns
        -------
        bool
            True if value is considered missing.
        """
        if value is None:
            return True
        if isinstance(value, float) and np.isnan(value):
            return True
        if isinstance(value, str) and value.strip() in self._config.missing_values:
            return True
        return False

    def _validate_numeric_range(
        self,
        field_name: str,
        value: Any,
        low: float,
        high: float,
    ) -> ValidationResult:
        """Validate a numeric field against expected range.

        Parameters
        ----------
        field_name : str
            Name of the field.
        value : Any
            Value to validate.
        low : float
            Minimum acceptable value.
        high : float
            Maximum acceptable value.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        try:
            numeric_value = float(value)
            if low <= numeric_value <= high:
                return ValidationResult(
                    field_name=field_name,
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Value {numeric_value} within range [{low}, {high}]",
                    original_value=value,
                )
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Value {numeric_value} outside range [{low}, {high}]",
                original_value=value,
                corrected_value=max(low, min(high, numeric_value)),
            )
        except (TypeError, ValueError):
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Cannot convert '{value}' to numeric",
                original_value=value,
            )

    def _validate_binary(
        self,
        field_name: str,
        value: Any,
    ) -> ValidationResult:
        """Validate a binary (0/1) field.

        Parameters
        ----------
        field_name : str
            Name of the field.
        value : Any
            Value to validate.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        try:
            int_value = int(float(value))
            if int_value in (0, 1):
                return ValidationResult(
                    field_name=field_name,
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Binary value {int_value} is valid",
                    original_value=value,
                )
            coerced = 1 if int_value else 0
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Value {int_value} not binary, coercing to {coerced}",
                original_value=value,
                corrected_value=coerced,
            )
        except (TypeError, ValueError):
            return ValidationResult(
                field_name=field_name,
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Cannot convert '{value}' to binary",
                original_value=value,
            )

    def _validate_sex(self, value: Any) -> ValidationResult:
        """Validate sex field.

        Parameters
        ----------
        value : Any
            Value to validate.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        str_value = str(value).upper().strip()
        if str_value in ("M", "F", "U", "MALE", "FEMALE", "UNKNOWN"):
            full_names = ("MALE", "FEMALE", "UNKNOWN")
            normalized = str_value[0] if str_value in full_names else str_value
            return ValidationResult(
                field_name="sex",
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message=f"Sex value '{normalized}' is valid",
                original_value=value,
                corrected_value=normalized if normalized != str_value else None,
            )
        return ValidationResult(
            field_name="sex",
            is_valid=False,
            severity=ValidationSeverity.WARNING,
            message=f"Unknown sex value '{value}', defaulting to 'U'",
            original_value=value,
            corrected_value="U",
        )

    def _validate_symptoms(self, value: Any) -> ValidationResult:
        """Validate symptoms field.

        Parameters
        ----------
        value : Any
            Value to validate (string or list).

        Returns
        -------
        ValidationResult
            Validation result.
        """
        if isinstance(value, str):
            symptoms = [s.strip().lower() for s in value.split(";") if s.strip()]
        elif isinstance(value, (list, tuple)):
            symptoms = [str(s).strip().lower() for s in value if s]
        else:
            return ValidationResult(
                field_name="symptoms",
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Cannot parse symptoms from type {type(value).__name__}",
                original_value=value,
            )

        unknown_symptoms = [s for s in symptoms if s not in KNOWN_SYMPTOMS]
        if unknown_symptoms and self._config.normalize_symptoms:
            return ValidationResult(
                field_name="symptoms",
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Unknown symptoms: {unknown_symptoms}",
                original_value=value,
                corrected_value=tuple(symptoms),
            )

        return ValidationResult(
            field_name="symptoms",
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message=f"Validated {len(symptoms)} symptoms",
            original_value=value,
            corrected_value=tuple(symptoms),
        )

    def _validate_outcome(self, value: Any) -> ValidationResult:
        """Validate outcome field.

        Parameters
        ----------
        value : Any
            Value to validate.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        str_value = str(value).lower().strip()
        valid_outcomes = ("survived", "deceased", "unknown")

        if str_value in valid_outcomes:
            return ValidationResult(
                field_name="outcome",
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message=f"Outcome '{str_value}' is valid",
                original_value=value,
            )

        # Try to map common alternatives
        mapping = {
            "alive": "survived",
            "living": "survived",
            "dead": "deceased",
            "death": "deceased",
            "died": "deceased",
            "fatal": "deceased",
            "na": "unknown",
            "n/a": "unknown",
        }

        if str_value in mapping:
            return ValidationResult(
                field_name="outcome",
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Mapped outcome '{value}' to '{mapping[str_value]}'",
                original_value=value,
                corrected_value=mapping[str_value],
            )

        return ValidationResult(
            field_name="outcome",
            is_valid=False,
            severity=ValidationSeverity.WARNING,
            message=f"Unknown outcome '{value}', defaulting to 'unknown'",
            original_value=value,
            corrected_value="unknown",
        )


class CSVRecordParser:
    """Parse clinical records from CSV format files.

    Parameters
    ----------
    config : ParserConfig
        Parser configuration.

    Examples
    --------
    >>> parser = CSVRecordParser(ParserConfig())
    >>> records = parser.parse(Path("clinical_data.csv"))
    >>> print(f"Loaded {len(records)} records")
    """

    __slots__ = ("_config", "_validator", "_parse_errors")

    def __init__(self, config: ParserConfig | None = None) -> None:
        """Initialize CSV parser with validation pipeline.

        Parameters
        ----------
        config : ParserConfig | None
            Parsing configuration and schema rules.
        """
        self._config = config or ParserConfig()
        self._validator = RecordValidator(self._config)
        self._parse_errors: list[tuple[int, str]] = []

    @property
    def parse_errors(self) -> list[tuple[int, str]]:
        """Return errors from most recent parse operation."""
        return self._parse_errors.copy()

    def parse(
        self,
        file_path: Path,
        id_column: str = "record_id",
    ) -> list[ClinicalRecord]:
        """Parse clinical records from CSV file.

        Parameters
        ----------
        file_path : Path
            Path to CSV file.
        id_column : str
            Column name for record identifier.

        Returns
        -------
        list[ClinicalRecord]
            Parsed clinical records.

        Raises
        ------
        FileNotFoundError
            If CSV file does not exist.
        ValueError
            If CSV parsing fails in strict mode.
        """
        self._parse_errors = []
        records: list[ClinicalRecord] = []

        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):  # Header is row 1
                try:
                    record = self._parse_row(row, row_num, id_column)
                    if record is not None:
                        records.append(record)
                except ValueError as e:
                    self._parse_errors.append((row_num, str(e)))
                    if self._config.strict_mode:
                        raise

        logger.info(
            "Parsed %d records from %s with %d errors",
            len(records),
            file_path,
            len(self._parse_errors),
        )
        return records

    def _parse_row(
        self,
        row: dict[str, str],
        row_num: int,
        id_column: str,
    ) -> ClinicalRecord | None:
        """Parse a single CSV row into a ClinicalRecord.

        Parameters
        ----------
        row : dict[str, str]
            CSV row as dictionary.
        row_num : int
            Row number for error reporting.
        id_column : str
            Column name for record identifier.

        Returns
        -------
        ClinicalRecord | None
            Parsed record, or None if validation fails.
        """
        # Generate record ID
        record_id = row.get(id_column, f"row_{row_num}")

        # Validate the row
        validation_results = self._validator.validate_record(row)
        critical_errors = [
            r for r in validation_results
            if not r.is_valid and r.severity == ValidationSeverity.CRITICAL
        ]

        if critical_errors and self._config.strict_mode:
            msg = f"Row {row_num}: {critical_errors[0].message}"
            raise ValueError(msg)

        if critical_errors:
            return None

        # Parse individual fields with coercion
        return ClinicalRecord(
            record_id=str(record_id),
            age=self._parse_int(row.get("age", "0")),
            sex=self._parse_sex(row.get("sex", "U")),
            csf_glucose=self._parse_float_optional(row.get("csf_glucose")),
            csf_protein=self._parse_float_optional(row.get("csf_protein")),
            csf_wbc=self._parse_int(row.get("csf_wbc", "0")),
            csf_rbc=self._parse_int_optional(row.get("csf_rbc")),
            microscopy=self._parse_binary(row.get("microscopy", "0")),
            pcr=self._parse_binary_optional(row.get("pcr")),
            culture=self._parse_binary_optional(row.get("culture")),
            symptoms=self._parse_symptoms(row.get("symptoms", "")),
            onset_date=self._parse_date(row.get("onset_date")),
            duration_hours=self._parse_int_optional(row.get("duration_hours")),
            exposure_type=self._parse_string_optional(row.get("exposure_type")),
            water_source=self._parse_string_optional(row.get("water_source")),
            geographic_region=self._parse_string_optional(
                row.get("geographic_region")
            ),
            diagnosis=self._parse_string_optional(row.get("diagnosis")),
            outcome=self._parse_outcome(row.get("outcome", "unknown")),
            days_to_outcome=self._parse_int_optional(row.get("days_to_outcome")),
        )

    def _parse_int(self, value: str | None) -> int:
        """Parse integer value with default."""
        if value is None or value.strip() in self._config.missing_values:
            return 0
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return 0

    def _parse_int_optional(self, value: str | None) -> int | None:
        """Parse optional integer value."""
        if value is None or value.strip() in self._config.missing_values:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _parse_float_optional(self, value: str | None) -> float | None:
        """Parse optional float value."""
        if value is None or value.strip() in self._config.missing_values:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_binary(self, value: str | None) -> int:
        """Parse binary (0/1) value."""
        if value is None or value.strip() in self._config.missing_values:
            return 0
        try:
            return 1 if int(float(value)) else 0
        except (TypeError, ValueError):
            return 0

    def _parse_binary_optional(self, value: str | None) -> int | None:
        """Parse optional binary value."""
        if value is None or value.strip() in self._config.missing_values:
            return None
        try:
            return 1 if int(float(value)) else 0
        except (TypeError, ValueError):
            return None

    def _parse_sex(self, value: str | None) -> SexType:
        """Parse sex field."""
        if value is None:
            return "U"
        v = value.strip().upper()
        if v in ("M", "MALE"):
            return "M"
        if v in ("F", "FEMALE"):
            return "F"
        return "U"

    def _parse_symptoms(self, value: str | None) -> tuple[str, ...]:
        """Parse symptoms field."""
        if value is None or value.strip() in self._config.missing_values:
            return ()
        symptoms = [s.strip().lower() for s in value.split(";") if s.strip()]
        return tuple(symptoms)

    def _parse_date(self, value: str | None) -> datetime | None:
        """Parse date field."""
        if value is None or value.strip() in self._config.missing_values:
            return None
        try:
            return datetime.strptime(value.strip(), self._config.date_format)
        except ValueError:
            return None

    def _parse_outcome(self, value: str | None) -> OutcomeType:
        """Parse outcome field."""
        if value is None:
            return "unknown"
        v = value.strip().lower()
        if v in ("survived", "alive", "living"):
            return "survived"
        if v in ("deceased", "dead", "died", "death", "fatal"):
            return "deceased"
        return "unknown"

    def _parse_string_optional(self, value: str | None) -> str | None:
        """Parse optional string field."""
        if value is None or value.strip() in self._config.missing_values:
            return None
        return value.strip()


class ClinicalRecordParser:
    """Unified interface for parsing clinical records from multiple formats.

    Automatically detects file format and delegates to appropriate parser.

    Parameters
    ----------
    config : ParserConfig
        Parser configuration.

    Examples
    --------
    >>> parser = ClinicalRecordParser(ParserConfig())
    >>> records = parser.parse(Path("data.csv"))  # Auto-detects CSV
    >>> records = parser.parse(Path("data.json"))  # Auto-detects JSON
    """

    __slots__ = ("_config", "_csv_parser")

    def __init__(self, config: ParserConfig | None = None) -> None:
        """Initialize format-detecting clinical record parser.

        Parameters
        ----------
        config : ParserConfig | None
            Parsing and validation configuration.
        """
        self._config = config or ParserConfig()
        self._csv_parser = CSVRecordParser(self._config)

    def parse(
        self,
        file_path: Path,
        id_column: str = "record_id",
    ) -> list[ClinicalRecord]:
        """Parse clinical records from file.

        Parameters
        ----------
        file_path : Path
            Path to data file (CSV or JSON).
        id_column : str
            Column name for record identifier.

        Returns
        -------
        list[ClinicalRecord]
            Parsed clinical records.

        Raises
        ------
        ValueError
            If file format is not supported.
        """
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            return self._csv_parser.parse(file_path, id_column)
        if suffix == ".json":
            return self._parse_json(file_path, id_column)

        msg = f"Unsupported file format: {suffix}"
        raise ValueError(msg)

    def _parse_json(
        self,
        file_path: Path,
        id_column: str,
    ) -> list[ClinicalRecord]:
        """Parse records from JSON file.

        Parameters
        ----------
        file_path : Path
            Path to JSON file.
        id_column : str
            Key for record identifier.

        Returns
        -------
        list[ClinicalRecord]
            Parsed records.
        """
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and "records" in data:
            rows = data["records"]
        else:
            msg = "JSON must contain array of records or {records: [...]}"
            raise ValueError(msg)

        # Convert to CSV-like format and parse
        records: list[ClinicalRecord] = []
        for i, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            str_row = {k: str(v) if v is not None else "" for k, v in row.items()}
            try:
                record = self._csv_parser._parse_row(str_row, i + 1, id_column)
                if record is not None:
                    records.append(record)
            except ValueError:
                if self._config.strict_mode:
                    raise

        return records

    def to_feature_matrix(
        self,
        records: Sequence[ClinicalRecord],
        symptom_vocabulary: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Convert clinical records to feature matrix.

        Parameters
        ----------
        records : Sequence[ClinicalRecord]
            Clinical records to convert.
        symptom_vocabulary : Sequence[str] | None
            Fixed vocabulary for symptom features.
            If None, uses KNOWN_SYMPTOMS.

        Returns
        -------
        tuple[np.ndarray, list[str]]
            Feature matrix and feature names.
        """
        if symptom_vocabulary is None:
            symptom_vocabulary = sorted(KNOWN_SYMPTOMS)

        feature_names = [
            "age",
            "csf_glucose",
            "csf_protein",
            "csf_wbc",
            "pcr",
            "microscopy",
            "exposure",
        ] + [f"sym_{s}" for s in symptom_vocabulary]

        n_records = len(records)
        n_features = len(feature_names)
        matrix = np.zeros((n_records, n_features), dtype=np.float32)

        for i, record in enumerate(records):
            matrix[i, 0] = record.age
            matrix[i, 1] = record.csf_glucose or 0.0
            matrix[i, 2] = record.csf_protein or 0.0
            matrix[i, 3] = record.csf_wbc
            matrix[i, 4] = record.pcr or 0
            matrix[i, 5] = record.microscopy
            matrix[i, 6] = 1 if record.exposure_type else 0

            for j, symptom in enumerate(symptom_vocabulary):
                if symptom in record.symptoms:
                    matrix[i, 7 + j] = 1.0

        return matrix, feature_names


class CSFReferenceRange(NamedTuple):
    """Reference ranges for cerebrospinal fluid parameters."""

    parameter: str
    unit: str
    normal_min: float
    normal_max: float
    critical_low: float | None
    critical_high: float | None
    age_adjusted: bool


class LaboratoryFlag(Enum):
    """Flags for laboratory value interpretation."""

    NORMAL = auto()
    LOW = auto()
    HIGH = auto()
    CRITICAL_LOW = auto()
    CRITICAL_HIGH = auto()
    INDETERMINATE = auto()


class DiagnosticCriteria(NamedTuple):
    """Diagnostic criteria for PAM evaluation."""

    criterion_id: str
    criterion_name: str
    category: str
    weight: float
    threshold_value: float | None
    comparison_operator: str
    mandatory: bool


class RiskScore(NamedTuple):
    """Calculated risk assessment score."""

    score_id: str
    patient_id: str
    score_value: float
    risk_category: str
    contributing_factors: tuple[str, ...]
    confidence_interval: tuple[float, float]
    calculation_timestamp: datetime


class ClinicalTimeline(NamedTuple):
    """Timeline entry for clinical progression."""

    event_id: str
    patient_id: str
    event_type: str
    event_timestamp: datetime
    description: str
    associated_values: dict[str, Any]


class TreatmentRecord(NamedTuple):
    """Treatment administration record."""

    treatment_id: str
    patient_id: str
    medication_name: str
    dose: float
    dose_unit: str
    route: str
    start_time: datetime
    end_time: datetime | None
    response_noted: bool


CSF_REFERENCE_RANGES: Final[dict[str, CSFReferenceRange]] = {
    "glucose": CSFReferenceRange(
        parameter="glucose",
        unit="mg/dL",
        normal_min=50.0,
        normal_max=80.0,
        critical_low=20.0,
        critical_high=None,
        age_adjusted=False,
    ),
    "protein": CSFReferenceRange(
        parameter="protein",
        unit="mg/dL",
        normal_min=15.0,
        normal_max=45.0,
        critical_low=None,
        critical_high=500.0,
        age_adjusted=True,
    ),
    "wbc": CSFReferenceRange(
        parameter="wbc",
        unit="cells/μL",
        normal_min=0.0,
        normal_max=5.0,
        critical_low=None,
        critical_high=1000.0,
        age_adjusted=True,
    ),
    "rbc": CSFReferenceRange(
        parameter="rbc",
        unit="cells/μL",
        normal_min=0.0,
        normal_max=0.0,
        critical_low=None,
        critical_high=None,
        age_adjusted=False,
    ),
}


@dataclass(frozen=True, slots=True)
class CSFInterpretation:
    """Comprehensive CSF analysis interpretation."""

    glucose_flag: LaboratoryFlag
    protein_flag: LaboratoryFlag
    wbc_flag: LaboratoryFlag
    rbc_flag: LaboratoryFlag
    glucose_ratio: float | None
    pleocytosis_present: bool
    predominant_cell_type: str | None
    interpretation_summary: str
    differential_considerations: tuple[str, ...]


@dataclass(slots=True)
class CSFAnalyzer:
    """Analyzes CSF parameters and provides clinical interpretation."""

    reference_ranges: dict[str, CSFReferenceRange] = field(
        default_factory=lambda: dict(CSF_REFERENCE_RANGES)
    )

    def _evaluate_parameter(
        self, value: float | None, parameter: str
    ) -> LaboratoryFlag:
        """Evaluate single parameter against reference range."""
        if value is None:
            return LaboratoryFlag.INDETERMINATE

        ref = self.reference_ranges.get(parameter)
        if ref is None:
            return LaboratoryFlag.INDETERMINATE

        if ref.critical_low is not None and value < ref.critical_low:
            return LaboratoryFlag.CRITICAL_LOW
        if ref.critical_high is not None and value > ref.critical_high:
            return LaboratoryFlag.CRITICAL_HIGH
        if value < ref.normal_min:
            return LaboratoryFlag.LOW
        if value > ref.normal_max:
            return LaboratoryFlag.HIGH
        return LaboratoryFlag.NORMAL

    def analyze(self, record: ClinicalRecord) -> CSFInterpretation:
        """Perform comprehensive CSF analysis."""
        glucose_flag = self._evaluate_parameter(record.csf_glucose, "glucose")
        protein_flag = self._evaluate_parameter(record.csf_protein, "protein")
        wbc_flag = self._evaluate_parameter(float(record.csf_wbc), "wbc")
        rbc_flag = self._evaluate_parameter(
            float(record.csf_rbc) if record.csf_rbc else None, "rbc"
        )

        glucose_ratio = None
        pleocytosis = record.csf_wbc > 5
        cell_type: str | None = None
        differentials: list[str] = []

        if pleocytosis:
            cell_type = "lymphocyte"
            differentials.append("Infectious meningitis")
            differentials.append("Viral encephalitis")

        if glucose_flag in (LaboratoryFlag.LOW, LaboratoryFlag.CRITICAL_LOW):
            differentials.append("Bacterial meningitis")
            differentials.append("Fungal meningitis")
            differentials.append("Tuberculous meningitis")

        if protein_flag in (LaboratoryFlag.HIGH, LaboratoryFlag.CRITICAL_HIGH):
            differentials.append("Inflammatory process")

        summary = self._generate_summary(
            glucose_flag, protein_flag, wbc_flag, pleocytosis
        )

        return CSFInterpretation(
            glucose_flag=glucose_flag,
            protein_flag=protein_flag,
            wbc_flag=wbc_flag,
            rbc_flag=rbc_flag,
            glucose_ratio=glucose_ratio,
            pleocytosis_present=pleocytosis,
            predominant_cell_type=cell_type,
            interpretation_summary=summary,
            differential_considerations=tuple(differentials),
        )

    def _generate_summary(
        self,
        glucose: LaboratoryFlag,
        protein: LaboratoryFlag,
        wbc: LaboratoryFlag,
        pleocytosis: bool,
    ) -> str:
        """Generate human-readable interpretation summary."""
        parts: list[str] = []

        if pleocytosis:
            parts.append("Pleocytosis present indicating CSF inflammation.")

        if glucose in (LaboratoryFlag.LOW, LaboratoryFlag.CRITICAL_LOW):
            parts.append("Low CSF glucose suggests bacterial or fungal etiology.")

        if protein in (LaboratoryFlag.HIGH, LaboratoryFlag.CRITICAL_HIGH):
            parts.append("Elevated protein indicates blood-brain barrier disruption.")

        if not parts:
            parts.append("CSF parameters within normal limits.")

        return " ".join(parts)


@dataclass(slots=True)
class DiagnosticScoreCalculator:
    """Calculates diagnostic probability scores for PAM."""

    criteria: list[DiagnosticCriteria] = field(default_factory=list)
    _weights_normalized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Initialize with default diagnostic criteria."""
        if not self.criteria:
            self.criteria = self._default_criteria()
        self._normalize_weights()

    def _default_criteria(self) -> list[DiagnosticCriteria]:
        """Define default PAM diagnostic criteria."""
        return [
            DiagnosticCriteria(
                criterion_id="csf_motile",
                criterion_name="Motile amoebae in CSF",
                category="microscopy",
                weight=10.0,
                threshold_value=1.0,
                comparison_operator=">=",
                mandatory=False,
            ),
            DiagnosticCriteria(
                criterion_id="csf_wbc_elevated",
                criterion_name="CSF WBC elevation",
                category="laboratory",
                weight=3.0,
                threshold_value=100.0,
                comparison_operator=">=",
                mandatory=False,
            ),
            DiagnosticCriteria(
                criterion_id="csf_glucose_low",
                criterion_name="CSF glucose depression",
                category="laboratory",
                weight=2.0,
                threshold_value=40.0,
                comparison_operator="<=",
                mandatory=False,
            ),
            DiagnosticCriteria(
                criterion_id="pcr_positive",
                criterion_name="Naegleria PCR positive",
                category="molecular",
                weight=8.0,
                threshold_value=1.0,
                comparison_operator=">=",
                mandatory=False,
            ),
            DiagnosticCriteria(
                criterion_id="water_exposure",
                criterion_name="Freshwater exposure history",
                category="epidemiology",
                weight=4.0,
                threshold_value=None,
                comparison_operator="exists",
                mandatory=False,
            ),
            DiagnosticCriteria(
                criterion_id="rapid_progression",
                criterion_name="Rapid clinical deterioration",
                category="clinical",
                weight=3.0,
                threshold_value=72.0,
                comparison_operator="<=",
                mandatory=False,
            ),
        ]

    def _normalize_weights(self) -> None:
        """Normalize criteria weights to sum to 1.0."""
        if self._weights_normalized:
            return
        total = sum(c.weight for c in self.criteria)
        if total > 0:
            self.criteria = [
                DiagnosticCriteria(
                    criterion_id=c.criterion_id,
                    criterion_name=c.criterion_name,
                    category=c.category,
                    weight=c.weight / total,
                    threshold_value=c.threshold_value,
                    comparison_operator=c.comparison_operator,
                    mandatory=c.mandatory,
                )
                for c in self.criteria
            ]
        self._weights_normalized = True

    def _evaluate_criterion(
        self, criterion: DiagnosticCriteria, record: ClinicalRecord
    ) -> tuple[bool, float]:
        """Evaluate single criterion against clinical record."""
        value: float | None = None
        met = False

        if criterion.criterion_id == "csf_motile":
            value = float(record.microscopy)
            met = record.microscopy == 1
        elif criterion.criterion_id == "csf_wbc_elevated":
            value = float(record.csf_wbc)
            if criterion.threshold_value is not None:
                met = record.csf_wbc >= criterion.threshold_value
        elif criterion.criterion_id == "csf_glucose_low":
            value = record.csf_glucose
            if value is not None and criterion.threshold_value is not None:
                met = value <= criterion.threshold_value
        elif criterion.criterion_id == "pcr_positive":
            if record.pcr is not None:
                value = float(record.pcr)
                met = record.pcr == 1
        elif criterion.criterion_id == "water_exposure":
            met = bool(record.exposure_type)
            value = 1.0 if met else 0.0
        elif criterion.criterion_id == "rapid_progression":
            if record.duration_hours is not None:
                value = float(record.duration_hours)
                if criterion.threshold_value is not None:
                    met = record.duration_hours <= criterion.threshold_value

        contribution = criterion.weight if met else 0.0
        return met, contribution

    def calculate_score(self, record: ClinicalRecord) -> RiskScore:
        """Calculate diagnostic probability score for record."""
        import uuid

        total_score = 0.0
        factors: list[str] = []

        for criterion in self.criteria:
            met, contribution = self._evaluate_criterion(criterion, record)
            total_score += contribution
            if met:
                factors.append(criterion.criterion_name)

        category = self._categorize_risk(total_score)
        ci_low = max(0.0, total_score - 0.1)
        ci_high = min(1.0, total_score + 0.1)

        return RiskScore(
            score_id=str(uuid.uuid4())[:8],
            patient_id=record.record_id,
            score_value=total_score,
            risk_category=category,
            contributing_factors=tuple(factors),
            confidence_interval=(ci_low, ci_high),
            calculation_timestamp=datetime.now(),
        )

    def _categorize_risk(self, score: float) -> str:
        """Categorize risk based on score value."""
        if score >= 0.8:
            return "VERY_HIGH"
        if score >= 0.6:
            return "HIGH"
        if score >= 0.4:
            return "MODERATE"
        if score >= 0.2:
            return "LOW"
        return "VERY_LOW"


@dataclass(slots=True)
class ClinicalTimelineBuilder:
    """Builds clinical progression timelines from records."""

    _events: list[ClinicalTimeline] = field(default_factory=list, init=False)

    def add_symptom_onset(
        self, patient_id: str, onset_time: datetime, symptoms: Sequence[str]
    ) -> None:
        """Record symptom onset event."""
        import uuid

        self._events.append(
            ClinicalTimeline(
                event_id=str(uuid.uuid4())[:8],
                patient_id=patient_id,
                event_type="SYMPTOM_ONSET",
                event_timestamp=onset_time,
                description=f"Initial symptoms: {', '.join(symptoms)}",
                associated_values={"symptoms": list(symptoms)},
            )
        )

    def add_laboratory_result(
        self,
        patient_id: str,
        result_time: datetime,
        test_name: str,
        value: float,
        unit: str,
    ) -> None:
        """Record laboratory result event."""
        import uuid

        self._events.append(
            ClinicalTimeline(
                event_id=str(uuid.uuid4())[:8],
                patient_id=patient_id,
                event_type="LAB_RESULT",
                event_timestamp=result_time,
                description=f"{test_name}: {value} {unit}",
                associated_values={
                    "test": test_name,
                    "value": value,
                    "unit": unit,
                },
            )
        )

    def add_diagnostic_event(
        self,
        patient_id: str,
        event_time: datetime,
        test_type: str,
        result: str,
    ) -> None:
        """Record diagnostic test event."""
        import uuid

        self._events.append(
            ClinicalTimeline(
                event_id=str(uuid.uuid4())[:8],
                patient_id=patient_id,
                event_type="DIAGNOSTIC",
                event_timestamp=event_time,
                description=f"{test_type}: {result}",
                associated_values={"test": test_type, "result": result},
            )
        )

    def add_treatment_event(
        self,
        patient_id: str,
        start_time: datetime,
        medication: str,
        dose: float,
        route: str,
    ) -> None:
        """Record treatment administration event."""
        import uuid

        self._events.append(
            ClinicalTimeline(
                event_id=str(uuid.uuid4())[:8],
                patient_id=patient_id,
                event_type="TREATMENT",
                event_timestamp=start_time,
                description=f"Started {medication} {dose} via {route}",
                associated_values={
                    "medication": medication,
                    "dose": dose,
                    "route": route,
                },
            )
        )

    def get_timeline(self, patient_id: str) -> list[ClinicalTimeline]:
        """Get sorted timeline for patient."""
        patient_events = [e for e in self._events if e.patient_id == patient_id]
        return sorted(patient_events, key=lambda e: e.event_timestamp)

    def get_all_events(self) -> list[ClinicalTimeline]:
        """Get all events sorted by timestamp."""
        return sorted(self._events, key=lambda e: e.event_timestamp)


@dataclass(slots=True)
class OutcomePredictor:
    """Predicts clinical outcomes based on presenting features."""

    _feature_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize feature weights."""
        self._feature_weights = {
            "age_pediatric": -0.2,
            "age_elderly": 0.3,
            "csf_wbc_high": 0.4,
            "csf_glucose_low": 0.5,
            "pcr_positive": 0.6,
            "microscopy_positive": 0.7,
            "rapid_presentation": -0.1,
            "delayed_presentation": 0.4,
        }

    def extract_features(self, record: ClinicalRecord) -> dict[str, float]:
        """Extract predictive features from clinical record."""
        features: dict[str, float] = {}

        features["age_pediatric"] = 1.0 if record.age < 18 else 0.0
        features["age_elderly"] = 1.0 if record.age > 65 else 0.0
        features["csf_wbc_high"] = 1.0 if record.csf_wbc > 500 else 0.0
        features["csf_glucose_low"] = (
            1.0 if record.csf_glucose is not None and record.csf_glucose < 40 else 0.0
        )
        features["pcr_positive"] = 1.0 if record.pcr == 1 else 0.0
        features["microscopy_positive"] = 1.0 if record.microscopy == 1 else 0.0

        if record.duration_hours is not None:
            features["rapid_presentation"] = (
                1.0 if record.duration_hours < 24 else 0.0
            )
            features["delayed_presentation"] = (
                1.0 if record.duration_hours > 72 else 0.0
            )

        return features

    def predict_outcome_probability(
        self, record: ClinicalRecord
    ) -> dict[str, float]:
        """Predict outcome probabilities."""
        features = self.extract_features(record)

        weighted_sum = sum(
            features.get(name, 0.0) * weight
            for name, weight in self._feature_weights.items()
        )

        sigmoid = 1.0 / (1.0 + np.exp(-weighted_sum))

        return {
            "mortality_probability": float(sigmoid),
            "survival_probability": float(1.0 - sigmoid),
            "confidence": 0.7,
        }


@dataclass(slots=True)
class CohortBuilder:
    """Builds patient cohorts for analysis."""

    _records: list[ClinicalRecord] = field(default_factory=list)
    _cohorts: dict[str, list[str]] = field(default_factory=dict, init=False)

    def add_records(self, records: Sequence[ClinicalRecord]) -> None:
        """Add records to cohort builder."""
        self._records.extend(records)

    def build_age_cohorts(
        self, bins: Sequence[int] = (0, 18, 40, 65, 120)
    ) -> dict[str, list[str]]:
        """Build cohorts based on age groups."""
        cohorts: dict[str, list[str]] = {}

        for i in range(len(bins) - 1):
            low, high = bins[i], bins[i + 1]
            cohort_name = f"age_{low}_{high}"
            cohorts[cohort_name] = []

        for record in self._records:
            for i in range(len(bins) - 1):
                low, high = bins[i], bins[i + 1]
                if low <= record.age < high:
                    cohort_name = f"age_{low}_{high}"
                    cohorts[cohort_name].append(record.record_id)
                    break

        self._cohorts.update(cohorts)
        return cohorts

    def build_outcome_cohorts(self) -> dict[str, list[str]]:
        """Build cohorts based on clinical outcome."""
        cohorts: dict[str, list[str]] = {
            "survived": [],
            "deceased": [],
            "unknown": [],
        }

        for record in self._records:
            if record.outcome in cohorts:
                cohorts[record.outcome].append(record.record_id)
            else:
                cohorts["unknown"].append(record.record_id)

        self._cohorts.update(cohorts)
        return cohorts

    def build_exposure_cohorts(self) -> dict[str, list[str]]:
        """Build cohorts based on exposure type."""
        cohorts: dict[str, list[str]] = {}

        for record in self._records:
            exposure = record.exposure_type or "unknown"
            if exposure not in cohorts:
                cohorts[exposure] = []
            cohorts[exposure].append(record.record_id)

        self._cohorts.update(cohorts)
        return cohorts

    def get_cohort(self, cohort_name: str) -> list[str]:
        """Get patient IDs in specified cohort."""
        return self._cohorts.get(cohort_name, [])

    def get_cohort_statistics(self, cohort_name: str) -> dict[str, Any]:
        """Calculate statistics for cohort."""
        record_ids = set(self._cohorts.get(cohort_name, []))
        cohort_records = [r for r in self._records if r.record_id in record_ids]

        if not cohort_records:
            return {"count": 0}

        ages = [r.age for r in cohort_records]
        wbc_values = [r.csf_wbc for r in cohort_records]

        return {
            "count": len(cohort_records),
            "age_mean": float(np.mean(ages)),
            "age_std": float(np.std(ages)),
            "age_min": min(ages),
            "age_max": max(ages),
            "csf_wbc_mean": float(np.mean(wbc_values)),
            "csf_wbc_std": float(np.std(wbc_values)),
            "microscopy_positive_rate": sum(
                1 for r in cohort_records if r.microscopy == 1
            ) / len(cohort_records),
        }


def create_clinical_parser(
    config: ParserConfig | None = None,
) -> ClinicalRecordParser:
    """Factory function to create clinical record parser.

    Parameters
    ----------
    config : ParserConfig | None
        Parser configuration. Uses defaults if None.

    Returns
    -------
    ClinicalRecordParser
        Configured parser instance.
    """
    if config is None:
        config = ParserConfig()
    return ClinicalRecordParser(config=config)


def create_diagnostic_calculator(
    custom_criteria: Sequence[DiagnosticCriteria] | None = None,
) -> DiagnosticScoreCalculator:
    """Factory function to create diagnostic score calculator.

    Parameters
    ----------
    custom_criteria : Sequence[DiagnosticCriteria] | None
        Custom diagnostic criteria. Uses defaults if None.

    Returns
    -------
    DiagnosticScoreCalculator
        Configured calculator instance.
    """
    criteria = list(custom_criteria) if custom_criteria else []
    return DiagnosticScoreCalculator(criteria=criteria)


class SymptomOntology(NamedTuple):
    """Symptom classification in medical ontology."""

    symptom_code: str
    symptom_name: str
    category: str
    system: str
    severity_range: tuple[int, int]
    synonyms: tuple[str, ...]


class DrugInteraction(NamedTuple):
    """Drug-drug interaction information."""

    drug_a: str
    drug_b: str
    interaction_type: str
    severity: str
    mechanism: str
    recommendation: str


class LaboratoryPanel(NamedTuple):
    """Laboratory test panel configuration."""

    panel_id: str
    panel_name: str
    tests: tuple[str, ...]
    specimen_type: str
    stability_hours: int
    turnaround_hours: int


SYMPTOM_ONTOLOGY: Final[dict[str, SymptomOntology]] = {
    "headache": SymptomOntology(
        symptom_code="R51",
        symptom_name="Headache",
        category="neurological",
        system="CNS",
        severity_range=(1, 10),
        synonyms=("cephalalgia", "head pain"),
    ),
    "fever": SymptomOntology(
        symptom_code="R50.9",
        symptom_name="Fever",
        category="constitutional",
        system="systemic",
        severity_range=(1, 5),
        synonyms=("pyrexia", "elevated temperature"),
    ),
    "altered_mental_status": SymptomOntology(
        symptom_code="R41.82",
        symptom_name="Altered Mental Status",
        category="neurological",
        system="CNS",
        severity_range=(1, 10),
        synonyms=("confusion", "encephalopathy", "delirium"),
    ),
    "seizure": SymptomOntology(
        symptom_code="R56.9",
        symptom_name="Seizure",
        category="neurological",
        system="CNS",
        severity_range=(5, 10),
        synonyms=("convulsion", "epileptic episode"),
    ),
    "neck_stiffness": SymptomOntology(
        symptom_code="R29.1",
        symptom_name="Neck Stiffness",
        category="neurological",
        system="musculoskeletal",
        severity_range=(1, 8),
        synonyms=("nuchal rigidity", "meningismus"),
    ),
    "photophobia": SymptomOntology(
        symptom_code="H53.14",
        symptom_name="Photophobia",
        category="sensory",
        system="visual",
        severity_range=(1, 7),
        synonyms=("light sensitivity",),
    ),
    "nausea": SymptomOntology(
        symptom_code="R11.0",
        symptom_name="Nausea",
        category="gastrointestinal",
        system="GI",
        severity_range=(1, 5),
        synonyms=("queasy", "stomach upset"),
    ),
    "vomiting": SymptomOntology(
        symptom_code="R11.1",
        symptom_name="Vomiting",
        category="gastrointestinal",
        system="GI",
        severity_range=(1, 8),
        synonyms=("emesis",),
    ),
}


@dataclass(slots=True)
class SymptomEncoder:
    """Encodes symptoms into structured features."""

    ontology: dict[str, SymptomOntology] = field(
        default_factory=lambda: dict(SYMPTOM_ONTOLOGY)
    )
    _code_to_name: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        """Build reverse lookup."""
        self._code_to_name = {
            v.symptom_code: k for k, v in self.ontology.items()
        }

    def encode(self, symptom: str) -> SymptomOntology | None:
        """Look up symptom in ontology."""
        normalized = symptom.lower().strip().replace(" ", "_")
        if normalized in self.ontology:
            return self.ontology[normalized]

        for name, entry in self.ontology.items():
            if symptom.lower() in (s.lower() for s in entry.synonyms):
                return entry

        return None

    def encode_to_vector(
        self, symptoms: Sequence[str]
    ) -> tuple[np.ndarray, list[str]]:
        """Encode symptoms to binary vector."""
        symptom_names = sorted(self.ontology.keys())
        vector = np.zeros(len(symptom_names), dtype=np.float32)

        for symptom in symptoms:
            entry = self.encode(symptom)
            if entry:
                for i, name in enumerate(symptom_names):
                    if self.ontology[name] == entry:
                        vector[i] = 1.0
                        break

        return vector, symptom_names

    def get_category_counts(
        self, symptoms: Sequence[str]
    ) -> dict[str, int]:
        """Count symptoms by category."""
        counts: dict[str, int] = {}
        for symptom in symptoms:
            entry = self.encode(symptom)
            if entry:
                category = entry.category
                counts[category] = counts.get(category, 0) + 1
        return counts


@dataclass(slots=True)
class TreatmentProtocol:
    """Standard treatment protocol definition."""

    protocol_id: str
    condition: str
    first_line_agents: tuple[str, ...]
    second_line_agents: tuple[str, ...]
    supportive_care: tuple[str, ...]
    monitoring_parameters: tuple[str, ...]
    contraindications: tuple[str, ...]

    def get_recommended_agents(
        self, failed_agents: Sequence[str] | None = None
    ) -> list[str]:
        """Get recommended agents excluding failed ones."""
        failed_set = set(failed_agents or [])
        result: list[str] = []

        for agent in self.first_line_agents:
            if agent not in failed_set:
                result.append(agent)

        if not result:
            for agent in self.second_line_agents:
                if agent not in failed_set:
                    result.append(agent)

        return result


PAM_TREATMENT_PROTOCOL: Final = TreatmentProtocol(
    protocol_id="PAM-2025-001",
    condition="Primary Amoebic Meningoencephalitis",
    first_line_agents=(
        "Amphotericin B (liposomal)",
        "Miltefosine",
    ),
    second_line_agents=(
        "Fluconazole",
        "Rifampin",
        "Azithromycin",
    ),
    supportive_care=(
        "Intracranial pressure management",
        "Seizure prophylaxis",
        "Temperature regulation",
        "Fluid management",
    ),
    monitoring_parameters=(
        "GCS (hourly)",
        "ICP monitoring",
        "CSF analysis (serial)",
        "Renal function",
        "Hepatic function",
    ),
    contraindications=(
        "Renal failure (relative for amphotericin)",
        "Pregnancy (miltefosine)",
    ),
)


@dataclass(slots=True)
class ClinicalDecisionSupport:
    """Clinical decision support engine for PAM diagnosis and treatment."""

    csf_analyzer: CSFAnalyzer = field(default_factory=CSFAnalyzer)
    score_calculator: DiagnosticScoreCalculator = field(
        default_factory=DiagnosticScoreCalculator
    )
    outcome_predictor: OutcomePredictor = field(default_factory=OutcomePredictor)
    treatment_protocol: TreatmentProtocol = field(
        default_factory=lambda: TreatmentProtocol(
            protocol_id="PAM-2025-001",
            condition="Primary Amoebic Meningoencephalitis",
            first_line_agents=("Amphotericin B (liposomal)", "Miltefosine"),
            second_line_agents=("Fluconazole", "Rifampin", "Azithromycin"),
            supportive_care=(
                "Intracranial pressure management",
                "Seizure prophylaxis",
                "Temperature regulation",
            ),
            monitoring_parameters=(
                "GCS (hourly)",
                "ICP monitoring",
                "CSF analysis (serial)",
            ),
            contraindications=(
                "Renal failure (relative for amphotericin)",
                "Pregnancy (miltefosine)",
            ),
        )
    )

    def evaluate_patient(self, record: ClinicalRecord) -> dict[str, Any]:
        """Comprehensive patient evaluation."""
        csf_interpretation = self.csf_analyzer.analyze(record)
        risk_score = self.score_calculator.calculate_score(record)
        outcome_prediction = self.outcome_predictor.predict_outcome_probability(record)
        recommended_treatment = self.treatment_protocol.get_recommended_agents()

        return {
            "patient_id": record.record_id,
            "csf_interpretation": {
                "glucose_status": csf_interpretation.glucose_flag.name,
                "protein_status": csf_interpretation.protein_flag.name,
                "wbc_status": csf_interpretation.wbc_flag.name,
                "pleocytosis": csf_interpretation.pleocytosis_present,
                "summary": csf_interpretation.interpretation_summary,
                "differentials": list(csf_interpretation.differential_considerations),
            },
            "risk_assessment": {
                "score": risk_score.score_value,
                "category": risk_score.risk_category,
                "contributing_factors": list(risk_score.contributing_factors),
                "confidence_interval": risk_score.confidence_interval,
            },
            "outcome_prediction": outcome_prediction,
            "recommended_treatment": recommended_treatment,
            "monitoring": list(self.treatment_protocol.monitoring_parameters),
            "supportive_care": list(self.treatment_protocol.supportive_care),
        }


def create_decision_support() -> ClinicalDecisionSupport:
    """Factory function to create clinical decision support engine.

    Returns
    -------
    ClinicalDecisionSupport
        Configured decision support instance.
    """
    return ClinicalDecisionSupport()


def create_symptom_encoder(
    custom_ontology: dict[str, SymptomOntology] | None = None,
) -> SymptomEncoder:
    """Factory function to create symptom encoder.

    Parameters
    ----------
    custom_ontology : dict[str, SymptomOntology] | None
        Custom symptom ontology. Uses defaults if None.

    Returns
    -------
    SymptomEncoder
        Configured encoder instance.
    """
    ontology = custom_ontology if custom_ontology else dict(SYMPTOM_ONTOLOGY)
    return SymptomEncoder(ontology=ontology)


def create_cohort_builder(
    records: Sequence[ClinicalRecord] | None = None,
) -> CohortBuilder:
    """Factory function to create cohort builder.

    Parameters
    ----------
    records : Sequence[ClinicalRecord] | None
        Initial records to add. Empty if None.

    Returns
    -------
    CohortBuilder
        Configured cohort builder instance.
    """
    builder = CohortBuilder()
    if records:
        builder.add_records(records)
    return builder

