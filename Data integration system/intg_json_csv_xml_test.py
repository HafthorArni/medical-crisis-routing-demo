from __future__ import annotations

import fnmatch
import csv
import io
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Optional, Set

import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 1) Canonical schema
# -----------------------------

@dataclass
class CanonicalHospital:
    name: str
    latitude: float
    longitude: float
    beds_total: int
    beds_available: int
    doctors_total: int
    nurses_total: int
    icu_beds: int
    has_burn_unit: bool
    specialties: List[str]


# Variants include:
# - your hosp1..hosp4 JSON files
# - hosp5..hosp8 XML files (paths include root tag)
# - hosp9..hosp12 CSV files
TARGET_SCHEMA: Dict[str, List[str]] = {
    "name": [
        # JSON
        "name", "hospitalName", "hospital_name", "facility_name", "center_name",
        # XML (root-prefixed)
        "HospitalProfile.PublicLabel",
        "HospitalSite.DisplayTitle",
        "MedicalFacility.NameText",
        "Institution.Identity.CommonName",
        # CSV
        "ProviderDisplay", "OrganizationName", "SiteName", "CenterTitle",
    ],
    "latitude": [
        # JSON
        "location.latitude", "geo.latitude", "coordinates.lat", "gps.lat", "latitude", "lat",
        # XML
        "HospitalProfile.Geo.Northing",
        "HospitalSite.Coordinates.LatValue",
        "MedicalFacility.Location.LatitudeDegrees",
        "Institution.Positioning.LatitudeNumber",
        # CSV
        "CoordLat", "Lat_num", "LatitudeValue", "LatDD",
    ],
    "longitude": [
        # JSON
        "location.longitude", "geo.longitude", "coordinates.lng", "coordinates.long",
        "gps.long", "gps.lng", "longitude", "long", "lng",
        # XML
        "HospitalProfile.Geo.Easting",
        "HospitalSite.Coordinates.LonValue",
        "MedicalFacility.Location.LongitudeDegrees",
        "Institution.Positioning.LongitudeNumber",
        # CSV
        "CoordLon", "Lon_num", "LongitudeValue", "LonDD",
    ],
    "beds_total": [
        # JSON
        "total_beds", "bedCapacity", "bed_count",
        # XML
        "HospitalProfile.CapacityMetrics.WardBedCeiling",
        "HospitalSite.Beds.LicensedBeds",
        "MedicalFacility.BedStats.BedInventoryTotal",
        "Institution.Utilization.BedSupply",
        # CSV
        "TotalWardBeds", "BedStock", "InpatientBedCapacity", "Beds_Allotted",
    ],
    "beds_available": [
        # JSON
        "available_beds", "availableBeds", "open_beds", "free_beds",
        # XML
        "HospitalProfile.CapacityMetrics.BedsVacantNow",
        "HospitalSite.Beds.VacantBeds",
        "MedicalFacility.BedStats.BedsCurrentlyFree",
        "Institution.Utilization.BedAvailability",
        # CSV
        "CurrentlyOpenBeds", "BedVacancy", "BedsFreeNow", "Beds_Available",
    ],
    "icu_beds": [
        # JSON
        "icu_beds", "ICUBeds", "critical_care_beds", "intensive_care_beds",
        # XML
        "HospitalProfile.CapacityMetrics.ICUStations",
        "HospitalSite.Beds.ICUCapacity",
        "MedicalFacility.BedStats.IntensiveCareBedsTotal",
        "Institution.Utilization.IcuCapacity",
        # CSV
        "ICUCots", "ICU_Beds_Count", "IntensiveCareUnits", "ICUBedSpaces",
    ],
    "doctors_total": [
        # JSON
        "staff.doctors", "staffInfo.numDoctors", "personnel.physicians", "employees.doctors",
        # XML
        "HospitalProfile.WorkforceCounts.PhysiciansTotal",
        "HospitalSite.Workforce.DoctorsOnRoster",
        "MedicalFacility.PersonnelCounts.TotalPhysicians",
        "Institution.HumanResources.PhysicianFTE",
        # CSV
        "DoctorsOnPayroll", "PhysiciansHeadcount", "DoctorTotalCount", "MedicsCount",
    ],
    "nurses_total": [
        # JSON
        "staff.nurses", "staffInfo.numNurses", "personnel.nursing_staff", "employees.nurses",
        # XML
        "HospitalProfile.WorkforceCounts.NursesTotal",
        "HospitalSite.Workforce.RegisteredNurses",
        "MedicalFacility.PersonnelCounts.TotalNurses",
        "Institution.HumanResources.NurseFTE",
        # CSV
        "NursesOnPayroll", "NursingHeadcountTotal", "NurseTotalCount", "NursesCount",
    ],
    "has_burn_unit": [
        # JSON
        "has_burn_unit", "isBurnUnit", "burn_unit",
        # XML
        "HospitalProfile.Units.BurnCareSupported",
        "HospitalSite.Services.BurnUnitAvailable",
        "MedicalFacility.SpecialtyFlags.BurnUnitFlag",
        "Institution.ServiceFlags.BurnUnitFlag",
        # CSV
        "BurnUnitYN", "BurnCareAvailable", "BurnUnitIndicator", "BurnUnitPresentFlag",
    ],
    "specialties": [
        # JSON
        "staff.specialists", "staffInfo.specialties", "personnel.medical_specialties", "employees.expertise",
        # XML
        "HospitalProfile.ClinicalFocus.Line",
        "Institution.ServiceLines.Service",
        # CSV
        "ServiceTags", "DeptList", "SpecialtiesCSV", "Services",
    ],
}
TARGET_SCHEMA.update({
    "name": TARGET_SCHEMA["name"] + [
        # Expanded JSON
        "facilityTitle_*",
        # Expanded XML
        "FacilityEnvelope*.DisplayName*",
        # Expanded CSV
        "ProviderLabel_*",
    ],
    "latitude": TARGET_SCHEMA["latitude"] + [
        # Expanded JSON
        "geoPoint_*.northLat_*",
        # Expanded XML
        "FacilityEnvelope*.CoordsBlock*.LatDeg*",
        # Expanded CSV
        "LatitudeDD_*",
    ],
    "longitude": TARGET_SCHEMA["longitude"] + [
        # Expanded JSON
        "geoPoint_*.eastLon_*",
        # Expanded XML
        "FacilityEnvelope*.CoordsBlock*.LonDeg*",
        # Expanded CSV
        "LongitudeDD_*",
    ],
    "beds_total": TARGET_SCHEMA["beds_total"] + [
        # Expanded JSON
        "capacityBlock.licensedBeds_*",
        # Expanded XML
        "FacilityEnvelope*.CapacityStats*.BedCeiling*",
        # Expanded CSV
        "BedsInventory_*",
    ],
    "beds_available": TARGET_SCHEMA["beds_available"] + [
        # Expanded JSON
        "capacityBlock.bedsOpenNow_*",
        # Expanded XML
        "FacilityEnvelope*.CapacityStats*.BedsFreeNow*",
        # Expanded CSV
        "BedsVacant_*",
    ],
    "icu_beds": TARGET_SCHEMA["icu_beds"] + [
        # Expanded JSON
        "capacityBlock.criticalCareSlots_*",
        # Expanded XML
        "FacilityEnvelope*.CapacityStats*.IcuBedsTotal*",
        # Expanded CSV
        "ICUSlots_*",
    ],
    "doctors_total": TARGET_SCHEMA["doctors_total"] + [
        # Expanded JSON
        "workforce_*.physicians_*",
        # Expanded XML
        "FacilityEnvelope*.StaffingTotals*.DoctorCount*",
        # Expanded CSV
        "MDHeadcount_*",
    ],
    "nurses_total": TARGET_SCHEMA["nurses_total"] + [
        # Expanded JSON
        "workforce_*.nursingStaff_*",
        # Expanded XML
        "FacilityEnvelope*.StaffingTotals*.NurseCount*",
        # Expanded CSV
        "RNHeadcount_*",
    ],
    "has_burn_unit": TARGET_SCHEMA["has_burn_unit"] + [
        # Expanded JSON
        "capabilities.burnCareCapable_*",
        # Expanded XML
        "FacilityEnvelope*.BurnUnitFlag*",
        # Expanded CSV
        "BurnCareYN_*",
    ],
    "specialties": TARGET_SCHEMA["specialties"] + [
        # Expanded JSON
        "clinical.serviceLines_*",
        # Expanded XML
        "FacilityEnvelope*.SpecialtyList*",
        # Expanded CSV
        "ServiceTags_*",
    ],
})
REQUIRED_FIELDS = {
    "name", "latitude", "longitude",
    "beds_total", "beds_available",
    "doctors_total", "nurses_total",
    "icu_beds",
}

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 2) Utility functions
# -----------------------------

def preprocess_field(field: str) -> str:
    return field.replace("_", " ").replace(".", " ").lower().strip()


def flatten_any(data: Any, parent_key: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}.{k}" if parent_key else str(k)
            out.update(flatten_any(v, new_key))
    elif isinstance(data, list):
        out[parent_key] = data
        for idx, item in enumerate(data):
            if isinstance(item, dict):
                out.update(flatten_any(item, f"{parent_key}[{idx}]"))
    else:
        out[parent_key] = data
    return out


def coerce_float(v: Any, field: str) -> float:
    if v is None:
        raise ValueError(f"{field} is missing")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        return float(v.strip())
    raise ValueError(f"Cannot coerce {field}={v!r} to float")


def coerce_int(v: Any, field: str) -> int:
    if v is None:
        raise ValueError(f"{field} is missing")
    if isinstance(v, bool):
        raise ValueError(f"Cannot coerce {field}={v!r} to int")
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"{field} is not finite")
        return int(v)
    if isinstance(v, str):
        s = v.strip()
        num = ""
        for ch in s:
            if ch.isdigit() or (ch == "-" and not num):
                num += ch
            elif num:
                break
        if num in {"", "-"}:
            raise ValueError(f"Cannot parse int from {field}={v!r}")
        return int(num)
    raise ValueError(f"Cannot coerce {field}={v!r} to int")


def coerce_bool(v: Any, field: str) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n", ""}:
            return False
        # allow numeric strings
        try:
            return bool(int(float(s)))
        except ValueError:
            pass
    raise ValueError(f"Cannot coerce {field}={v!r} to bool")


def coerce_str_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        for delim in [";", "|", ","]:
            if delim in s:
                return [p.strip() for p in s.split(delim) if p.strip()]
        return [s]
    return [str(v).strip()] if str(v).strip() else []


def validate_constraints(h: CanonicalHospital) -> None:
    if not h.name.strip():
        raise ValueError("name must be a non-empty string")
    if not (-90.0 <= h.latitude <= 90.0):
        raise ValueError("latitude must be in [-90, 90]")
    if not (-180.0 <= h.longitude <= 180.0):
        raise ValueError("longitude must be in [-180, 180]")

    for fname in ["beds_total", "beds_available", "doctors_total", "nurses_total", "icu_beds"]:
        if getattr(h, fname) < 0:
            raise ValueError(f"{fname} must be >= 0")

    if h.beds_available > h.beds_total:
        raise ValueError("beds_available cannot exceed beds_total")
    if h.icu_beds > h.beds_total:
        raise ValueError("icu_beds cannot exceed beds_total")


# -----------------------------
# 3) Parsing: JSON / XML / CSV
# -----------------------------

def parse_json_bytes(b: bytes) -> Any:
    return json.loads(b.decode("utf-8"))


def parse_xml_bytes(b: bytes) -> Any:
    """
    XML -> dict WITH root tag preserved to keep paths stable like:
      HospitalProfile.Geo.Northing
    """
    def elem_to_obj(elem: ET.Element) -> Any:
        children = list(elem)
        if not children:
            return (elem.text or "").strip()
        grouped: Dict[str, List[Any]] = {}
        for c in children:
            grouped.setdefault(c.tag, []).append(elem_to_obj(c))
        out: Dict[str, Any] = {}
        for tag, items in grouped.items():
            out[tag] = items[0] if len(items) == 1 else items
        return out

    root = ET.fromstring(b)
    return {root.tag: elem_to_obj(root)}


def parse_csv_bytes(b: bytes) -> List[Dict[str, Any]]:
    """
    CSV -> list[dict] (each row is a record)
    Robust to leading junk lines (e.g. a first line that is just 'hosp10.csv').
    """
    lines = b.decode("utf-8-sig").splitlines()

    # Find first line that looks like a real CSV header (contains a comma)
    start = 0
    while start < len(lines) and (not lines[start].strip() or "," not in lines[start]):
        start += 1

    if start >= len(lines):
        raise ValueError("CSV has no header row")

    buf = "\n".join(lines[start:])
    reader = csv.DictReader(io.StringIO(buf))
    return [dict(row) for row in reader]


def parse_file(path: str) -> Tuple[str, Any]:
    ext = os.path.splitext(path.lower())[1]
    with open(path, "rb") as f:
        b = f.read()
    if ext == ".json":
        return "json", parse_json_bytes(b)
    if ext == ".xml":
        return "xml", parse_xml_bytes(b)
    if ext == ".csv":
        return "csv", parse_csv_bytes(b)
    raise ValueError(f"Unsupported file extension: {ext}")


# -----------------------------
# 4) REST ingestion (no requests)
# -----------------------------

def fetch_rest(url: str, fmt: str, headers: Optional[Dict[str, str]] = None, timeout_s: int = 30) -> Any:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            b = resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} for {url}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Request failed for {url}: {e}")

    fmt = fmt.lower()
    if fmt == "json":
        return parse_json_bytes(b)
    if fmt == "xml":
        return parse_xml_bytes(b)
    if fmt == "csv":
        return parse_csv_bytes(b)
    raise ValueError(f"Unsupported REST format: {fmt}")


# -----------------------------
# 5) Mapping (synonyms first, optional semantic fallback)
# -----------------------------

#def map_by_synonyms(flat: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for canonical_key, variants in TARGET_SCHEMA.items():
        found = None
        found_key = None
        for v in variants:
            if v in flat:
                found = flat[v]
                found_key = v
                break
        out[canonical_key] = {"value": found, "matched": found_key, "method": "synonym" if found_key else None}
    return out
def map_by_synonyms(flat: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    keys = list(flat.keys())

    for canonical_key, variants in TARGET_SCHEMA.items():
        found = None
        found_key = None

        for v in variants:
            # exact match first
            if v in flat:
                found = flat[v]
                found_key = v
                break

            # wildcard/glob match (supports "*" anywhere in the path)
            if "*" in v:
                matches = [k for k in keys if fnmatch.fnmatch(k, v)]
                if matches:
                    # pick a deterministic winner
                    found_key = sorted(matches)[0]
                    found = flat[found_key]
                    break

        out[canonical_key] = {
            "value": found,
            "matched": found_key,
            "method": "synonym" if found_key else None
        }

    return out

CANONICAL_HINTS = {
    "beds_total": ["total beds", "bed capacity", "licensed beds"],
    "beds_available": ["available beds", "open beds", "free beds", "vacant beds"],
    "icu_beds": ["icu beds", "critical care beds", "intensive care beds"],
    "doctors_total": ["doctors", "physicians", "medical staff doctors"],
    "nurses_total": ["nurses", "rn", "nursing staff"],
    "latitude": ["latitude", "lat"],
    "longitude": ["longitude", "lon", "lng"],
    "name": ["name", "hospital name", "facility name"],
    "has_burn_unit": ["burn unit", "burn care"],
    "specialties": ["specialties", "services", "service lines", "departments"],
}

def semantic_fill_missing(flat: Dict[str, Any], mapped: Dict[str, Dict[str, Any]], threshold: float) -> None:
    input_keys = [k for k in flat.keys() if k]
    if not input_keys:
        return

    # build targets per canonical key (canonical key + hints)
    canonical_targets: Dict[str, List[str]] = {}
    for canonical_key in TARGET_SCHEMA.keys():
        canonical_targets[canonical_key] = [canonical_key] + CANONICAL_HINTS.get(canonical_key, []) + TARGET_SCHEMA[canonical_key]

    input_texts = [preprocess_field(k) for k in input_keys]

    # pre-embed inputs once
    input_emb = MODEL.encode(input_texts)

    for canonical_key, info in mapped.items():
        if info["value"] is not None:
            continue

        targets = canonical_targets[canonical_key]
        target_texts = [preprocess_field(t) for t in targets]
        target_emb = MODEL.encode(target_texts)

        sims = cosine_similarity(input_emb, target_emb)

        # best input key for this canonical field
        best_score = -1.0
        best_in_idx = None
        best_t_idx = None
        for i in range(sims.shape[0]):
            j = int(sims[i].argmax())
            score = float(sims[i, j])
            if score > best_score:
                best_score = score
                best_in_idx = i
                best_t_idx = j

        if best_score >= threshold and best_in_idx is not None:
            in_key = input_keys[best_in_idx]
            mapped[canonical_key] = {
                "value": flat[in_key],
                "matched": in_key,
                "method": f"semantic({targets[best_t_idx]})",
                "score": best_score,
            }

#def semantic_fill_missing(flat: Dict[str, Any], mapped: Dict[str, Dict[str, Any]], threshold: float) -> None:
    input_keys = [k for k in flat.keys() if k]
    if not input_keys:
        return

    variant_to_canonical: Dict[str, str] = {}
    variants: List[str] = []
    for canonical_key, vs in TARGET_SCHEMA.items():
        for v in vs:
            variants.append(v)
            variant_to_canonical[v] = canonical_key

    input_texts = [preprocess_field(k) for k in input_keys]
    variant_texts = [preprocess_field(v) for v in variants]
    embeddings = MODEL.encode(input_texts + variant_texts)

    n_in = len(input_texts)
    sims = cosine_similarity(embeddings[:n_in], embeddings[n_in:])

    for canonical_key, info in mapped.items():
        if info["value"] is not None:
            continue

        best_score = -1.0
        best_in_key = None
        best_variant = None

        for i, in_key in enumerate(input_keys):
            for j, variant in enumerate(variants):
                score = float(sims[i, j])
                if score > best_score:
                    best_score = score
                    best_in_key = in_key
                    best_variant = variant

        if best_score >= threshold and best_in_key and best_variant:
            if variant_to_canonical[best_variant] == canonical_key:
                mapped[canonical_key] = {
                    "value": flat[best_in_key],
                    "matched": best_in_key,
                    "method": f"semantic({best_variant})",
                    "score": best_score,
                }


# -----------------------------
# 6) Normalization
# -----------------------------

def build_canonical(mapped: Dict[str, Dict[str, Any]]) -> CanonicalHospital:
    h = CanonicalHospital(
        name=str(mapped["name"]["value"]).strip() if mapped["name"]["value"] is not None else "",
        latitude=coerce_float(mapped["latitude"]["value"], "latitude"),
        longitude=coerce_float(mapped["longitude"]["value"], "longitude"),
        beds_total=coerce_int(mapped["beds_total"]["value"], "beds_total"),
        beds_available=coerce_int(mapped["beds_available"]["value"], "beds_available"),
        doctors_total=coerce_int(mapped["doctors_total"]["value"], "doctors_total"),
        nurses_total=coerce_int(mapped["nurses_total"]["value"], "nurses_total"),
        icu_beds=coerce_int(mapped["icu_beds"]["value"], "icu_beds"),
        has_burn_unit=coerce_bool(mapped["has_burn_unit"]["value"], "has_burn_unit"),
        specialties=coerce_str_list(mapped["specialties"]["value"]),
    )
    validate_constraints(h)
    return h


def normalize_record(record_obj: Any, semantic_threshold: float) -> Tuple[CanonicalHospital, Dict[str, Any], Dict[str, Any]]:
    if not isinstance(record_obj, dict):
        raise ValueError("Record must be a dict-like object")

    flat = flatten_any(record_obj)
    mapped = map_by_synonyms(flat)
    semantic_fill_missing(flat, mapped, threshold=semantic_threshold)

    missing_required = [k for k in REQUIRED_FIELDS if mapped.get(k, {}).get("value") is None]
    if missing_required:
        raise ValueError(f"Missing required canonical fields: {missing_required}")

    canonical = build_canonical(mapped)

    used_input_keys = {info.get("matched") for info in mapped.values() if info.get("matched")}
    unmatched = {k: v for k, v in flat.items() if k not in used_input_keys}

    return canonical, mapped, unmatched


# -----------------------------
# 7) Aggregation across ALL sources (no fusion)
# -----------------------------

# def aggregate_all(records: List[CanonicalHospital]) -> Dict[str, Any]:
    specialties_union = sorted({s for h in records for s in h.specialties if s})
    if any(h.has_burn_unit for h in records):
        if "burn_unit" not in {s.lower() for s in specialties_union}:
            specialties_union.append("burn_unit")

    return {
        "source_records_total": len(records),
        "beds_total_sum": sum(h.beds_total for h in records),
        "beds_available_sum": sum(h.beds_available for h in records),
        "doctors_total_sum": sum(h.doctors_total for h in records),
        "nurses_total_sum": sum(h.nurses_total for h in records),
        "icu_beds_sum": sum(h.icu_beds for h in records),
        "burn_unit_true_count": sum(1 for h in records if h.has_burn_unit),
        "burn_unit_sources": [h.name for h in records if h.has_burn_unit],
        "specialties_union": specialties_union,
    }

def aggregate_all(records: List[CanonicalHospital]) -> Dict[str, Any]:
    # specialty -> sorted unique hospital names
    specialty_to_hospitals: Dict[str, Set[str]] = {}

    for h in records:
        # treat burn unit as a specialty label too (optional but useful)
        if h.has_burn_unit:
            specialty_to_hospitals.setdefault("burn_unit", set()).add(h.name)

        for s in h.specialties:
            if not s:
                continue
            key = str(s).strip()
            if not key:
                continue
            specialty_to_hospitals.setdefault(key, set()).add(h.name)

    # convert sets to sorted lists for JSON output
    specialty_index = {k: sorted(v) for k, v in sorted(specialty_to_hospitals.items(), key=lambda x: x[0].lower())}

    return {
        "source_records_total": len(records),
        "beds_total_sum": sum(h.beds_total for h in records),
        "beds_available_sum": sum(h.beds_available for h in records),
        "doctors_total_sum": sum(h.doctors_total for h in records),
        "nurses_total_sum": sum(h.nurses_total for h in records),
        "icu_beds_sum": sum(h.icu_beds for h in records),
        "burn_unit_true_count": sum(1 for h in records if h.has_burn_unit),
        "burn_unit_sources": sorted({h.name for h in records if h.has_burn_unit}),
        "hospitals_by_specialty": specialty_index,
    }
# -----------------------------
# 8) Ingestion + pipeline runner
# -----------------------------

def ingest_source(source: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns list of raw record dicts.

    Supported:
      - {"type":"file","path":"x.json|x.xml|x.csv"}
      - {"type":"rest","url":"...","format":"json|xml|csv","headers":{...}}
    """
    stype = source["type"].lower()
    if stype == "file":
        fmt, data = parse_file(source["path"])
    elif stype == "rest":
        data = fetch_rest(source["url"], fmt=source["format"], headers=source.get("headers"))
        fmt = source["format"].lower()
    else:
        raise ValueError(f"Unknown source type: {stype}")

    if fmt == "csv":
        return data  # list[dict]
    if isinstance(data, dict):
        return [data]
    raise ValueError("Unsupported parsed payload shape")


#def run_pipeline(sources: List[Dict[str, Any]], semantic_threshold: float = 0.78) -> Dict[str, Any]:
    normalized: List[CanonicalHospital] = []
    errors: List[Dict[str, Any]] = []
    unmatched_by_source: Dict[str, List[Dict[str, Any]]] = {}

    for src in sources:
        src_label = src.get("path") or src.get("url") or str(src)
        try:
            raw_records = ingest_source(src)
            for idx, rec in enumerate(raw_records):
                try:
                    canon, mapped, unmatched = normalize_record(rec, semantic_threshold=semantic_threshold)
                    normalized.append(canon)
                    if unmatched:
                        unmatched_by_source.setdefault(src_label, []).append({
                            "record_index": idx,
                            "unmatched_fields": unmatched
                        })
                except Exception as e:
                    errors.append({"source": src_label, "record_index": idx, "error": str(e)})
        except Exception as e:
            errors.append({"source": src_label, "record_index": None, "error": str(e)})

    return {
        "aggregate_all_sources": aggregate_all(normalized),
        "normalized_records": [asdict(h) for h in normalized],
        "unmatched_by_source": unmatched_by_source,
        "errors": errors,
    }
def run_pipeline(sources: List[Dict[str, Any]], semantic_threshold: float = 0.78) -> Dict[str, Any]:
    normalized: List[CanonicalHospital] = []
    errors: List[Dict[str, Any]] = []
    unmatched_by_source: Dict[str, List[Dict[str, Any]]] = {}

    # NEW counters
    sources_total = len(sources)
    records_attempted_total = 0   # raw records seen (including failures)
    records_normalized_total = 0  # successful normalizations

    for src in sources:
        src_label = src.get("path") or src.get("url") or str(src)

        try:
            raw_records = ingest_source(src)

            for idx, rec in enumerate(raw_records):
                records_attempted_total += 1

                try:
                    canon, mapped, unmatched = normalize_record(rec, semantic_threshold=semantic_threshold)
                    normalized.append(canon)
                    records_normalized_total += 1

                    if unmatched:
                        unmatched_by_source.setdefault(src_label, []).append({
                            "record_index": idx,
                            "unmatched_fields": unmatched
                        })

                except Exception as e:
                    errors.append({"source": src_label, "record_index": idx, "error": str(e)})

        except Exception as e:
            # ingestion failure: we still count the source, but no record_index
            errors.append({"source": src_label, "record_index": None, "error": str(e)})

    agg = aggregate_all(normalized)

    # NEW: include attempted vs normalized counts
    agg["sources_total"] = sources_total
    agg["records_attempted_total"] = records_attempted_total
    agg["records_normalized_total"] = records_normalized_total

    return {
        "aggregate_all_sources": agg,
        "normalized_records": [asdict(h) for h in normalized],
        "unmatched_by_source": unmatched_by_source,
        "errors": errors,
    }

# -----------------------------
# Example entrypoint (12 sources)
# -----------------------------

if __name__ == "__main__":
    sources = []

    # 20 JSON sources
    for i in range(1, 21):
        sources.append({
            "type": "file",
            "path": f"hosp_json_{i:02d}.json"
        })

    # 20 XML sources
    for i in range(1, 21):
        sources.append({
            "type": "file",
            "path": f"hosp_xml_{i:02d}.xml"
        })

    # 20 CSV sources
    for i in range(1, 21):
        sources.append({
            "type": "file",
            "path": f"hosp_csv_{i:02d}.csv"
        })

    out = run_pipeline(sources)
    print(json.dumps(out["aggregate_all_sources"], indent=2, ensure_ascii=False))
    # sources = [
    #     {"type": "file", "path": "hosp1.json"},
    #     {"type": "file", "path": "hosp2.json"},
    #     {"type": "file", "path": "hosp3.json"},
    #     {"type": "file", "path": "hosp4.json"},
    #     {"type": "file", "path": "hosp5.xml"},
    #     {"type": "file", "path": "hosp6.xml"},
    #     {"type": "file", "path": "hosp7.xml"},
    #     {"type": "file", "path": "hosp8.xml"},
    #     {"type": "file", "path": "hosp9.csv"},
    #     {"type": "file", "path": "hosp10.csv"},
    #     {"type": "file", "path": "hosp11.csv"},
    #     {"type": "file", "path": "hosp12.csv"},
    #     {"type": "file", "path": "hosp13.csv"},
    #     {"type": "file", "path": "hosp14.csv"},
    #     {"type": "file", "path": "hosp15.csv"},
    #     {"type": "file", "path": "hosp16.csv"},
    #     {"type": "file", "path": "hosp17.csv"},
    #     {"type": "file", "path": "hosp18.csv"},
    #     {"type": "file", "path": "hosp19.csv"},
    #     {"type": "file", "path": "hosp2.csv"}
    # ]

    out = run_pipeline(sources)
    print(json.dumps(out["aggregate_all_sources"], indent=2, ensure_ascii=False))
    print("errors:", len(out["errors"]))
    print(json.dumps(out["errors"][:10], indent=2, ensure_ascii=False))  # first 10 only
    # To inspect issues:
    #print(json.dumps(out["unmatched_by_source"], indent=2, ensure_ascii=False))
    #print(json.dumps(out["errors"], indent=2, ensure_ascii=False))