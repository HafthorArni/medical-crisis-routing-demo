"""
Hospital normalizer + aggregator:
- Normalizes JSON and XML inputs into ONE canonical JSON per file
- Validates constraints
- Aggregates totals across all successfully normalized sources:
    * beds_total_sum
    * beds_available_sum
    * doctors_total_sum
    * nurses_total_sum
    * icu_beds_sum
    * burn_unit_true_count
    * burn_unit_sources (names)
    * specialties_union (optional, if present in any source)
- Optional semantic fallback mapping via SentenceTransformer

Dependencies:
  pip install sentence-transformers scikit-learn
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple, Set

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
    specialties: List[str]  # aggregated/normalized list (may be empty)


TARGET_SCHEMA: Dict[str, List[str]] = {
    "name": [
        # JSON variants
        "name", "hospitalName", "hospital_name", "facility_name", "center_name",
        "Medical_center", "facility", "center",
        # XML variants
        "FacilityName", "DisplayTitle", "NameText", "CenterLabel",
    ],
    "latitude": [
        # JSON
        "location.latitude", "geo.latitude", "coordinates.lat", "gps.lat",
        "latitude", "lat",
        # XML
        "GeoPoint.YCoord", "Coordinates.LatValue", "Location.LatitudeDegrees", "MapRef.Lat",
    ],
    "longitude": [
        # JSON
        "location.longitude", "geo.longitude", "coordinates.lng", "coordinates.long",
        "gps.long", "gps.lng", "longitude", "long", "lng",
        # XML
        "GeoPoint.XCoord", "Coordinates.LonValue", "Location.LongitudeDegrees", "MapRef.Long",
    ],
    "beds_total": [
        # JSON
        "total_beds", "bedCapacity", "bed_count", "Amount_of_Beds", "Beds",
        # XML
        "Capacity.InpatientBedSlots", "Beds.LicensedBeds", "BedStats.BedInventoryTotal", "Resources.BedComplement",
    ],
    "beds_available": [
        # JSON
        "available_beds", "availableBeds", "open_beds", "free_beds",
        # XML
        "Capacity.BedsOpenNow", "Beds.VacantBeds", "BedStats.BedsCurrentlyFree", "Resources.BedsAvailableNow",
    ],
    "icu_beds": [
        # JSON
        "icu_beds", "ICUBeds", "intensive_care_beds", "critical_care_beds",
        # XML
        "Capacity.CriticalCareSlots", "Beds.ICUCapacity", "BedStats.IntensiveCareBedsTotal", "Resources.IcuBedCount",
    ],
    "doctors_total": [
        # JSON
        "staff.doctors", "staffInfo.numDoctors", "personnel.physicians", "employees.doctors",
        # XML
        "Staffing.PhysicianHeadcount", "Workforce.DoctorsOnRoster", "PersonnelCounts.TotalPhysicians", "StaffTotals.DoctorTotal",
    ],
    "nurses_total": [
        # JSON
        "staff.nurses", "staffInfo.numNurses", "personnel.nursing_staff", "employees.nurses",
        # XML
        "Staffing.NursingHeadcount", "Workforce.RegisteredNurses", "PersonnelCounts.TotalNurses", "StaffTotals.NurseTotal",
    ],
    "has_burn_unit": [
        # JSON
        "has_burn_unit", "isBurnUnit", "burn_unit",
        # XML
        "SpecialUnits.BurnCare", "Services.BurnUnitAvailable", "SpecialtyFlags.BurnUnitFlag", "SpecialServices.BurnUnit",
    ],
    "specialties": [
        # JSON
        "staff.specialists", "staffInfo.specialties", "personnel.medical_specialties", "employees.expertise",
        # XML (examples)
        "Specialties.Specialty", "Specialties.Item", "Services.Specialty", "Clinical.Specialty",
    ],
}

REQUIRED_FIELDS = {
    "name", "latitude", "longitude",
    "beds_total", "beds_available",
    "doctors_total", "nurses_total",
    "icu_beds",
}

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# 2) Utilities
# -----------------------------

def preprocess_field(field: str) -> str:
    return field.replace("_", " ").replace(".", " ").lower().strip()


def flatten_dict(data: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in data.items():
        new_key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, new_key))
        elif isinstance(v, list):
            # Keep lists as-is at this path; optionally also flatten list elements
            out[new_key] = v
            # If list contains dicts, flatten them with indices (helps some XML/list structures)
            for idx, item in enumerate(v):
                if isinstance(item, dict):
                    out.update(flatten_dict(item, f"{new_key}[{idx}]"))
        else:
            out[new_key] = v
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
    raise ValueError(f"Cannot coerce {field}={v!r} to bool")


def coerce_str_list(v: Any) -> List[str]:
    """
    Accepts:
      - list[str]
      - comma-separated string
      - single string
      - XML-style nested like {"Item": ["a","b"]} (handled earlier via flatten if needed)
    """
    if v is None:
        return []
    if isinstance(v, list):
        out: List[str] = []
        for item in v:
            if item is None:
                continue
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
            else:
                s = str(item).strip()
                if s:
                    out.append(s)
        return out
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        # allow CSV-like lists
        if "," in s:
            return [p.strip() for p in s.split(",") if p.strip()]
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
# 3) Input loaders
# -----------------------------

def load_json(filepath: str) -> Dict[str, Any]:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_xml(filepath: str) -> Dict[str, Any]:
    """
    XML -> dict WITHOUT wrapping root tag.
    Repeated siblings become lists.
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

    root = ET.parse(filepath).getroot()
    return elem_to_obj(root)


def load_input(filepath: str) -> Dict[str, Any]:
    ext = os.path.splitext(filepath.lower())[1]
    if ext == ".json":
        return load_json(filepath)
    if ext == ".xml":
        return load_xml(filepath)
    raise ValueError(f"Unsupported file type: {ext}")


# -----------------------------
# 4) Mapping
# -----------------------------

def map_by_synonyms(flat: Dict[str, Any], schema: Dict[str, List[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for canonical_key, variants in schema.items():
        found = None
        found_key = None
        for v in variants:
            if v in flat:
                found = flat[v]
                found_key = v
                break
        out[canonical_key] = {"value": found, "matched": found_key, "method": "synonym" if found_key else None}
    return out


def semantic_fill_missing(flat: Dict[str, Any], mapped: Dict[str, Any], schema: Dict[str, List[str]], threshold: float) -> None:
    input_keys = list(flat.keys())
    if not input_keys:
        return

    variant_to_canonical: Dict[str, str] = {}
    variants: List[str] = []
    for canonical_key, vs in schema.items():
        for v in vs:
            variants.append(v)
            variant_to_canonical[v] = canonical_key

    input_texts = [preprocess_field(k) for k in input_keys]
    variant_texts = [preprocess_field(v) for v in variants]
    embeddings = MODEL.encode(input_texts + variant_texts)

    n_in = len(input_texts)
    in_emb = embeddings[:n_in]
    var_emb = embeddings[n_in:]
    sims = cosine_similarity(in_emb, var_emb)

    for canonical_key, info in mapped.items():
        if info["value"] is not None:
            continue

        best = (-1.0, None, None)  # score, input_key, variant
        for i, in_key in enumerate(input_keys):
            for j, variant in enumerate(variants):
                score = float(sims[i, j])
                if score > best[0]:
                    best = (score, in_key, variant)

        score, in_key, variant = best
        if score >= threshold and in_key and variant:
            if variant_to_canonical[variant] == canonical_key:
                mapped[canonical_key] = {
                    "value": flat[in_key],
                    "matched": in_key,
                    "method": f"semantic({variant})",
                    "score": score,
                }


# -----------------------------
# 5) Build canonical
# -----------------------------

def build_canonical(mapped: Dict[str, Any]) -> CanonicalHospital:
    name = str(mapped["name"]["value"]).strip() if mapped["name"]["value"] is not None else ""
    latitude = coerce_float(mapped["latitude"]["value"], "latitude")
    longitude = coerce_float(mapped["longitude"]["value"], "longitude")
    beds_total = coerce_int(mapped["beds_total"]["value"], "beds_total")
    beds_available = coerce_int(mapped["beds_available"]["value"], "beds_available")
    doctors_total = coerce_int(mapped["doctors_total"]["value"], "doctors_total")
    nurses_total = coerce_int(mapped["nurses_total"]["value"], "nurses_total")
    icu_beds = coerce_int(mapped["icu_beds"]["value"], "icu_beds")
    has_burn_unit = coerce_bool(mapped["has_burn_unit"]["value"], "has_burn_unit")
    specialties = coerce_str_list(mapped["specialties"]["value"])

    h = CanonicalHospital(
        name=name,
        latitude=latitude,
        longitude=longitude,
        beds_total=beds_total,
        beds_available=beds_available,
        doctors_total=doctors_total,
        nurses_total=nurses_total,
        icu_beds=icu_beds,
        has_burn_unit=has_burn_unit,
        specialties=specialties,
    )
    validate_constraints(h)
    return h


def canonical_to_json_str(h: CanonicalHospital) -> str:
    return json.dumps(asdict(h), indent=2, ensure_ascii=False)


def normalize_file(filepath: str, semantic_threshold: float = 0.78) -> Tuple[CanonicalHospital, Dict[str, Any], Dict[str, Any]]:
    raw = load_input(filepath)
    flat = flatten_dict(raw)

    mapped = map_by_synonyms(flat, TARGET_SCHEMA)
    semantic_fill_missing(flat, mapped, TARGET_SCHEMA, threshold=semantic_threshold)

    missing_required = [k for k in REQUIRED_FIELDS if mapped.get(k, {}).get("value") is None]
    if missing_required:
        raise ValueError(f"{filepath}: missing required canonical fields: {missing_required}")

    canonical = build_canonical(mapped)

    used_input_keys = {info.get("matched") for info in mapped.values() if info.get("matched")}
    unmatched = {k: v for k, v in flat.items() if k not in used_input_keys}

    return canonical, mapped, unmatched


# -----------------------------
# 6) Aggregation across sources
# -----------------------------

def aggregate(canonicals: List[CanonicalHospital]) -> Dict[str, Any]:
    beds_total_sum = sum(h.beds_total for h in canonicals)
    beds_available_sum = sum(h.beds_available for h in canonicals)
    doctors_total_sum = sum(h.doctors_total for h in canonicals)
    nurses_total_sum = sum(h.nurses_total for h in canonicals)
    icu_beds_sum = sum(h.icu_beds for h in canonicals)

    burn_unit_true_count = sum(1 for h in canonicals if h.has_burn_unit)
    burn_unit_sources = [h.name for h in canonicals if h.has_burn_unit]

    specialties_union: Set[str] = set()
    for h in canonicals:
        for s in h.specialties:
            if s:
                specialties_union.add(s.strip())

    return {
        "source_count": len(canonicals),
        "beds_total_sum": beds_total_sum,
        "beds_available_sum": beds_available_sum,
        "doctors_total_sum": doctors_total_sum,
        "nurses_total_sum": nurses_total_sum,
        "icu_beds_sum": icu_beds_sum,
        "burn_unit_true_count": burn_unit_true_count,
        "burn_unit_sources": burn_unit_sources,
        "specialties_union": sorted(specialties_union),
    }


# -----------------------------
# 7) Main
# -----------------------------

def main():
    files = [
        "hosp1.json", "hosp2.json", "hosp3.json", "hosp4.json",
        "hosp1.xml", "hosp2.xml", "hosp3.xml", "hosp4.xml",
    ]

    canonicals: List[CanonicalHospital] = []
    failures: Dict[str, str] = {}

    for fp in files:
        if not os.path.exists(fp):
            continue

        try:
            canonical, mapped, unmatched = normalize_file(fp)
            canonicals.append(canonical)

            print(f"\n=== {fp} ===")
            print(canonical_to_json_str(canonical))

        except Exception as e:
            failures[fp] = str(e)

    print("\n=== AGGREGATE TOTALS ===")
    if canonicals:
        agg = aggregate(canonicals)
        print(json.dumps(agg, indent=2, ensure_ascii=False))
    else:
        print("No sources normalized successfully.")

    if failures:
        print("\n=== FAILURES ===")
        for fp, err in failures.items():
            print(f"- {fp}: {err}")


if __name__ == "__main__":
    main()