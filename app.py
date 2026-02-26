from __future__ import annotations

import os
import math
import json
import random
import threading
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime, timezone

from flask import Flask, Response, abort, render_template, jsonify, request

# External HTTP requests for Overpass API
try:
    import requests  # used for fetching real facility data from OpenStreetMap
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

# --- Healthsites vector-tile stack (optional) ---
# These imports are only required if you enable the Healthsites shapefile → vector tile layers.
HAVE_GEOSPATIAL = True
try:
    import geopandas as gpd
    import pyogrio
    from shapely.geometry import box
    from shapely.ops import unary_union
    from cachetools import TTLCache
    import mapbox_vector_tile
    from pyproj import CRS, Transformer
except Exception:
    HAVE_GEOSPATIAL = False

# --- Demo / simulation data ---
import pandas as pd

# =========================
# CONFIG
# =========================

# Point directly to the folder containing your shapefiles
DATA_DIR = Path(__file__).resolve().parent / "map_data"

LAYER_FILES = {
    "nodes": DATA_DIR / "Europe-node.shp",
    "ways":  DATA_DIR / "Europe-way.shp",
}

# Demo assets shipped with this repo
DEMO_DIR = Path(__file__).resolve().parent / "data"
DEMO_FACILITIES_GEOJSON = Path(os.getenv("DEMO_FACILITIES_GEOJSON", DEMO_DIR / "sample_facilities.geojson"))
DEMO_CASES_XLSX = Path(os.getenv("DEMO_CASES_XLSX", DEMO_DIR / "NATO_MASCAL_500_cases.xlsx"))

# --- OSRM Local Cache ---
DEMO_CACHE_DIR = Path(__file__).resolve().parent / "demo_cache"
OSRM_CACHE_FILE = DEMO_CACHE_DIR / "osrm_routes.json"
_osrm_cache_lock = threading.Lock()

# --- Facilities caching ---
# To avoid repeatedly hitting external APIs during a demo, we store returned
# facility data keyed by a rounded bounding box in a simple JSON cache.
#
# Cache file location: demo_cache/facilities_cache.json
# You can delete this file at any time to force a fresh fetch.

FACILITIES_CACHE_FILE = DEMO_CACHE_DIR / "facilities_cache.json"
_facilities_cache_lock = threading.Lock()

def _load_facilities_cache() -> Dict[str, Any]:
    """Load cached facility FeatureCollections from disk."""
    if not FACILITIES_CACHE_FILE.exists():
        return {}
    try:
        with FACILITIES_CACHE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_facilities_cache(cache: Dict[str, Any]) -> None:
    """Write the facilities cache to disk atomically."""
    FACILITIES_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = FACILITIES_CACHE_FILE.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f)
    tmp_path.replace(FACILITIES_CACHE_FILE)

# Performance: don't serve/request node tiles until this zoom
MIN_ZOOM_NODES = 11

# Hard caps per tile (keeps server responsive on huge datasets)
MAX_FEATURES_WAYS = 8000
MAX_FEATURES_NODES = 1500

# Vector tile settings
EXTENT = 4096
WEBMERC_MAX = 20037508.342789244  # meters


# -------------------------
# Healthsites attribute normalization (shapefile columns -> canonical keys)
# -------------------------
SHAPE_TO_CANON = {
    "osm_id": "upstream_id",
    "name": "name",
    "source": "source",
    "amenity": "amenity",
    "healthcare": "healthcare",
    "speciality": "healthcare:speciality",
    "operator": "operator",
    "operator_t": "operator:type",
    "contact_nu": "contact:phone",
    "opening_ho": "opening_hours",
    "beds": "bed_count",
    "staff_doct": "doctors_num",
    "staff_nurs": "nurses_num",
    "addr_house": "addr:housenumber",
    "addr_stree": "addr:street",
    "addr_postc": "addr:postcode",
    "addr_city": "addr:city",
    "url": "website",
    "water_sour": "water_source",
    "electricit": "electricity",
    "dispensing": "dispensing",
    "wheelchair": "wheelchair",
    "emergency": "emergency",
    "insurance": "insurance",
    "operationa": "operational_status",
    "health_ame": "health_amenity",
}

NODE_FIELDS_LOW = ["osm_id", "name", "source", "amenity", "healthcare", "contact_nu", "opening_ho", "beds"]
NODE_FIELDS_HIGH = NODE_FIELDS_LOW + [
    "operator", "operator_t", "speciality", "staff_doct", "staff_nurs",
    "addr_house", "addr_stree", "addr_postc", "addr_city",
    "url", "wheelchair", "emergency", "dispensing", "insurance",
    "water_sour", "electricit", "operationa",
]

WAYS_FIELDS_LOW = ["osm_id", "name", "source", "amenity", "healthcare"]
WAYS_FIELDS_HIGH = WAYS_FIELDS_LOW + [
    "operator", "operator_t", "speciality", "opening_ho", "beds", "staff_doct", "staff_nurs",
    "addr_house", "addr_stree", "addr_postc", "addr_city",
    "url", "wheelchair", "emergency", "dispensing", "insurance",
    "water_sour", "electricit", "operationa",
]


def _iso_utc_from_epoch(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")

def _fix_mojibake(text: Any) -> Any:
    """
    Recovers valid UTF-8 strings that were mistakenly decoded as Latin-1.
    """
    if isinstance(text, str):
        try:
            return text.encode('latin1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return text

# =========================
# APP + CACHES
# =========================

app = Flask(__name__)

# Vector-tile cache (MVT)
tile_cache = TTLCache(maxsize=15000, ttl=24 * 3600) if HAVE_GEOSPATIAL else {}

# Layer metadata caches
layer_crs: Dict[str, CRS] = {}
to_data_bounds: Dict[str, Transformer] = {}     # 3857 -> data CRS
to_3857: Dict[str, Transformer] = {}            # data CRS -> 3857
layer_fields: Dict[str, List[str]] = {}
layer_encoding: Dict[str, str] = {}
layer_file_modified_iso: Dict[str, str] = {}

layer_init_lock = threading.Lock()

# Demo caches (in-memory, resettable)
_demo_lock = threading.RLock()
_demo_facilities_initial: Optional[Dict[str, Any]] = None
_demo_facilities_state: Optional[Dict[str, Any]] = None
_demo_cases: Optional[Dict[str, Any]] = None
_current_buffer: float = 1.5


# =========================
# DEMO DATA LOADING
# =========================

def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _generate_healthsites_facilities(cases_fc: Dict[str, Any], buffer_val: float = 1.5) -> Optional[Dict[str, Any]]:
    """
    Attempt to load real-world facility locations and metadata.

    This function now tries multiple data sources in order of preference:

    1. **Overpass API** – If available and requests can be made, the function will query
       OpenStreetMap via the Overpass API for healthcare-related facilities (hospitals,
       clinics, doctors) within a bounding box around the casualty locations.  This
       data source can provide richer tag information than the shipped shapefile and
       does not require a separately downloaded dataset.  The Overpass API uses a
       bounding box defined by minimum latitude, minimum longitude, maximum latitude and
       maximum longitude, as described in the OpenStreetMap Overpass documentation【878746415656765†L468-L470】.

       If the query succeeds and returns one or more facilities, the results are
       normalized and mapped to the expected output format.  Randomized capacity
       values are generated (seeded by each facility’s OSM identifier) to preserve
       deterministic behaviour across runs.  A simple heuristic is applied to map
       facility types (hospital/clinic/doctor) to an approximate role of care.

    2. **Healthsites Shapefile** – If Overpass is unavailable or returns no data,
       the function falls back to the original behaviour: reading from the Healthsites
       shapefile export (Europe-node.shp).  If the shapefile is missing entirely,
       `None` is returned and a warning is logged.

    Parameters:
        cases_fc (dict): GeoJSON FeatureCollection of casualty locations.
        buffer_val (float): Padding (in degrees) around the casualty bounding box used to
            expand the search area.

    Returns:
        dict or None: A GeoJSON FeatureCollection containing facility features with
            synthetic capacity data.
    """
    feats = cases_fc.get("features", []) or []
    if not feats:
        print(" -> FAILED: No MASCAL cases found to build a bounding box.")
        return None

    # 1. Compute bounding box of casualties and apply a buffer
    lons = [float(f["geometry"]["coordinates"][0]) for f in feats]
    lats = [float(f["geometry"]["coordinates"][1]) for f in feats]
    BUFFER = float(buffer_val)
    minlon, maxlon = min(lons) - BUFFER, max(lons) + BUFFER
    minlat, maxlat = min(lats) - BUFFER, max(lats) + BUFFER

    # Round the bounding box to ~2 decimal places for caching to reduce key
    # cardinality without sacrificing too much geographic specificity.  Two
    # decimals (~1.1 km at equator) strike a good balance for caching under the
    # hackathon's scale.
    bbox_key = f"{minlat:.2f}_{minlon:.2f}_{maxlat:.2f}_{maxlon:.2f}"

    # Attempt to load a cached FeatureCollection for this bounding box.  This
    # prevents redundant API calls on subsequent runs.  We always guard
    # modifications with a lock to handle concurrent requests.
    with _facilities_cache_lock:
        cache = _load_facilities_cache()
        cached_fc = cache.get(bbox_key)
    if cached_fc:
        # Deep-copy to avoid mutation of cached object downstream
        return json.loads(json.dumps(cached_fc))

    fc_result: Optional[Dict[str, Any]] = None
    # 2. Attempt to query Healthsites.io API if API key is provided
    try:
        api_data = _generate_healthsites_api_facilities(
            minlat=minlat, minlon=minlon, maxlat=maxlat, maxlon=maxlon
        )
        if api_data and api_data.get("features"):
            fc_result = api_data
    except Exception as e:
        print(f"WARNING: Healthsites API query failed: {e}")

    # 3. Attempt to query Overpass API if Healthsites returned nothing
    if fc_result is None and HAVE_REQUESTS:
        try:
            overpass_data = _generate_overpass_facilities(
                minlat=minlat, minlon=minlon, maxlat=maxlat, maxlon=maxlon
            )
            if overpass_data and overpass_data.get("features"):
                fc_result = overpass_data
        except Exception as e:
            print(f"WARNING: Overpass API query failed: {e}")

    # 4. Fallback to Healthsites shapefile if geospatial libs are available and still no data
    if fc_result is None:
        if not _healthsites_available():
            print(" -> FAILED: _healthsites_available() returned False.")
            print(" -> Check if 'Europe-node.shp' is inside your 'DATA_DIR' folder.")
            fc_result = None
        else:
            # Perform shapefile filtering and generate synthetic facilities
            print(f" -> Querying shapefile bounding box: {minlon:.2f}, {minlat:.2f}, {maxlon:.2f}, {maxlat:.2f}")
            try:
                gdf = gpd.read_file(
                    LAYER_FILES["nodes"],
                    bbox=(minlon, minlat, maxlon, maxlat),
                    engine="pyogrio",
                    encoding="utf-8"
                )
            except UnicodeDecodeError:
                print(" -> UTF-8 decode failed, falling back to Latin-1 encoding...")
                try:
                    gdf = gpd.read_file(
                        LAYER_FILES["nodes"],
                        bbox=(minlon, minlat, maxlon, maxlat),
                        engine="pyogrio",
                        encoding="latin1"
                    )
                except Exception as e:
                    print(f" -> FAILED: Error reading shapefile with Latin-1: {e}")
                    fc_result = None
                    gdf = None  # type: ignore
                except Exception:
                    gdf = None  # type: ignore
            except Exception as e:
                print(f" -> FAILED: Error reading shapefile: {e}")
                gdf = None  # type: ignore
            if gdf is not None:
                if gdf.empty:
                    print(" -> FAILED: Bounding box query returned 0 facilities in this area.")
                    fc_result = None
                else:
                    # Filter by relevant amenities/healthcare types
                    valid_types = ["hospital", "clinic", "doctors"]
                    mask = (gdf["amenity"].isin(valid_types)) | (gdf["healthcare"].isin(valid_types))
                    gdf = gdf[mask]
                    features = []
                    possible_specialties = [
                        "General Surgery", "Orthopedics", "Trauma", "Neurosurgery",
                        "Burn", "Pediatrics", "OBGYN"
                    ]
                    for idx, row in gdf.iterrows():
                        geom = row.geometry
                        if geom is None or geom.is_empty:
                            continue
                        name = _fix_mojibake(row.get("name"))
                        if not name or str(name).lower() == "nan":
                            name = f"Unnamed Facility ({row.get('osm_id', idx)})"
                        osm_id = row.get("osm_id", idx)
                        rng = random.Random(str(osm_id))

                        # --- Real-world type (from OSM/Healthsites tags) ---
                        amenity = (row.get("amenity") or "")
                        healthcare = (row.get("healthcare") or "")
                        fac_type = (str(amenity) or str(healthcare)).lower()

                        # Heuristic mapping to Role of Care (NATO-style) based on facility type.
                        # OSM doesn't encode NATO roles directly, so this is an inference.
                        if "hospital" in fac_type:
                            role_int = rng.choices([3, 4], weights=[0.8, 0.2])[0]
                        elif "clinic" in fac_type:
                            role_int = 2
                        elif "doctors" in fac_type:
                            role_int = 1
                        else:
                            role_int = rng.choices([1, 2, 3, 4], weights=[0.5, 0.3, 0.15, 0.05])[0]

                        # Prefer any real bed count if present (often sparse in OSM)
                        def _safe_int(v):
                            try:
                                if v is None:
                                    return None
                                s = str(v).strip()
                                if not s or s.lower() == "nan":
                                    return None
                                return int(float(s))
                            except Exception:
                                return None

                        beds_real = _safe_int(row.get("beds"))
                        beds_total = beds_real if (beds_real is not None and beds_real > 0) else (rng.randint(15, 60) * role_int)
                        beds_av = int(beds_total * rng.uniform(0.1, 0.9))
                        icu_total = rng.randint(2, 10) * role_int if role_int >= 2 else 0
                        icu_av = int(icu_total * rng.uniform(0.1, 0.9))
                        vent_total = icu_total * 2 if icu_total > 0 else rng.randint(0, 2)
                        vent_av = int(vent_total * rng.uniform(0.1, 0.9))

                        # --- Specialties: prefer real OSM/Healthsites specialty tags, else demo-fill ---
                        specialties: List[str] = []
                        spec_raw = row.get("speciality")
                        if spec_raw is not None and str(spec_raw).strip() and str(spec_raw).lower() != "nan":
                            for part in str(spec_raw).replace(",", ";").split(";"):
                                p = part.strip()
                                if p:
                                    specialties.append(p)

                        # If we still don't have anything, generate demo specialties
                        if not specialties:
                            num_specs = rng.randint(1, 2) + (role_int - 1)
                            specialties = []
                            if role_int >= 2:
                                specialties += ["Blood Bank", "CT", "ICU"]
                            specialties += rng.sample(possible_specialties, min(num_specs, len(possible_specialties)))

                        # Add emergency as a specialty if tagged
                        emergency_tag = (row.get("emergency") or "")
                        if str(emergency_tag).strip().lower() in ("yes", "true", "1"):
                            specialties.append("Emergency")

                        specialties = sorted(set([s for s in specialties if s]))

                        # --- Additional real-world metadata from the Healthsites export (if present) ---
                        operator = _fix_mojibake(row.get("operator"))
                        operator_type = _fix_mojibake(row.get("operator_t"))
                        contact_phone = _fix_mojibake(row.get("contact_nu"))
                        opening_hours = _fix_mojibake(row.get("opening_ho"))
                        website = _fix_mojibake(row.get("url"))

                        addr = {
                            "housenumber": _fix_mojibake(row.get("addr_house")),
                            "street": _fix_mojibake(row.get("addr_stree")),
                            "postcode": _fix_mojibake(row.get("addr_postc")),
                            "city": _fix_mojibake(row.get("addr_city")),
                        }
                        # compact address string for display
                        addr_parts = [addr.get("street"), addr.get("housenumber"), addr.get("postcode"), addr.get("city")]
                        address = ", ".join([p for p in addr_parts if p and str(p).lower() != "nan"])

                        staff_doctors = _safe_int(row.get("staff_doct"))
                        staff_nurses = _safe_int(row.get("staff_nurs"))
                        props = {
                            "facility_id": str(osm_id),
                            "name": name,
                            "country": "EU",
                            "in_crisis_area": True,
                            "role_of_care": f"Role{role_int}",
                            "specialties": specialties,
                            "capacity": {
                                "beds_total": beds_total,
                                "beds_available": beds_av,
                                "icu_total": icu_total,
                                "icu_available": icu_av,
                                "vent_total": vent_total,
                                "vent_available": vent_av,
                            },
                            # extra metadata (real when available)
                            "amenity": amenity,
                            "healthcare": healthcare,
                            "operator": operator,
                            "operator_type": operator_type,
                            "contact_phone": contact_phone,
                            "opening_hours": opening_hours,
                            "website": website,
                            "address": address,
                            "wheelchair": _fix_mojibake(row.get("wheelchair")),
                            "dispensing": _fix_mojibake(row.get("dispensing")),
                            "insurance": _fix_mojibake(row.get("insurance")),
                            "water_source": _fix_mojibake(row.get("water_sour")),
                            "electricity": _fix_mojibake(row.get("electricit")),
                            "operational_status": _fix_mojibake(row.get("operationa")),
                            "staff": {"doctors": staff_doctors, "nurses": staff_nurses},
                            "source": "healthmap.io (Auto-generated)",
                            "last_updated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        }
                        features.append(
                            {
                                "type": "Feature",
                                "geometry": {"type": "Point", "coordinates": [geom.x, geom.y]},
                                "properties": props,
                            }
                        )
                    fc_result = {"type": "FeatureCollection", "features": features}
    # Persist to cache if we produced a result
    if fc_result is not None:
        with _facilities_cache_lock:
            cache = _load_facilities_cache()
            # Store a deep copy of the result to avoid accidental mutation
            cache[bbox_key] = json.loads(json.dumps(fc_result))
            try:
                _write_facilities_cache(cache)
            except Exception as e:
                print(f"WARNING: Failed to write facilities cache: {e}")
    return fc_result


def _generate_overpass_facilities(*, minlat: float, minlon: float, maxlat: float, maxlon: float) -> Optional[Dict[str, Any]]:
    """
    Query the Overpass API for healthcare facilities within a bounding box and
    construct a GeoJSON FeatureCollection with synthetic capacity data.

    The Overpass query searches for nodes, ways and relations tagged as
    `amenity=hospital`, `amenity=clinic`, `amenity=doctors`, or with
    equivalent `healthcare` tags.  Ways and relations are represented by
    their centroid (via the `center` keyword) to obtain a point geometry.

    Parameters:
        minlat, minlon, maxlat, maxlon (float): south, west, north and east edges of
            the bounding box (lat/lon order).  The order matches the Overpass
            bounding box format where the first two values are minimum latitude
            and minimum longitude and the last two are maximum latitude and
            maximum longitude【878746415656765†L468-L470】.

    Returns:
        dict or None: GeoJSON FeatureCollection of facilities, or None on failure.
    """
    if not HAVE_REQUESTS:
        return None

    # Compose Overpass QL query
    query = f"""
    [out:json][timeout:120];
    (
      node["amenity"~"hospital|clinic|doctors"]({minlat},{minlon},{maxlat},{maxlon});
      node["healthcare"~"hospital|clinic|doctors"]({minlat},{minlon},{maxlat},{maxlon});
      way["amenity"~"hospital|clinic|doctors"]({minlat},{minlon},{maxlat},{maxlon});
      way["healthcare"~"hospital|clinic|doctors"]({minlat},{minlon},{maxlat},{maxlon});
      relation["amenity"~"hospital|clinic|doctors"]({minlat},{minlon},{maxlat},{maxlon});
      relation["healthcare"~"hospital|clinic|doctors"]({minlat},{minlon},{maxlat},{maxlon});
    );
    out center tags;
    """
    url = "https://overpass-api.de/api/interpreter"
    resp = requests.post(url, data=query.encode("utf-8"), headers={"Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"}, timeout=180)
    if resp.status_code != 200:
        raise RuntimeError(f"Overpass API returned HTTP {resp.status_code}")
    data = resp.json()
    elements = data.get("elements", [])
    if not elements:
        return None

    features: List[Dict[str, Any]] = []
    # Predefined specialties map to convert OSM specialties into human-friendly names
    osm_specialty_mapping = {
        "pediatrics": "Pediatrics",
        "paediatrics": "Pediatrics",
        "gynaecology": "OBGYN",
        "gynecology": "OBGYN",
        "surgery": "General Surgery",
        "trauma": "Trauma",
        "orthopedics": "Orthopedics",
        "orthopaedics": "Orthopedics",
        "neurosurgery": "Neurosurgery",
        "burn": "Burn",
        "icu": "ICU",
        "emergency": "Emergency",
        "cardiology": "Cardiology",
        "urology": "Urology",
    }
    for el in elements:
        # Determine coordinates
        if el.get("type") == "node":
            lon, lat = el.get("lon"), el.get("lat")
        else:
            center = el.get("center") or {}
            lon, lat = center.get("lon"), center.get("lat")
        if lon is None or lat is None:
            continue
        tags = el.get("tags") or {}
        # Determine name
        name = _fix_mojibake(tags.get("name")) if tags.get("name") else None
        if not name:
            name = f"Unnamed Facility ({el.get('type')}_{el.get('id')})"
        osm_id = f"{el.get('type')}_{el.get('id')}"
        # Determine type and infer role
        fac_type = tags.get("amenity") or tags.get("healthcare") or ""
        fac_type_lower = str(fac_type).lower()
        if "hospital" in fac_type_lower:
            role_int = 3
            # Upgrade to Role4 for large/teaching hospitals
            if any(
                kw in str(tags.get("name", "")).lower()
                for kw in ["university", "clinic centre", "central"]
            ):
                role_int = 4
        elif "clinic" in fac_type_lower:
            role_int = 2
        elif "doctors" in fac_type_lower:
            role_int = 1
        else:
            role_int = 1
        # Determine specialties from OSM tags
        specialties: List[str] = []
        # The tag healthcare:speciality may be semicolon separated
        spec_tags = tags.get("healthcare:speciality") or tags.get("speciality") or tags.get("healthcare_speciality")
        if spec_tags:
            for spec in str(spec_tags).split(";"):
                spec_clean = spec.strip().lower()
                mapped = osm_specialty_mapping.get(spec_clean)
                if mapped:
                    specialties.append(mapped)
                else:
                    specialties.append(spec_clean.capitalize())
        # Additional speciality inference based on keywords
        for key, val in tags.items():
            # if a tag key or value contains known specialties
            for k, v in osm_specialty_mapping.items():
                if k in str(val).lower() and v not in specialties:
                    specialties.append(v)
        # Always include general services based on role
        if role_int >= 2:
            for base in ["Blood Bank", "CT", "ICU"]:
                if base not in specialties:
                    specialties.append(base)
        specialties = sorted(set(specialties))
        # Seed RNG for deterministic capacities
        rng = random.Random(str(osm_id))
        beds_total = rng.randint(15, 60) * role_int
        beds_av = int(beds_total * rng.uniform(0.1, 0.9))
        icu_total = rng.randint(2, 10) * role_int if role_int >= 2 else 0
        icu_av = int(icu_total * rng.uniform(0.1, 0.9))
        vent_total = icu_total * 2 if icu_total > 0 else rng.randint(0, 2)
        vent_av = int(vent_total * rng.uniform(0.1, 0.9))
        props = {
            "facility_id": osm_id,
            "name": name,
            "country": tags.get("addr:country") or "Unknown",
            "in_crisis_area": True,
            "role_of_care": f"Role{role_int}",
            "specialties": specialties,
            "capacity": {
                "beds_total": beds_total,
                "beds_available": beds_av,
                "icu_total": icu_total,
                "icu_available": icu_av,
                "vent_total": vent_total,
                "vent_available": vent_av,
            },
            "source": "OpenStreetMap via Overpass API (Auto-generated)",
            "last_updated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": props,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _generate_healthsites_api_facilities(*, minlat: float, minlon: float, maxlat: float, maxlon: float) -> Optional[Dict[str, Any]]:
    """
    Query the Healthsites.io API for facilities within a bounding box.

    This function requires an API key to be provided via the environment variable
    `HEALTHSITES_API_KEY`.  If no key is set, the function returns `None` without
    performing any network requests.  The API is paginated; results are fetched
    page-by-page until no further pages are available.

    The API endpoint accepts an `extent` parameter (minLng,minLat,maxLng,maxLat) to
    restrict results【546452282617198†L54-L60】.  We request GeoJSON output and flat
    properties so that tags are returned in a simple dictionary rather than
    nested structures.  If the API call fails or returns no features, this
    function returns `None`.

    Parameters:
        minlat, minlon, maxlat, maxlon (float): bounding box coordinates.

    Returns:
        dict or None: GeoJSON FeatureCollection or None.
    """
    if not HAVE_REQUESTS:
        return None
    api_key = os.getenv("HEALTHSITES_API_KEY")
    if not api_key:
        # No API key provided; skip attempt
        return None
    base_url = "https://healthsites.io/api/v3/facilities/"
    page = 1
    features_out: List[Dict[str, Any]] = []
    while True:
        params = {
            "api-key": api_key,
            "page": page,
            "extent": f"{minlon},{minlat},{maxlon},{maxlat}",
            "output": "geojson",
            "flat-properties": "true",
            "tag-format": "osm",
        }
        resp = requests.get(base_url, params=params, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"Healthsites API returned HTTP {resp.status_code}")
        try:
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Failed to decode Healthsites response: {e}")
        # Determine location of feature collection
        # Some healthsites deployments wrap features under 'results'
        fc = None
        if isinstance(data, dict):
            if data.get("type") == "FeatureCollection" and "features" in data:
                fc = data
            elif "results" in data and isinstance(data["results"], dict):
                res = data["results"]
                if res.get("type") == "FeatureCollection" and "features" in res:
                    fc = res
        if fc is None:
            break
        feats = fc.get("features", []) or []
        if not feats:
            break
        for f in feats:
            geom = f.get("geometry") or {}
            coords = None
            if geom.get("type") == "Point":
                coords = geom.get("coordinates")
            elif geom.get("type") in ("Polygon", "MultiPolygon", "LineString"):
                # Use centroid-like approximation: compute average of first ring
                coords_list = []
                def _flatten(nested):
                    for item in nested:
                        if isinstance(item[0], (list, tuple)):
                            for sub in _flatten(item):
                                yield sub
                        else:
                            yield item
                try:
                    coords_gen = list(_flatten(geom.get("coordinates")))
                    lons = [c[0] for c in coords_gen if isinstance(c, (list, tuple))]
                    lats = [c[1] for c in coords_gen if isinstance(c, (list, tuple))]
                    if lons and lats:
                        coords = [sum(lons) / len(lons), sum(lats) / len(lats)]
                except Exception:
                    coords = None
            if not coords or len(coords) < 2:
                continue
            lon, lat = coords[0], coords[1]
            props_raw = f.get("properties") or {}
            # Flatten tags if nested
            tags = props_raw
            # Many healthsites outputs include a 'tags' object; merge it
            if isinstance(props_raw.get("tags"), dict):
                tags.update(props_raw.get("tags"))
            name = _fix_mojibake(tags.get("name")) if tags.get("name") else None
            if not name:
                name = f"Unnamed Facility ({f.get('id')})"
            fac_type = tags.get("amenity") or tags.get("healthcare") or ""
            fac_type_lower = str(fac_type).lower()
            if "hospital" in fac_type_lower:
                role_int = 3
                if any(
                    kw in str(name).lower()
                    for kw in ["university", "central", "teaching"]
                ):
                    role_int = 4
            elif "clinic" in fac_type_lower:
                role_int = 2
            elif "doctor" in fac_type_lower:
                role_int = 1
            else:
                role_int = 1
            # Extract specialties
            specialties: List[str] = []
            spec_tags = tags.get("healthcare:speciality") or tags.get("speciality") or tags.get("healthcare_speciality")
            if spec_tags:
                for spec in str(spec_tags).split(";"):
                    spec_clean = spec.strip().lower()
                    mapped = None
                    # Map known specialities to nicer names
                    for k, v in {
                        "pediatrics": "Pediatrics",
                        "paediatrics": "Pediatrics",
                        "gynaecology": "OBGYN",
                        "gynecology": "OBGYN",
                        "surgery": "General Surgery",
                        "trauma": "Trauma",
                        "orthopedics": "Orthopedics",
                        "orthopaedics": "Orthopedics",
                        "neurosurgery": "Neurosurgery",
                        "burn": "Burn",
                        "icu": "ICU",
                        "emergency": "Emergency",
                        "cardiology": "Cardiology",
                        "urology": "Urology",
                    }.items():
                        if spec_clean == k:
                            mapped = v
                            break
                    if mapped:
                        specialties.append(mapped)
                    else:
                        specialties.append(spec_clean.capitalize())
            # Add base specialties for higher roles
            if role_int >= 2:
                for base in ["Blood Bank", "CT", "ICU"]:
                    if base not in specialties:
                        specialties.append(base)
            specialties = sorted(set(specialties))
            # Deterministic capacity
            osm_id = str(f.get("id"))
            rng = random.Random(osm_id)
            beds_total = rng.randint(15, 60) * role_int
            beds_av = int(beds_total * rng.uniform(0.1, 0.9))
            icu_total = rng.randint(2, 10) * role_int if role_int >= 2 else 0
            icu_av = int(icu_total * rng.uniform(0.1, 0.9))
            vent_total = icu_total * 2 if icu_total > 0 else rng.randint(0, 2)
            vent_av = int(vent_total * rng.uniform(0.1, 0.9))
            props = {
                "facility_id": osm_id,
                "name": name,
                "country": tags.get("addr:country") or tags.get("country") or "Unknown",
                "in_crisis_area": True,
                "role_of_care": f"Role{role_int}",
                "specialties": specialties,
                "capacity": {
                    "beds_total": beds_total,
                    "beds_available": beds_av,
                    "icu_total": icu_total,
                    "icu_available": icu_av,
                    "vent_total": vent_total,
                    "vent_available": vent_av,
                },
                "source": "Healthsites.io API (Auto-generated)",
                "last_updated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            features_out.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": props,
                }
            )
        # Check for pagination; assume that if number of features < 20, no more pages
        if len(feats) < 20:
            break
        page += 1
    if not features_out:
        return None
    return {"type": "FeatureCollection", "features": features_out}

def _load_demo_facilities(buffer_val: Optional[float] = None) -> Dict[str, Any]:
    global _demo_facilities_initial, _demo_facilities_state, _current_buffer

    # Pre-fetch cases outside the lock to avoid a deadlock
    cases_fc = _load_demo_cases()

    with _demo_lock:
        # If the user changed the radius on the UI, wipe the old cache
        if buffer_val is not None and buffer_val != _current_buffer:
            _demo_facilities_initial = None
            _current_buffer = buffer_val

        if _demo_facilities_initial is None:
            print(f"\n--- ATTEMPTING TO LOAD REAL FACILITIES (Buffer: {_current_buffer}) ---")
            
            # 1. Attempt to build the real facilities over the mock ones
            generated_fc = _generate_healthsites_facilities(cases_fc, buffer_val=_current_buffer)
            
            if generated_fc and generated_fc.get("features"):
                print(f"SUCCESS: Generated {len(generated_fc['features'])} real facilities mapped to demo capacities.")
                _demo_facilities_initial = generated_fc
            else:
                print("WARNING: Could not load real facilities. Falling back to sample_facilities.geojson.")
                if not DEMO_FACILITIES_GEOJSON.exists():
                    raise FileNotFoundError(f"Demo facilities missing: {DEMO_FACILITIES_GEOJSON}")
                _demo_facilities_initial = _read_json(DEMO_FACILITIES_GEOJSON)
                
            _demo_facilities_state = json.loads(json.dumps(_demo_facilities_initial))
            
        assert _demo_facilities_state is not None
        return _demo_facilities_state

def _reset_demo_facilities(buffer_val: Optional[float] = None) -> None:
    global _demo_facilities_state, _demo_facilities_initial, _current_buffer
    with _demo_lock:
        if buffer_val is not None and buffer_val != _current_buffer:
            _demo_facilities_initial = None
            _current_buffer = buffer_val

        if _demo_facilities_initial is None:
            _load_demo_facilities(buffer_val=_current_buffer)
            
        if _demo_facilities_initial is not None:
            _demo_facilities_state = json.loads(json.dumps(_demo_facilities_initial))

def _load_demo_cases() -> Dict[str, Any]:
    global _demo_cases
    with _demo_lock:
        if _demo_cases is not None:
            return _demo_cases
        if not DEMO_CASES_XLSX.exists():
            raise FileNotFoundError(f"Demo cases missing: {DEMO_CASES_XLSX}")
        df = pd.read_excel(DEMO_CASES_XLSX, sheet_name=0)
        feats = []
        for _, r in df.iterrows():
            # Extract base coordinates
            base_lat = float(r.get("Latitude"))
            base_lon = float(r.get("Longitude"))
            case_id = int(r.get("Case_ID"))
            
            # --- NEW: Seed random with case_id so jitter survives restarts perfectly ---
            rng = random.Random(case_id)
            lat = base_lat + rng.uniform(-0.0005, 0.0005)
            lon = base_lon + rng.uniform(-0.0005, 0.0005)
            
            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "sequence": int(r.get("Sequence")),
                    "case_id": case_id,
                    "name": str(r.get("Name")),
                    "age": int(r.get("Age")),
                    "sex": str(r.get("Sex")),
                    "status": str(r.get("Status")),
                    "branch": str(r.get("Branch")),
                    "clinical_category": str(r.get("Clinical_Category")),
                    "diagnosis": str(r.get("Diagnosis")),
                    "nrbc_suspected": str(r.get("NRBC_Suspected")),
                    "nato_triage": str(r.get("NATO_Triage")),
                    "evacuation_recommendation": str(r.get("Evacuation_Recommendation")),
                }
            })
        _demo_cases = {"type": "FeatureCollection", "features": feats}
        return _demo_cases

# =========================
# ROUTING / DECISION SUPPORT (DEMO)
# =========================

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _triage_requirements(triage: str) -> Dict[str, Any]:
    triage = (triage or "").lower()
    if "immediate" in triage or "(red)" in triage:
        return {"min_role": 3, "needs_icu": True, "needs_beds": True}
    if "delayed" in triage or "(yellow)" in triage:
        return {"min_role": 2, "needs_icu": False, "needs_beds": True}
    return {"min_role": 1, "needs_icu": False, "needs_beds": True}

def _role_to_int(role: str) -> int:
    role = (role or "").lower()
    if "role4" in role: return 4
    if "role3" in role: return 3
    if "role2" in role: return 2
    return 1

def _facility_score(case_lat, case_lon, fac_props: Dict[str, Any], distance_km: float, triage: str) -> float:
    cap = fac_props.get("capacity") or {}
    beds_total = max(1, int(cap.get("beds_total") or 1))
    beds_av = int(cap.get("beds_available") or 0)
    occ = 1.0 - (beds_av / beds_total)
    
    # Increased occupancy penalty from 200 to 1000. 
    # This forces the system to balance the load among all nearby facilities 
    # instead of filling up the closest one to 100% first.
    score = distance_km + (occ * 1000.0) 
    
    # Check for critical equipment during fallback routing.
    # If a critical patient is pushed into the "Relaxed/Desperate" pass,
    # heavily penalize facilities that lack the needed resources.
    triage_low = (triage or "").lower()
    if "immediate" in triage_low or "(red)" in triage_low:
        icu_av = int(cap.get("icu_available") or 0)
        vent_av = int(cap.get("vent_available") or 0)
        if icu_av <= 0:
            score += 2000.0   # Prefer traveling 2000km over going to a place with no ICU
        if vent_av <= 0:
            score += 5000.0   # Massively avoid facilities with no ventilators
            
    return score

def _search_facilities(case_lat, case_lon, facilities_fc, req, max_dist, allow_outside, triage):
    best, best_score, best_dist, best_reason = None, None, None, None
    for fac in facilities_fc.get("features", []):
        fprops = fac.get("properties") or {}
        if not allow_outside and not bool(fprops.get("in_crisis_area", True)):
            continue

        role_int = _role_to_int(fprops.get("role_of_care"))
        if role_int < int(req["min_role"]):
            continue

        cap = fprops.get("capacity") or {}
        beds_av = int(cap.get("beds_available") or 0)
        icu_av = int(cap.get("icu_available") or 0)

        # Hard stops for capacity
        if req["needs_beds"] and beds_av <= 0:
            continue
        if req["needs_icu"] and icu_av <= 0:
            continue

        flon, flat = fac["geometry"]["coordinates"]
        dist_km = _haversine_km(case_lat, case_lon, flat, flon)
        if dist_km > max_dist:
            continue

        score = _facility_score(case_lat, case_lon, fprops, dist_km, triage)
        if best_score is None or score < best_score:
            best = fac
            best_score = score
            best_dist = dist_km
            best_reason = f"Role>={req['min_role']}, ICU_OK={not req['needs_icu'] or icu_av>0}"
            
    return best, best_reason, best_dist

def _pick_facility_for_case(case_feat: Dict[str, Any], facilities_fc: Dict[str, Any], *,
                            max_distance_km: float,
                            allow_outside_area: bool) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[float]]:
    props = case_feat.get("properties") or {}
    triage = props.get("nato_triage", "")
    req = _triage_requirements(triage)
    case_lon, case_lat = case_feat["geometry"]["coordinates"]

    # PASS 1: Strict Match (Ideal conditions)
    best, reason, dist = _search_facilities(case_lat, case_lon, facilities_fc, req, max_distance_km, allow_outside_area, triage)
    if best: return best, reason, dist

    # PASS 2: Relaxed Match (Drop hard ICU requirement, accept 1 step lower role of care just to get them stabilized)
    req_relaxed = req.copy()
    req_relaxed["needs_icu"] = False
    req_relaxed["min_role"] = max(1, req["min_role"] - 1)
    best, reason, dist = _search_facilities(case_lat, case_lon, facilities_fc, req_relaxed, max_distance_km, allow_outside_area, triage)
    if best: return best, reason + " (Relaxed Constraints)", dist

    # PASS 3: Desperation Match (Expand radius x3, find literally ANY available bed)
    req_desp = {"min_role": 1, "needs_icu": False, "needs_beds": True}
    best, reason, dist = _search_facilities(case_lat, case_lon, facilities_fc, req_desp, max_distance_km * 3, allow_outside_area, triage)
    if best: return best, reason + " (Desperate/Expanded Radius)", dist

    return None, "System overwhelmed: No beds available within expanded radius", None

def _decrement_capacity(fac_props: Dict[str, Any], triage: str) -> None:
    cap = fac_props.setdefault("capacity", {})
    cap["beds_available"] = max(0, int(cap.get("beds_available") or 0) - 1)
    if "immediate" in (triage or "").lower() or "(red)" in (triage or "").lower():
        cap["icu_available"] = max(0, int(cap.get("icu_available") or 0) - 1)
        cap["vent_available"] = max(0, int(cap.get("vent_available") or 0) - 1)
    fac_props["last_updated"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# =========================
# HEALTHSITES TILE SERVER (OPTIONAL)
# =========================

def _healthsites_available() -> bool:
    if not HAVE_GEOSPATIAL:
        return False
    return all(p.exists() for p in LAYER_FILES.values())

def _ensure_layer_ready(layer: str) -> None:
    if not HAVE_GEOSPATIAL:
        raise RuntimeError('Geospatial dependencies are not installed; Healthsites layers disabled.')
    if layer in to_data_bounds:
        return

    with layer_init_lock:
        if layer in to_data_bounds:
            return

        shp = LAYER_FILES[layer]
        if not shp.exists():
            raise FileNotFoundError(f"Missing shapefile: {shp}")

        info = pyogrio.read_info(str(shp))
        crs_val = info.get("crs")
        crs = CRS.from_user_input(crs_val) if crs_val else CRS.from_epsg(4326)

        raw_fields = info.get("fields")
        fields = list(raw_fields) if raw_fields is not None else []

        enc_raw = info.get("encoding")
        enc = (str(enc_raw).strip() if enc_raw is not None else "")
        if not enc:
            enc = "UTF-8"

        t_data = Transformer.from_crs(3857, crs, always_xy=True)
        t_3857 = Transformer.from_crs(crs, 3857, always_xy=True)

        layer_crs[layer] = crs
        layer_fields[layer] = fields
        layer_encoding[layer] = enc
        to_data_bounds[layer] = t_data
        to_3857[layer] = t_3857

        layer_file_modified_iso[layer] = _iso_utc_from_epoch(shp.stat().st_mtime)

def tile_bounds_3857(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    n = 2 ** z
    tile_size = (2 * WEBMERC_MAX) / n
    minx = -WEBMERC_MAX + x * tile_size
    maxx = -WEBMERC_MAX + (x + 1) * tile_size
    maxy = WEBMERC_MAX - y * tile_size
    miny = WEBMERC_MAX - (y + 1) * tile_size
    return minx, miny, maxx, maxy

def _downsample_every_n(gdf, n: int):
    if n <= 1:
        return gdf
    return gdf.iloc[::n].copy()

def _stable_sample_by_fraction(gdf, id_field: str, target: int):
    if target <= 0:
        return gdf.iloc[0:0].copy()
    total = len(gdf)
    if total <= target:
        return gdf
    frac = target / float(total)

    try:
        ids = gdf[id_field].astype(str).fillna("").tolist()
    except Exception:
        ids = [str(i) for i in range(total)]

    import zlib
    thr = int(frac * (2**32 - 1))
    keep_mask = []
    for s in ids:
        hv = zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF
        keep_mask.append(hv <= thr)

    out = gdf.loc[keep_mask].copy()
    if len(out) < int(target * 0.85):
        need = target - len(out)
        if need > 0:
            out = gdf.iloc[:target].copy()
    return out

def _read_bbox_layer(shp: Path, bbox_tuple, *, encoding_try: str, columns: Optional[List[str]]):
    try:
        return gpd.read_file(shp, bbox=bbox_tuple, engine="pyogrio", encoding=encoding_try, columns=columns)
    except UnicodeDecodeError:
        return gpd.read_file(shp, bbox=bbox_tuple, engine="pyogrio", encoding="latin1", columns=columns)

def _clean_geom(geom):
    if geom is None or geom.is_empty:
        return None
    gt = geom.geom_type
    if gt == "GeometryCollection":
        parts = [g for g in geom.geoms if g is not None and not g.is_empty]
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return unary_union(parts)
    return geom

def _nodes_target_count(z: int) -> int:
    if z <= 11:
        return 500
    if z == 12:
        return 900
    if z == 13:
        return 1400
    return MAX_FEATURES_NODES

def _compose_addr_full(props: Dict[str, object]) -> Optional[str]:
    hn = props.get("addr:housenumber")
    st = props.get("addr:street")
    pc = props.get("addr:postcode")
    ct = props.get("addr:city")

    parts = []
    street_line = " ".join([str(x).strip() for x in [st, hn] if x and str(x).strip()])
    if street_line:
        parts.append(street_line)
    city_line = " ".join([str(x).strip() for x in [pc, ct] if x and str(x).strip()])
    if city_line:
        parts.append(city_line)

    return ", ".join(parts) if parts else None

def _normalize_props(layer: str, row_dict: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}

    for src_key, val in row_dict.items():
        if src_key not in SHAPE_TO_CANON:
            continue
        canon_key = SHAPE_TO_CANON[src_key]
        try:
            if val != val:  # NaN
                val = None
            else:
                # Add the mojibake fix here
                val = _fix_mojibake(val)
        except Exception:
            pass
        out[canon_key] = val

    if "upstream_id" not in out and row_dict.get("osm_id") is not None:
        out["upstream_id"] = row_dict.get("osm_id")

    if "date-modified" not in out:
        out["date-modified"] = layer_file_modified_iso.get(layer)

    addr_full = _compose_addr_full(out)
    if addr_full:
        out["addr:full"] = addr_full

    if "health_facility:type" not in out:
        t = out.get("healthcare") or out.get("amenity") or out.get("health_amenity")
        if t:
            out["health_facility:type"] = t

    return out


# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/favicon.ico")
def favicon():
    return Response(b"", mimetype="image/x-icon")

@app.route("/api/healthsites/status")
def api_healthsites_status():
    return jsonify({
        "have_geospatial": HAVE_GEOSPATIAL,
        "healthsites_available": _healthsites_available(),
        "data_dir": str(DATA_DIR),
        "expected_files": {k: str(v) for k, v in LAYER_FILES.items()},
        "min_zoom_nodes": MIN_ZOOM_NODES,
    })

# --- Demo APIs ---
@app.route("/api/demo/facilities")
def api_demo_facilities():
    buffer_param = request.args.get("buffer")
    buffer_val = float(buffer_param) if buffer_param else None
    fc = _load_demo_facilities(buffer_val=buffer_val)
    
    # Optional bbox filter: ?bbox=minLon,minLat,maxLon,maxLat
    bbox_str = request.args.get("bbox")
    in_area = request.args.get("in_crisis_area")
    feats = fc.get("features", [])
    if bbox_str:
        try:
            minlon, minlat, maxlon, maxlat = [float(x) for x in bbox_str.split(",")]
            feats = [f for f in feats if (minlon <= f["geometry"]["coordinates"][0] <= maxlon and
                                          minlat <= f["geometry"]["coordinates"][1] <= maxlat)]
        except Exception:
            pass
    if in_area in ("1", "true", "True"):
        feats = [f for f in feats if bool((f.get("properties") or {}).get("in_crisis_area"))]
    elif in_area in ("0", "false", "False"):
        feats = [f for f in feats if not bool((f.get("properties") or {}).get("in_crisis_area"))]
    return jsonify({"type": "FeatureCollection", "features": feats})

@app.route("/api/demo/cases")
def api_demo_cases():
    fc = _load_demo_cases()
    triage = request.args.get("triage")
    feats = fc.get("features", [])
    if triage:
        triage_l = triage.lower()
        feats = [f for f in feats if triage_l in (f.get("properties", {}).get("nato_triage","").lower())]
    return jsonify({"type": "FeatureCollection", "features": feats})

@app.route("/api/demo/reset", methods=["POST"])
def api_demo_reset():
    body = request.get_json(force=True, silent=True) or {}
    buffer_val = body.get("buffer")
    if buffer_val is not None:
        buffer_val = float(buffer_val)
    _reset_demo_facilities(buffer_val=buffer_val)
    return jsonify({"ok": True})

@app.route("/api/demo/osrm_cache", methods=["GET", "POST"])
def api_demo_osrm_cache():
    with _osrm_cache_lock:
        if request.method == "GET":
            # Serve existing cache to the frontend
            if OSRM_CACHE_FILE.exists():
                try:
                    with open(OSRM_CACHE_FILE, "r", encoding="utf-8") as f:
                        return Response(f.read(), mimetype="application/json")
                except Exception as e:
                    print(f"Error reading OSRM cache: {e}")
            return jsonify({})
        
        # POST: Save newly calculated routes from the frontend
        new_data = request.get_json(force=True, silent=True) or {}
        if not new_data:
            return jsonify({"ok": False, "reason": "No data"})
            
        cache = {}
        if OSRM_CACHE_FILE.exists():
            try:
                with open(OSRM_CACHE_FILE, "r", encoding="utf-8") as f:
                    cache = json.load(f)
            except Exception:
                pass
                
        cache.update(new_data)
        DEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with open(OSRM_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        except Exception as e:
            print(f"Error writing OSRM cache: {e}")
            return jsonify({"ok": False, "reason": str(e)})
            
        return jsonify({"ok": True})

@app.route("/api/demo/route", methods=["POST"])
def api_demo_route():
    body = request.get_json(force=True, silent=True) or {}
    triage_filters = body.get("triage_filters")  
    max_distance_km = float(body.get("max_distance_km", 350))
    allow_outside_area = True
    max_routes_draw = int(body.get("max_routes_draw", 120))
    buffer_val = body.get("buffer")
    if buffer_val is not None:
        buffer_val = float(buffer_val)

    _reset_demo_facilities(buffer_val=buffer_val)

    cases_fc = _load_demo_cases()
    facilities_fc = _load_demo_facilities(buffer_val=buffer_val)

    feats = cases_fc.get("features", [])
    if triage_filters and isinstance(triage_filters, list):
        allowed = {str(x).strip().lower() for x in triage_filters if str(x).strip()}
        feats = [f for f in feats if (f.get("properties", {}).get("nato_triage", "").strip().lower() in allowed)]

    assignments = []
    unassigned = 0

    # NOTE: This mutates facility capacity state. Call /api/demo/reset to re-run from baseline.
    for cf in feats:
        best, reason, dist = _pick_facility_for_case(cf, facilities_fc, max_distance_km=max_distance_km,
                                                    allow_outside_area=allow_outside_area)
        if best is None:
            unassigned += 1
            continue

        cprops = cf.get("properties") or {}
        triage = cprops.get("nato_triage", "")
        _decrement_capacity(best["properties"], triage)

        assignments.append({
            "case_id": int(cprops.get("case_id")),
            "nato_triage": triage,
            "facility_id": best["properties"].get("facility_id"),
            "facility_name": best["properties"].get("name"),
            "distance_km": float(dist or 0.0),
            "reason": reason,
            "case_coord": cf["geometry"]["coordinates"],
            "facility_coord": best["geometry"]["coordinates"],
        })

    # Summarize facility utilization
    facility_summary = []
    for f in facilities_fc.get("features", []):
        p = f.get("properties") or {}
        cap = p.get("capacity") or {}
        facility_summary.append({
            "facility_id": p.get("facility_id"),
            "name": p.get("name"),
            "in_crisis_area": bool(p.get("in_crisis_area", True)),
            "role_of_care": p.get("role_of_care"),
            "beds_available": int(cap.get("beds_available") or 0),
            "beds_total": int(cap.get("beds_total") or 0),
            "icu_available": int(cap.get("icu_available") or 0),
            "icu_total": int(cap.get("icu_total") or 0),
            "last_updated": p.get("last_updated"),
        })

    # Limit route lines for UI
    routes_for_map = assignments[:max_routes_draw]

    return jsonify({
        "ok": True,
        "params": {
            "max_distance_km": max_distance_km,
            "allow_outside_area": allow_outside_area,
            "triage_filters": triage_filters,
        },
        "results": {
            "cases_considered": len(feats),
            "assigned": len(assignments),
            "unassigned": unassigned,
        },
        "routes_for_map": routes_for_map,
        "facility_summary": facility_summary,
    })


# --- Vector tiles (Healthsites) ---
@app.route("/tiles/<layer>/<int:z>/<int:x>/<int:y>.pbf")
def tiles(layer: str, z: int, x: int, y: int):
    if layer not in LAYER_FILES:
        abort(404)

    if not _healthsites_available():
        # Return empty tiles cleanly (so UI can keep working with demo layer)
        return Response(b"", mimetype="application/x-protobuf")

    key = (layer, z, x, y)
    if key in tile_cache:
        data = tile_cache[key]
    else:
        data = build_mvt_tile(layer, z, x, y)
        tile_cache[key] = data

    return Response(
        data,
        mimetype="application/x-protobuf",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def build_mvt_tile(layer: str, z: int, x: int, y: int) -> bytes:
    if layer == "nodes" and z < MIN_ZOOM_NODES:
        return b""

    _ensure_layer_ready(layer)

    minx, miny, maxx, maxy = tile_bounds_3857(z, x, y)
    tile_poly_3857 = box(minx, miny, maxx, maxy)

    tr = to_data_bounds[layer]
    data_minx, data_miny, data_maxx, data_maxy = tr.transform_bounds(
        minx, miny, maxx, maxy, densify_pts=21
    )

    shp = LAYER_FILES[layer]
    fields = layer_fields[layer]

    if layer == "nodes":
        desired = NODE_FIELDS_LOW if z <= 12 else NODE_FIELDS_HIGH
    else:
        desired = WAYS_FIELDS_LOW if z <= 10 else WAYS_FIELDS_HIGH

    wanted = [c for c in desired if c in fields]
    gdf = _read_bbox_layer(
        shp,
        (data_minx, data_miny, data_maxx, data_maxy),
        encoding_try=layer_encoding[layer],
        columns=wanted if wanted else None,
    )
    if gdf.empty:
        return b""

    gdf = gdf.to_crs(3857)
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[~gdf.geometry.is_empty]
    if gdf.empty:
        return b""

    if layer == "nodes":
        xs = gdf.geometry.x
        ys = gdf.geometry.y
        gdf = gdf[(xs >= minx) & (xs <= maxx) & (ys >= miny) & (ys <= maxy)]
    else:
        gdf = gdf[gdf.geometry.intersects(tile_poly_3857)]
        if gdf.empty:
            return b""
        gdf["geometry"] = gdf.geometry.intersection(tile_poly_3857)
        meters_per_pixel = (maxx - minx) / EXTENT
        tol = meters_per_pixel * 2.0
        gdf["geometry"] = gdf.geometry.simplify(tol, preserve_topology=True)

    if gdf.empty:
        return b""

    if layer == "ways" and len(gdf) > MAX_FEATURES_WAYS:
        step = max(1, len(gdf) // MAX_FEATURES_WAYS)
        gdf = _downsample_every_n(gdf, step)

    if layer == "nodes":
        target = _nodes_target_count(z)
        if len(gdf) > target:
            id_field = "osm_id" if "osm_id" in gdf.columns else (gdf.columns[0] if len(gdf.columns) else "osm_id")
            gdf = _stable_sample_by_fraction(gdf, id_field, target)
        if len(gdf) > MAX_FEATURES_NODES:
            id_field = "osm_id" if "osm_id" in gdf.columns else (gdf.columns[0] if len(gdf.columns) else "osm_id")
            gdf = _stable_sample_by_fraction(gdf, id_field, MAX_FEATURES_NODES)

    features = []
    for _, row in gdf.iterrows():
        geom = _clean_geom(row.geometry)
        if geom is None or geom.is_empty:
            continue

        row_dict = {}
        for c in wanted:
            try:
                row_dict[c] = row[c]
            except Exception:
                pass

        props = _normalize_props(layer, row_dict)

        if z <= 12:
            keep_keys = {
                "upstream_id", "name", "source", "date-modified",
                "health_facility:type", "amenity", "healthcare",
                "contact:phone", "opening_hours", "bed_count",
                "operator", "operator:type",
                "addr:full", "website",
            }
            props = {k: v for k, v in props.items() if (k in keep_keys and v not in (None, ""))}

        f = {"geometry": geom, "properties": props}
        try:
            f["id"] = int(props.get("upstream_id"))
        except Exception:
            pass
        features.append(f)

    if not features:
        return b""

    tile = mapbox_vector_tile.encode(
        [{"name": layer, "features": features}],
        default_options={
            "extents": EXTENT,
            "quantize_bounds": (minx, miny, maxx, maxy),
        },
    )
    return tile


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True, threaded=True)