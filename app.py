from __future__ import annotations

import os
import math
import json
import random
import threading
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime, timezone

from flask import Flask, Response, abort, render_template, jsonify, request

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
    if not _healthsites_available():
        print(" -> FAILED: _healthsites_available() returned False.")
        print(" -> Check if 'Europe-node.shp' is inside your 'DATA_DIR' folder.")
        return None

    feats = cases_fc.get("features", [])
    if not feats:
        print(" -> FAILED: No MASCAL cases found to build a bounding box.")
        return None

    # 1. Establish the "area" by calculating the bounding box of all casualties
    lons = [f["geometry"]["coordinates"][0] for f in feats]
    lats = [f["geometry"]["coordinates"][1] for f in feats]

    BUFFER = buffer_val
    minlon, maxlon = min(lons) - BUFFER, max(lons) + BUFFER
    minlat, maxlat = min(lats) - BUFFER, max(lats) + BUFFER

    # 2. Fast spatial filter using pyogrio bbox reading
    shp = LAYER_FILES["nodes"]
    try:
        print(f" -> Querying shapefile bounding box: {minlon:.2f}, {minlat:.2f}, {maxlon:.2f}, {maxlat:.2f}")
        # Try default UTF-8 first
        gdf = gpd.read_file(shp, bbox=(minlon, minlat, maxlon, maxlat), engine="pyogrio", encoding="utf-8")
    except UnicodeDecodeError:
        print(" -> UTF-8 decode failed, falling back to Latin-1 encoding...")
        try:
            # Fallback for European shapefiles
            gdf = gpd.read_file(shp, bbox=(minlon, minlat, maxlon, maxlat), engine="pyogrio", encoding="latin1")
        except Exception as e:
            print(f" -> FAILED: Error reading shapefile with Latin-1: {e}")
            return None
    except Exception as e:
        print(f" -> FAILED: Error reading shapefile: {e}")
        return None

    if gdf.empty:
        print(" -> FAILED: Bounding box query returned 0 facilities in this area.")
        return None
        
    print(f" -> Found {len(gdf)} raw locations. Filtering and applying demo data...")

    # 3. Filter out irrelevant facilities (like pharmacies, dentists)
    valid_types = ["hospital", "clinic", "doctors"]
    mask = (gdf["amenity"].isin(valid_types)) | (gdf["healthcare"].isin(valid_types))
    gdf = gdf[mask]

    # 4. Generate mock data for the real locations
    features = []
    
    # Possible specialties to randomize
    possible_specialties = ["General Surgery", "Orthopedics", "Trauma", "Neurosurgery", "Burn", "Pediatrics", "OBGYN"]

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        name = _fix_mojibake(row.get("name"))
        
        if not name or str(name).lower() == "nan":
            name = f"Unnamed Facility ({row.get('osm_id', idx)})"
            
        osm_id = row.get("osm_id", idx)

        # Seed random with facility ID so beds/ICU stay constant across restarts
        rng = random.Random(str(osm_id))

        # Assign a random Role of Care
        role_int = rng.choices([1, 2, 3, 4], weights=[0.4, 0.3, 0.2, 0.1])[0]
        
        # Base Beds
        beds_total = rng.randint(15, 60) * role_int
        beds_av = int(beds_total * rng.uniform(0.1, 0.9))

        # ICU
        icu_total = rng.randint(2, 10) * role_int if role_int >= 2 else 0
        icu_av = int(icu_total * rng.uniform(0.1, 0.9))
        
        # Ventilators (typically scales with ICU)
        vent_total = icu_total * 2 if icu_total > 0 else rng.randint(0, 2)
        vent_av = int(vent_total * rng.uniform(0.1, 0.9))

        # Randomize specialties based on Role of Care
        num_specs = rng.randint(1, 2) + (role_int - 1)
        specialties = ["Blood Bank", "CT", "ICU"] if role_int >= 2 else []
        specialties += rng.sample(possible_specialties, min(num_specs, len(possible_specialties)))
        specialties = sorted(list(set(specialties)))

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
                "vent_available": vent_av
            },
            "source": "healthmap.io (Auto-generated)",
            "last_updated": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        }

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [geom.x, geom.y]},
            "properties": props
        })

    return {"type": "FeatureCollection", "features": features}

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
            
            # Seed random with case_id so jitter survives restarts perfectly
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

def _case_requirements(props: Dict[str, Any]) -> Dict[str, Any]:
    triage = (props.get("nato_triage") or "").lower()
    diagnosis = props.get("diagnosis") or ""
    
    # Determine base bed/ICU needs
    if "immediate" in triage or "(red)" in triage:
        req = {"min_role": 3, "needs_icu": True, "needs_beds": True}
    elif "delayed" in triage or "(yellow)" in triage:
        req = {"min_role": 2, "needs_icu": False, "needs_beds": True}
    else:
        req = {"min_role": 1, "needs_icu": False, "needs_beds": True}
        
    # Determine specialty requirement
    req["required_specialty"] = DIAGNOSIS_SPECIALTY_MAP.get(diagnosis)
    
    return req

def _role_to_int(role: str) -> int:
    role = (role or "").lower()
    if "role4" in role: return 4
    if "role3" in role: return 3
    if "role2" in role: return 2
    return 1

def _facility_score(case_lat, case_lon, fac_props: Dict[str, Any], distance_km: float, triage: str, required_specialty: str = None) -> float:
    cap = fac_props.get("capacity") or {}
    beds_total = max(1, int(cap.get("beds_total") or 1))
    beds_av = int(cap.get("beds_available") or 0)
    occ = 1.0 - (beds_av / beds_total)
    
    # Base score is distance + occupancy penalty
    score = distance_km + (occ * 1000.0) 
    
    # --- SPECIALTY AI LOGIC ---
    if required_specialty:
        fac_specialties = fac_props.get("specialties", [])
        if required_specialty not in fac_specialties:
            # Add a massive 3000km penalty if they don't have the right specialty.
            # This makes the algorithm drive past a close facility to find the right specialist.
            score += 3000.0
            
    # Check for critical equipment during fallback routing.
    triage_low = (triage or "").lower()
    if "immediate" in triage_low or "(red)" in triage_low:
        icu_av = int(cap.get("icu_available") or 0)
        vent_av = int(cap.get("vent_available") or 0)
        if icu_av <= 0:
            score += 2000.0   
        if vent_av <= 0:
            score += 5000.0   
            
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

        if req["needs_beds"] and beds_av <= 0:
            continue
        if req["needs_icu"] and icu_av <= 0:
            continue

        flon, flat = fac["geometry"]["coordinates"]
        dist_km = _haversine_km(case_lat, case_lon, flat, flon)
        if dist_km > max_dist:
            continue

        # Pass the specialty to the scoring function
        score = _facility_score(case_lat, case_lon, fprops, dist_km, triage, req.get("required_specialty"))
        
        if best_score is None or score < best_score:
            best = fac
            best_score = score
            best_dist = dist_km
            
            # Format the reason for the UI
            spec_text = f"Requires {req['required_specialty']}" if req.get("required_specialty") else "Gen Med"
            best_reason = f"Role>={req['min_role']}, ICU_OK={not req['needs_icu'] or icu_av>0} ({spec_text})"
            
    return best, best_reason, best_dist

def _pick_facility_for_case(case_feat: Dict[str, Any], facilities_fc: Dict[str, Any], *,
                            max_distance_km: float,
                            allow_outside_area: bool) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[float]]:
    props = case_feat.get("properties") or {}
    triage = props.get("nato_triage", "")
    
    # Use the new function that considers Diagnosis
    req = _case_requirements(props) 
    
    case_lon, case_lat = case_feat["geometry"]["coordinates"]

    # PASS 1: Strict Match
    best, reason, dist = _search_facilities(case_lat, case_lon, facilities_fc, req, max_distance_km, allow_outside_area, triage)
    if best: return best, reason, dist

    # PASS 2: Relaxed Match (Drop ICU req, accept lower role)
    req_relaxed = req.copy()
    req_relaxed["needs_icu"] = False
    req_relaxed["min_role"] = max(1, req["min_role"] - 1)
    best, reason, dist = _search_facilities(case_lat, case_lon, facilities_fc, req_relaxed, max_distance_km, allow_outside_area, triage)
    if best: return best, reason + " (Relaxed Constraints)", dist

    # PASS 3: Desperation Match (Expand radius x3, find ANY bed, ignore specialty preference)
    req_desp = {"min_role": 1, "needs_icu": False, "needs_beds": True, "required_specialty": None} # Drop specialty requirement
    best, reason, dist = _search_facilities(case_lat, case_lon, facilities_fc, req_desp, max_distance_km * 3, allow_outside_area, triage)
    if best: return best, reason + " (Desperate/Expanded Radius)", dist

    return None, "System overwhelmed: No beds available", None

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

DIAGNOSIS_SPECIALTY_MAP = {
    # Combat Trauma
    "Blast polytrauma": "Trauma",
    "Traumatic amputation": "Trauma", # or General Surgery
    "GSW extremity": "Orthopedics",
    "GSW abdomen": "General Surgery",
    "Blast TBI": "Neurosurgery",
    "Penetrating shrapnel injury": "Trauma",
    "Burns (explosion)": "Burn",
    "Vehicle crush injury": "Trauma",
    
    # Civilian / Medical
    "Burns 20%": "Burn",
    "Closed fracture": "Orthopedics",
    "Open fracture": "Orthopedics",
    "Acute MI": "ICU",
    "Stroke": "ICU",
    "Ischemic stroke": "ICU",
    "Diabetic ketoacidosis": "ICU",
    "Blast lung injury": "Trauma",
    "Smoke inhalation": "Burn", # Or ICU
    "Crush injury": "Trauma",
    "Severe pneumonia": "ICU",
    "Sepsis": "ICU",
    
    # Lower acuity (can be handled by general medical / Role 1 or 2)
    "Concussion": None,
    "Psychological trauma": None,
    "Asthma attack": None,
    "Severe asthma exacerbation": None,
    "COPD exacerbation": None,
    "Acute appendicitis": "General Surgery",
    "Renal colic": None,
    "Pulmonary embolism": "ICU",
    "Heat stroke": None,
}

# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    mapbox_token = os.environ.get("MAPBOX_TOKEN", "")
    return render_template("index.html", mapbox_token=mapbox_token)

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
            except Exception as e:
                # If corrupted, don't just pass silently—print a warning
                print(f"WARNING: Could not load cache, resetting to empty. Error: {e}")
                
        cache.update(new_data)
        DEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            # ATOMIC WRITE: Write to a temp file first, then replace.
            # This prevents corruption if the server is stopped mid-write.
            fd, temp_path = tempfile.mkstemp(dir=DEMO_CACHE_DIR, text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(cache, f)
            os.replace(temp_path, OSRM_CACHE_FILE)
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