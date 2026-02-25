"""
Generate a small synthetic facility dataset for demos.

Usage:
  python scripts/generate_sample_facilities.py --out data/sample_facilities.geojson
"""
from __future__ import annotations
import argparse, json, random
from datetime import datetime, timezone

def random_facilities(n_inside=30, n_outside=20, seed=42):
    rng=random.Random(seed)
    inside_lat=(56.0,57.6)
    inside_lon=(21.0,25.8)
    outside_lat=(54.0,59.5)
    outside_lon=(19.0,30.0)

    def gen_point(in_area=True):
        if in_area:
            lat = rng.uniform(*inside_lat)
            lon = rng.uniform(*inside_lon)
        else:
            for _ in range(1000):
                lat = rng.uniform(*outside_lat)
                lon = rng.uniform(*outside_lon)
                if not (inside_lat[0]<=lat<=inside_lat[1] and inside_lon[0]<=lon<=inside_lon[1]):
                    break
        return lat, lon

    roles=["Role1","Role2","Role3","Role4"]
    specialties_pool=["Trauma","Orthopedics","Neurosurgery","Burn","Pediatrics","General Surgery","ICU","CT","Blood Bank","OBGYN"]
    countries=["LV","LT","EE","PL","SE","FI"]

    feats=[]
    fid=1
    for in_area, count in [(True,n_inside),(False,n_outside)]:
        for _ in range(count):
            lat,lon=gen_point(in_area)
            role=rng.choices(roles, weights=[2,3,3,1])[0] if in_area else rng.choices(roles, weights=[1,2,3,3])[0]

            base=[]
            if role in ("Role3","Role4"):
                base += ["Trauma","General Surgery","ICU","CT","Blood Bank"]
            elif role=="Role2":
                base += ["General Surgery","ICU"]
            else:
                base += ["Trauma"]

            extras=rng.sample([s for s in specialties_pool if s not in base], k=rng.randint(1,3))
            specs=sorted(set(base+extras))

            if role=="Role4":
                beds_total=rng.randint(200,800); icu_total=rng.randint(20,80)
            elif role=="Role3":
                beds_total=rng.randint(80,250);  icu_total=rng.randint(8,30)
            elif role=="Role2":
                beds_total=rng.randint(20,80);   icu_total=rng.randint(2,10)
            else:
                beds_total=rng.randint(5,25);    icu_total=0

            beds_avail=rng.randint(int(beds_total*0.05), int(beds_total*0.6))
            icu_avail=rng.randint(0, int(icu_total*0.6)) if icu_total>0 else 0
            vent_total=icu_total*2 if icu_total>0 else rng.randint(0,4)
            vent_avail=rng.randint(0, max(vent_total//2,1)) if vent_total>0 else 0

            country=rng.choice(countries)
            name=f"{country}-Facility-{fid:03d}"

            feats.append({
                "type":"Feature",
                "geometry":{"type":"Point","coordinates":[lon,lat]},
                "properties":{
                    "facility_id": f"FAC{fid:04d}",
                    "name": name,
                    "country": country,
                    "in_crisis_area": bool(in_area),
                    "role_of_care": role,
                    "specialties": specs,
                    "capacity": {
                        "beds_total": beds_total,
                        "beds_available": beds_avail,
                        "icu_total": icu_total,
                        "icu_available": icu_avail,
                        "vent_total": vent_total,
                        "vent_available": vent_avail
                    },
                    "source": "demo",
                    "last_updated": datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
                }
            })
            fid += 1

    return {"type":"FeatureCollection","features":feats}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sample_facilities.geojson")
    ap.add_argument("--inside", type=int, default=30)
    ap.add_argument("--outside", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    fc=random_facilities(args.inside, args.outside, args.seed)
    with open(args.out,"w",encoding="utf-8") as f:
        json.dump(fc,f)
    print(f"Wrote {len(fc['features'])} facilities to {args.out}")

if __name__=="__main__":
    main()
