import json
from fuzzywuzzy import fuzz

# Keyword-based rules
KEYWORD_RULES = {
    "bed": ["total_beds", "available_beds", "bedCapacity", "availableBeds", "ICUBeds"],
    "trauma": ["has_trauma_center", "isTraumaCenter"],
    "lat": ["location.latitude", "coordinates.lat"],
    "lng": ["location.longitude", "coordinates.lng"],
    "doctor": ["staff.doctors", "staffInfo.numDoctors"],
    "nurse": ["staff.nurses", "staffInfo.numNurses"],
    "specialist": ["staff.specialists", "staffInfo.specialties"],
    "id": ["hospital_id", "hospitalID"],
    "name": ["name", "hospitalName"],
    "update": ["last_updated", "lastUpdate"],
}

def load_json_file(filepath):
    """Load a JSON file and return its flattened fields."""
    with open(filepath, "r") as f:
        data = json.load(f)

    flat_fields = set()
    def flatten(data, parent_key=""):
        for k, v in data.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                flatten(v, new_key)
            else:
                flat_fields.add(new_key)
    flatten(data)
    return flat_fields

def match_fields_keywords(fields1, fields2):
    """Match fields using keyword rules."""
    matches = []
    unmatched1 = set(fields1)
    unmatched2 = set(fields2)

    for field1 in fields1:
        for field2 in fields2:
            for keyword, variants in KEYWORD_RULES.items():
                if (keyword in field1.lower() and keyword in field2.lower()):
                    matches.append((field1, field2, 1.0))
                    unmatched1.discard(field1)
                    unmatched2.discard(field2)
                    break
            else:
                continue
            break

    return matches, unmatched1, unmatched2

def match_fields_fuzzy(fields1, fields2, threshold=80):
    """Match fields using fuzzy string matching."""
    matches = []
    unmatched1 = set(fields1)
    unmatched2 = set(fields2)

    for field1 in fields1:
        for field2 in fields2:
            score = fuzz.ratio(field1.lower(), field2.lower())
            if score >= threshold:
                matches.append((field1, field2, score))
                unmatched1.discard(field1)
                unmatched2.discard(field2)
                break

    return matches, unmatched1, unmatched2

def match_fields_hybrid(fields1, fields2, fuzzy_threshold=80):
    """Match fields using both keyword rules and fuzzy matching."""
    # Step 1: Apply keyword rules
    matches, unmatched1, unmatched2 = match_fields_keywords(fields1, fields2)

    # Step 2: Apply fuzzy matching to remaining fields
    fuzzy_matches, unmatched1, unmatched2 = match_fields_fuzzy(
        unmatched1, unmatched2, fuzzy_threshold
    )
    matches.extend(fuzzy_matches)

    return matches, unmatched1, unmatched2

def main():
    # Load fields from both JSON files
    fields1 = load_json_file("hosp1.json")
    fields2 = load_json_file("hosp2.json")

    print("--- Fields in hosp1.json ---")
    for field in sorted(fields1):
        print(f"- {field}")

    print("\n--- Fields in hosp2.json ---")
    for field in sorted(fields2):
        print(f"- {field}")

    # Match fields using hybrid approach
    matches, unmatched1, unmatched2 = match_fields_hybrid(fields1, fields2)

    print("\n--- Matched Fields ---")
    for field1, field2, score in matches:
        print(f"- {field1} <-> {field2} (score: {score:.2f})")

    print("\n--- Unmatched Fields in hosp1.json ---")
    for field in sorted(unmatched1):
        print(f"- {field}")

    print("\n--- Unmatched Fields in hosp2.json ---")
    for field in sorted(unmatched2):
        print(f"- {field}")

if __name__ == "__main__":
    # Create mock JSON files for testing (if they don't exist)
    mock_hosp1 = {
        "hospital_id": "city_general",
        "name": "City General Hospital",
        "location": {"latitude": 64.1265, "longitude": -21.8174},
        "total_beds": 200,
        "available_beds": 45,
        "icu_beds": 10,
        "has_trauma_center": True,
        "has_burn_unit": False,
        "staff": {
            "doctors": 50,
            "nurses": 120,
            "specialists": ["trauma", "cardiology"]
        },
        "last_updated": "2026-02-24T12:00:00Z"
    }

    mock_hosp2 = {
        "hospitalID": "northside_medical",
        "hospitalName": "Northside Medical Center",
        "coordinates": {"lat": 64.1312, "lng": -21.8987},
        "bedCapacity": 150,
        "availableBeds": 30,
        "ICUBeds": 5,
        "isTraumaCenter": False,
        "isPediatric": True,
        "staffInfo": {
            "numDoctors": 35,
            "numNurses": 90,
            "specialties": ["pediatrics", "neurology"]
        },
        "lastUpdate": "2026-02-24T11:30:00Z"
    }

    # Save mock data to JSON files
    with open("hosp1.json", "w") as f:
        json.dump(mock_hosp1, f, indent=2)
    with open("hosp2.json", "w") as f:
        json.dump(mock_hosp2, f, indent=2)

    # Run the matching
    main()
