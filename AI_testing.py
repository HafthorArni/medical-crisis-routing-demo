import json
import spacy
from itertools import product

# Load spaCy's pre-trained model
nlp = spacy.load("en_core_web_lg")

def preprocess_field(field):
    """Preprocess field names for similarity comparison."""
    return field.replace("_", " ").replace(".", " ").lower()

def load_json_file(filepath):
    """Load a JSON file and return its flattened fields."""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Flatten the JSON structure to get all field paths
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

def match_fields(fields1, fields2, threshold=0.7):
    """Match fields between two sets using spaCy similarity."""
    matches = []
    unmatched1 = set(fields1)
    unmatched2 = set(fields2)

    for field1, field2 in product(fields1, fields2):
        doc1 = nlp(preprocess_field(field1))
        doc2 = nlp(preprocess_field(field2))
        similarity = doc1.similarity(doc2)

        if similarity >= threshold:
            matches.append((field1, field2, similarity))
            unmatched1.discard(field1)
            unmatched2.discard(field2)

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

    # Match fields
    matches, unmatched1, unmatched2 = match_fields(fields1, fields2)

    print("\n--- Matched Fields (Similarity >= 0.7) ---")
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
