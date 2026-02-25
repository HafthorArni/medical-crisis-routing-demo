import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define your target schema
TARGET_SCHEMA = {
    "Number_of_beds": ["total_beds", "bedCapacity", "bed_count", "Amount_of_Beds", "Beds"],
    "Number_of_ICU_beds": ["icu_beds", "ICUBeds", "intensive_care_beds", "critical_care_beds"],
    "Number_of_available_beds": ["available_beds", "availableBeds", "open_beds", "free_beds"],
    "Has_trauma_center": ["has_trauma_center", "isTraumaCenter", "trauma_facility", "emergency_trauma_unit"],
    "Has_burn_unit": ["has_burn_unit", "isBurnUnit", "burn_unit"],
    "Has_pediatric_unit": ["isPediatric", "child_care_unit", "pediatric_ward"],
    "Number_of_doctors": ["staff.doctors", "staffInfo.numDoctors", "personnel.physicians", "employees.doctors"],
    "Number_of_nurses": ["staff.nurses", "staffInfo.numNurses", "personnel.nursing_staff", "employees.nurses"],
    "Medical_specialties": ["staff.specialists", "staffInfo.specialties", "personnel.medical_specialties", "employees.expertise"],
    "Last_updated": ["last_updated", "lastUpdate", "last_modified", "update_time"],
    "Latitude": ["latitude", "lat"],
    "Longitude": ["longitude", "long"],
    "Hospital_name": ["Medical_center", "hospital_id", "ID", "facility", "center", "healthcare"]
}

def preprocess_field(field):
    """Preprocess field names for semantic comparison."""
    return field.replace("_", " ").replace(".", " ").lower()

def load_json_file(filepath):
    """Load a JSON file and return its flattened fields."""
    with open(filepath, "r") as f:
        data = json.load(f)

    flat_fields = {}
    def flatten(data, parent_key=""):
        for k, v in data.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                flatten(v, new_key)
            else:
                flat_fields[new_key] = v
    flatten(data)
    return flat_fields

def map_to_target_schema(api_fields, target_schema, threshold=0.7):
    """Map API fields to the target schema using Sentence-BERT."""
    # Flatten target schema variants
    target_variants = []
    target_field_names = []
    for target_field, variants in target_schema.items():
        target_field_names.append(target_field)
        target_variants.extend(variants)

    # Preprocess and encode all fields
    api_field_texts = list(api_fields.keys())
    target_field_texts = [preprocess_field(variant) for variant in target_variants]
    all_texts = api_field_texts + target_field_texts
    processed_texts = [preprocess_field(text) for text in all_texts]
    embeddings = model.encode(processed_texts)

    # Split embeddings
    n = len(api_field_texts)
    api_embeddings = embeddings[:n]
    target_embeddings = embeddings[n:]

    # Compute cosine similarity
    similarities = cosine_similarity(api_embeddings, target_embeddings)

    # Find best match for each API field
    mapped = {}
    unmatched = {}
    variant_to_target = {}
    for target_field, variants in target_schema.items():
        for variant in variants:
            variant_to_target[variant] = target_field

    for i, api_field in enumerate(api_field_texts):
        best_score = 0
        best_target = None
        best_variant = None

        for j, variant in enumerate(target_variants):
            score = similarities[i, j]
            if score > best_score:
                best_score = score
                best_variant = variant
                best_target = variant_to_target[variant]

        if best_score >= threshold:
            mapped[api_field] = {
                "target_field": best_target,
                "matched_variant": best_variant,
                "score": best_score,
                "value": api_fields[api_field]
            }
        else:
            unmatched[api_field] = api_fields[api_field]

    return mapped, unmatched

def main():
    # List of hospital JSON files to process
    hospital_files = ["hosp1.json", "hosp2.json", "hosp3.json", "hosp4.json"]

    # Track which files have unmatched fields
    files_with_unmatched = {}

    for file in hospital_files:
        try:
            api_fields = load_json_file(file)
            mapped, unmatched = map_to_target_schema(api_fields, TARGET_SCHEMA)

            if unmatched:
                files_with_unmatched[file] = unmatched

            print(f"\n--- Results for {file} ---")
            print("\nMapped Fields:")
            for api_field, info in mapped.items():
                print(f"- {api_field} -> {info['target_field']} "
                      f"(matched: {info['matched_variant']}, score: {info['score']:.2f}, value: {info['value']})")

            if unmatched:
                print("\nUnmatched Fields:")
                for field, value in unmatched.items():
                    print(f"- {field}: {value}")
            else:
                print("\nNo unmatched fields.")
        except FileNotFoundError:
            print(f"\n--- File {file} not found. ---")

    # Print summary of files with unmatched fields
    print("\n=== Summary: Files with Unmatched Fields ===")
    if files_with_unmatched:
        for file, unmatched_fields in files_with_unmatched.items():
            print(f"\n{file}:")
            for field, value in unmatched_fields.items():
                print(f"  - {field}: {value}")
    else:
        print("\nAll fields were matched in all files!")

if __name__ == "__main__":
    # Create test JSON files (if they don't exist)
    test_files = {
        "hosp1.json": {
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
        },
        "hosp2.json": {
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
        },
        "hosp3.json": {
            "facility_id": "south_valley",
            "facility_name": "South Valley Medical",
            "geo": {"latitude": 64.1500, "longitude": -21.9500},
            "total_beds": 180,
            "open_beds": 50,
            "critical_care_beds": 8,
            "emergency_trauma_unit": True,
            "child_care_unit": True,
            "personnel": {
                "physicians": 40,
                "nursing_staff": 100,
                "medical_specialties": ["orthopedics", "oncology"]
            },
            "last_modified": "2026-02-25T09:00:00Z"
        },
        "hosp4.json": {
            "medical_center_id": "east_river",
            "center_name": "East River Healthcare",
            "gps": {"lat": 64.1600, "long": -21.9600},
            "bed_count": 220,
            "free_beds": 60,
            "intensive_care_beds": 12,
            "trauma_facility": True,
            "pediatric_ward": False,
            "employees": {
                "doctors": 45,
                "nurses": 110,
                "expertise": ["neurology", "cardiovascular"]
            },
            "update_time": "2026-02-25T10:00:00Z"
        }
    }

#     # Save test files
#     for filename, data in test_files.items():
#         with open(filename, "w") as f:
#             json.dump(data, f, indent=2)

    # Run the mapping for all files
    main()
