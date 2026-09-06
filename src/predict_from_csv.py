"""
predict_from_csv.py
--------------------
Reads a CSV of new patients (with an 'id' column + the 30 required
features) and predicts Benign/Malignant for all of them - or just one
specific patient by ID.

CSV FORMAT REQUIRED:
Your CSV must have these 31 columns (id + 30 features), with these EXACT
column names (case-sensitive, including the space in "concave points"):

    id, radius_mean, texture_mean, perimeter_mean, area_mean,
    smoothness_mean, compactness_mean, concavity_mean, concave points_mean,
    symmetry_mean, fractal_dimension_mean, radius_se, texture_se,
    perimeter_se, area_se, smoothness_se, compactness_se, concavity_se,
    concave points_se, symmetry_se, fractal_dimension_se, radius_worst,
    texture_worst, perimeter_worst, area_worst, smoothness_worst,
    compactness_worst, concavity_worst, concave points_worst,
    symmetry_worst, fractal_dimension_worst

The 'id' column can be anything you want - a patient number, a name, a
record ID - it's just used to look the row up afterward. It is NOT used
by the model to make predictions.

USAGE:
    # Predict for every patient in the file:
    python src/predict_from_csv.py --file new_patients.csv

    # Predict for one specific patient by ID:
    python src/predict_from_csv.py --file new_patients.csv --id 1001

    # Generate a blank template CSV to fill in:
    python src/predict_from_csv.py --make-template
"""

import argparse
import os
import sys

import pandas as pd

from cancer_classifier import MODEL_PATH, BASE_DIR
from predict import load_model, predict_diagnosis, FEATURE_NAMES


def make_template(path: str):
    """Writes a blank CSV with the correct headers and one example row."""
    columns = ["id"] + FEATURE_NAMES
    example_row = {
        "id": "PATIENT_001",
        "radius_mean": 14.2, "texture_mean": 18.5, "perimeter_mean": 92.3, "area_mean": 632.1,
        "smoothness_mean": 0.098, "compactness_mean": 0.105, "concavity_mean": 0.089,
        "concave points_mean": 0.048, "symmetry_mean": 0.178, "fractal_dimension_mean": 0.061,
        "radius_se": 0.35, "texture_se": 0.92, "perimeter_se": 2.4, "area_se": 28.5,
        "smoothness_se": 0.0065, "compactness_se": 0.021, "concavity_se": 0.028,
        "concave points_se": 0.012, "symmetry_se": 0.019, "fractal_dimension_se": 0.0032,
        "radius_worst": 16.8, "texture_worst": 24.1, "perimeter_worst": 108.9, "area_worst": 863.4,
        "smoothness_worst": 0.135, "compactness_worst": 0.29, "concavity_worst": 0.31,
        "concave points_worst": 0.14, "symmetry_worst": 0.28, "fractal_dimension_worst": 0.081,
    }
    df = pd.DataFrame([example_row], columns=columns)
    df.to_csv(path, index=False)
    print(f"Template written to {path}")
    print("Open it, replace the example row with your real patient(s), add more rows as needed, then run:")
    print(f"    python src/predict_from_csv.py --file {path}")


def main():
    parser = argparse.ArgumentParser(description="Predict Benign/Malignant from a CSV of patients.")
    parser.add_argument("--file", type=str, help="Path to your patients CSV file.")
    parser.add_argument("--id", type=str, default=None, help="Look up only this patient ID (optional).")
    parser.add_argument("--make-template", action="store_true", help="Write a blank template CSV and exit.")
    args = parser.parse_args()

    if args.make_template:
        template_path = os.path.join(BASE_DIR, "new_patients_template.csv")
        make_template(template_path)
        return

    if not args.file:
        print("Error: --file is required (or use --make-template to generate a starter CSV).")
        sys.exit(1)

    if not os.path.exists(args.file):
        print(f"Error: file not found: {args.file}")
        sys.exit(1)

    df = pd.read_csv(args.file)

    if "id" not in df.columns:
        print("Error: your CSV must have an 'id' column to identify each patient.")
        sys.exit(1)

    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        print(f"Error: your CSV is missing these required columns: {sorted(missing)}")
        print("Run 'python src/predict_from_csv.py --make-template' to see the exact format needed.")
        sys.exit(1)

    model = load_model()
    results = predict_diagnosis(model, df)

    if args.id is not None:
        # Look up one specific patient
        match = results[results["id"].astype(str) == str(args.id)]
        if match.empty:
            print(f"No patient found with id '{args.id}' in {args.file}.")
            print(f"Available ids: {results['id'].astype(str).tolist()}")
            sys.exit(1)
        row = match.iloc[0]
        print(f"\n=== Patient {row['id']} ===")
        print(f"  Prediction:              {row['prediction']}")
        print(f"  Malignant probability:   {row['malignant_probability']:.1%}")
        print(f"  Model confidence:        {row['confidence']:.1%}")
    else:
        # Show every patient in the file
        print(f"\n=== Predictions for all {len(results)} patients in {args.file} ===\n")
        for _, row in results.iterrows():
            print(f"Patient {row['id']}: {row['prediction']}  "
                  f"(malignant probability: {row['malignant_probability']:.1%}, "
                  f"confidence: {row['confidence']:.1%})")

    out_path = os.path.join(BASE_DIR, "outputs", "csv_predictions.csv")
    results.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
