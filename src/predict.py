"""
predict.py
----------
Loads the trained model and classifies NEW patient biopsy samples.

This is what a clinician-facing tool would call: given the 30 measured
features from a new fine-needle-aspirate biopsy, predict Benign/Malignant
and report the model's confidence.

Run:
    python src/predict.py
"""

import os
import joblib
import pandas as pd

from cancer_classifier import MODEL_PATH, BASE_DIR

FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst",
    "symmetry_worst", "fractal_dimension_worst",
]


def load_model(path: str = MODEL_PATH):
    return joblib.load(path)


def predict_diagnosis(model, samples_df: pd.DataFrame) -> pd.DataFrame:
    missing = set(FEATURE_NAMES) - set(samples_df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    X = samples_df[FEATURE_NAMES]
    probabilities = model.predict_proba(X)[:, 1]  # P(Malignant)

    result = samples_df.copy()
    result["malignant_probability"] = probabilities.round(4)
    result["prediction"] = ["Malignant" if p >= 0.5 else "Benign" for p in probabilities]
    result["confidence"] = [max(p, 1 - p) for p in probabilities]
    return result


if __name__ == "__main__":
    model = load_model()

    # Two real rows taken from the dataset itself (one confirmed Malignant,
    # one confirmed Benign) as a sanity check that the model gets them right.
    # In production, replace this with real new biopsy measurements.
    sample_patients = pd.DataFrame(
        [
            {  # actual diagnosis: Malignant (id 842302)
                "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8, "area_mean": 1001,
                "smoothness_mean": 0.1184, "compactness_mean": 0.2776, "concavity_mean": 0.3001,
                "concave points_mean": 0.1471, "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871,
                "radius_se": 1.095, "texture_se": 0.9053, "perimeter_se": 8.589, "area_se": 153.4,
                "smoothness_se": 0.006399, "compactness_se": 0.04904, "concavity_se": 0.05373,
                "concave points_se": 0.01587, "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
                "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.6, "area_worst": 2019,
                "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119,
                "concave points_worst": 0.2654, "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.1189,
            },
            {  # actual diagnosis: Benign (id 8510426)
                "radius_mean": 13.54, "texture_mean": 14.36, "perimeter_mean": 87.46, "area_mean": 566.3,
                "smoothness_mean": 0.09779, "compactness_mean": 0.08129, "concavity_mean": 0.06664,
                "concave points_mean": 0.04781, "symmetry_mean": 0.1885, "fractal_dimension_mean": 0.05766,
                "radius_se": 0.2699, "texture_se": 0.7886, "perimeter_se": 2.058, "area_se": 23.56,
                "smoothness_se": 0.008462, "compactness_se": 0.0146, "concavity_se": 0.02387,
                "concave points_se": 0.01315, "symmetry_se": 0.0198, "fractal_dimension_se": 0.0023,
                "radius_worst": 15.11, "texture_worst": 19.26, "perimeter_worst": 99.7, "area_worst": 711.2,
                "smoothness_worst": 0.144, "compactness_worst": 0.1773, "concavity_worst": 0.239,
                "concave points_worst": 0.1288, "symmetry_worst": 0.2977, "fractal_dimension_worst": 0.07259,
            },
        ]
    )

    results = predict_diagnosis(model, sample_patients)

    print("\n=== Diagnosis Predictions ===\n")
    for i, row in results.iterrows():
        print(f"Patient {i + 1}:")
        print(f"  Prediction:              {row['prediction']}")
        print(f"  Malignant probability:   {row['malignant_probability']:.1%}")
        print(f"  Model confidence:        {row['confidence']:.1%}")
        print()

    out_path = os.path.join(BASE_DIR, "outputs", "sample_predictions.csv")
    results.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
