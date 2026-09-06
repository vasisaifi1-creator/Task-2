# 🩺 Breast Cancer Diagnosis Classifier

A machine learning project that predicts whether a breast tumor is
**Benign** (not cancer) or **Malignant** (cancer), based on real
measurements taken from a biopsy.

---

## 🔗 GitHub Repository

https://github.com/vasisaifi1-creator/Task-1

---

## 📌 What This Project Does

Doctors take a small tissue sample from a breast lump (called a
**fine-needle-aspirate biopsy**) and measure things like the size, shape,
and texture of the cells under a microscope. This project trains a machine
learning model to look at those measurements and predict:

> **Is this tumor Benign or Malignant?**

---

## 📊 About the Data

- **569 real patients** — this is NOT made-up data
- **30 measurements per patient** (radius, texture, smoothness, etc.)
- Comes from the well-known **Wisconsin Breast Cancer Dataset**
  (used by researchers and data scientists worldwide)
- 357 patients were Benign, 212 were Malignant

---

## 📁 What's Inside This Project

```
breast_cancer/
│
├── data/                     → the real patient dataset (CSV file)
├── models/                   → the trained model (saved after training)
├── outputs/                  → charts and results get saved here
│
├── src/
│   ├── cancer_classifier.py  → trains the model
│   ├── predict.py            → tests the model on 2 built-in example patients
│   └── predict_from_csv.py   → check YOUR OWN patient data
│
├── requirements.txt          → list of Python libraries needed
└── README.md                 → this file
```

---

## 🚀 How to Run This Project (3 Simple Steps)

### Step 1: Install the required libraries
```bash
pip install -r requirements.txt
```

### Step 2: Train the model
```bash
python src/cancer_classifier.py
```
This teaches the model using the 569 real patients, then saves it.
It also creates charts in the `outputs/` folder so you can see how well it worked.

### Step 3: Test it
```bash
python src/predict.py
```
This checks 2 example patients and tells you if they're Benign or Malignant.

---

## 🔍 How to Check a New Patient

### Option A — Generate a blank form to fill in
```bash
python src/predict_from_csv.py --make-template
```
This creates `new_patients_template.csv`. Open it, replace the example
numbers with your real patient's measurements, and save it.

### Option B — Check everyone in your file
```bash
python src/predict_from_csv.py --file new_patients_template.csv
```

### Option C — Check just ONE specific patient by their ID
```bash
python src/predict_from_csv.py --file new_patients_template.csv --id PATIENT_001
```

---

## 📈 How Good Is This Model?

| Metric | Score | What It Means |
|---|---|---|
| **Accuracy** | 97% | Gets the diagnosis right 97 times out of 100 |
| **Precision** | 100% | When it says "Malignant," it's ALWAYS right |
| **Recall** | 93% | Catches 93 out of every 100 real cancer cases |

### ⚠️ Why "Recall" Matters Most Here
- Missing a real cancer (saying "Benign" when it's actually cancer) is
  **dangerous** — the patient thinks they're fine when they're not.
- A false alarm (saying "Malignant" when it's actually fine) just means
  an extra check-up — annoying, but not dangerous.

So this project is built to care more about **not missing cancer cases**
than about being perfectly accurate overall.

---

## 🧠 What the Charts in `outputs/` Show You

| File | What It Shows |
|---|---|
| `eda_overview.png` | How Benign vs Malignant patients differ |
| `confusion_matrix.png` | How many predictions were right/wrong |
| `roc_curve.png` | Overall model performance (higher = better) |
| `precision_recall_tradeoff.png` | The trade-off between catching cancer vs false alarms |
| `feature_importance.png` | Which measurements matter most for the prediction |

---

## ⚠️ Important Disclaimer

This is a **student/learning project**, not a real medical tool. It should
never be used to make actual medical decisions. Always consult a real
doctor for real diagnoses.

---

## 🙌 Credits

- Dataset: Wisconsin Breast Cancer Diagnostic Dataset (UCI Machine
  Learning Repository / Kaggle)
- Built using Python, scikit-learn, pandas, and matplotlib


🙋 Author
MOHD VASI Saifi — https://github.com/vasisaifi1-creator