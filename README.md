# 🏥 BC Health Helper

An AI-powered healthcare navigation tool for BC patients — built at Cornerstone College (2026).

> ⚠️ Information only. Not medical advice. Call 911 in emergencies.

> 💻 Runs locally due to large dataset files (190MB). See setup instructions below.

---

## 🎯 What It Does

BC has 1.2M patients waiting for specialists, 1 in 4 residents without a family doctor, and median wait times of 32.2 weeks. BC Health Helper helps patients navigate the healthcare system — not replace it.

| Feature | What it does |
|---|---|
| 🩺 Symptom Helper | Select symptoms + severity (1-10) → get care pathway based on official BC guidelines |
| ⏱️ Wait Time Predictor | See current BC surgical wait times by procedure and region + AI forecast |
| 🔬 Lab Form Reader | Upload lab requisition or results → plain English explanations + flagged results |
| 📊 Patient Clustering | K-Means clustering of patient symptom profiles (unsupervised ML) |

---

## 🤖 ML Components & Performance

| Component | Method | Dataset | Performance |
|---|---|---|---|
| Symptom Helper | Logistic Regression (best of 3) | Disease-Symptom 246k | **93.16% accuracy** |
| Care Pathway Classifier | Logistic Regression | Disease-Symptom 246k | **95.89% accuracy** |
| Wait Time Predictor | Prophet Time Series | BC Gov Wait Times 2009–2026 | **MAE 0.32 weeks** |
| Lab Form Reader | EasyOCR (CV) + spaCy NLP | MTSamples 4,999 notes | 19/30 terms identified |
| Patient Clustering | K-Means (K=14) | Disease-Symptom 246k | Silhouette 0.119 |

### Model Comparison — Symptom Helper
| Model | Test Accuracy | CV Score | Train Time |
|---|---|---|---|
| **Logistic Regression** ✅ | **93.16%** | 93.31% ± 0.17% | 3.7s |
| Random Forest | 92.04% | 92.37% ± 0.10% | 1.7s |
| XGBoost | 91.91% | 92.25% ± 0.16% | 67.4s |

---

## 📋 Official Sources Used

| Source | Used For |
|---|---|
| BC HealthLink 811 — healthlinkbc.ca (Jan 2025) | Emergency red flag symptoms |
| BC Government Pharmacy Services — gov.bc.ca (Feb 2026) | 21 official minor ailments → Pharmacy pathway |
| Canadian Triage Acuity Scale (CTAS) | Severity 1-10 routing |
| LifeLabs BC Burnaby Reference Laboratory (March 2024) | Lab reference ranges |
| Fraser Institute (2025) | BC specialist wait time statistics |
| Doctors of BC (2025) | BC health crisis statistics |
| BC Family Doctors Report (June 2025) | Family doctor statistics |

---

## 📁 Project Structure

```
bc_health_helper/
├── notebooks/
│   ├── 01_symptom_helper.ipynb          # ML classifier — 93.16% accuracy
│   ├── 02_bc_wait_times.ipynb           # Prophet forecasting — MAE 0.32 weeks
│   ├── 03_clustering.ipynb              # K-Means — K=14 clusters
│   ├── 04_lab_reader.ipynb              # EasyOCR CV + spaCy NLP
│   └── 05_care_pathway_classifier.ipynb # BC guideline-based pathway — 95.89% accuracy
├── app/
│   └── streamlit_app.py                 # Full Streamlit dashboard
├── src/
│   ├── symptom_model.pkl                # Trained LR symptom model
│   └── pathway_model.pkl                # Trained pathway classifier
├── data/
│   ├── raw/                             # BC Gov wait times, Disease-Symptom, MTSamples
│   └── processed/                       # Prophet forecast output
└── README.md
```

---

## ⚠️ Dataset Setup Required

The raw datasets are not included in this repo (too large for GitHub). Download them separately and place in `data/raw/`:

| Dataset | Download Link | Save As |
|---|---|---|
| Disease-Symptom | [Kaggle — dhivyeshrk](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset) | `data/raw/Final_Augmented_dataset_Diseases_and_Symptoms.csv` |
| BC Surgical Wait Times | [catalogue.data.gov.bc.ca](https://catalogue.data.gov.bc.ca/dataset/bc-surgical-wait-times) | `data/raw/` |
| MTSamples | [Kaggle — tboyle10](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) | `data/raw/mtsamples.csv` |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/pramodkumarpodila-source/bc_health_helper.git
cd bc_health_helper

# Create virtual environment
python3 -m venv bc_env
source bc_env/bin/activate

# Install dependencies
pip install pandas numpy matplotlib seaborn plotly jupyter ipykernel scikit-learn streamlit
pip install openpyxl xgboost easyocr prophet spacy pdf2image
python -m spacy download en_core_web_sm

# Install poppler for PDF support (Mac)
brew install poppler

# Run the app
cd app
streamlit run streamlit_app.py
```

---

## 🔒 Privacy & PIPEDA Compliance

- All processing runs locally on the user's device
- No patient data is stored or transmitted to any server
- Lab form images processed in memory only and immediately discarded
- Compliant with Canadian PIPEDA privacy requirements

---

## ⚠️ Limitations

- Not a diagnostic tool — navigation only
- Symptom dataset is global, not BC-specific
- Lab reader works best on printed (not handwritten) forms
- Lab reference ranges vary by lab — always use your own report's ranges
- Severity is self-reported — no vital signs used
- Surgical wait times only — BC does not publish specialist outpatient wait times
- Clinical validation by a medical professional recommended before real deployment

---

## 🔮 Future Work

- Connect to BC Data Catalogue API for live quarterly data updates
- Expand to full 773-disease model with cloud compute (AWS/GCP)
- Add SHAP explainability for symptom predictions
- LSTM/ARIMA comparison for wait time forecasting
- BC-specific clinical validation on real patient data
- GPU acceleration for EasyOCR
- Specialist outpatient wait times when BC makes data public
- Multilingual support for BC's Punjabi, Mandarin and Cantonese speaking populations

---

## 👨‍💻 Built By

**Sai Pramod Podila**
Data Science Diploma — Cornerstone College
2026
GitHub: [pramodkumarpodila-source](https://github.com/pramodkumarpodila-source)

---

> ⚠️ Disclaimer: Information only. Not medical advice. Always consult a healthcare provider. Call 911 in emergencies.