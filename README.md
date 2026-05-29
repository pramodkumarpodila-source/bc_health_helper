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

## 📊 Datasets Used

| Dataset | Source | Size |
|---|---|---|
| BC Surgical Wait Times | [catalogue.data.gov.bc.ca](https://catalogue.data.gov.bc.ca/dataset/bc-surgical-wait-times) | 2009–2026 |
| Disease-Symptom Dataset | [Kaggle — dhivyeshrk](https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset) | 246,945 records |
| MTSamples Medical Transcriptions | [Kaggle — tboyle10](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions) | 4,999 notes |

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