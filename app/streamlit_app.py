import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import easyocr
import spacy
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BC Health Helper",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #146EB4;
        text-align: center;
        padding: 1rem 0;
    }
    .disclaimer {
        background-color: #FFF3CD;
        border-left: 5px solid #FF9900;
        padding: 10px 15px;
        border-radius: 5px;
        font-size: 0.9rem;
        color: #333;
    }
    .pathway-box {
        background-color: #F0F7FF;
        border-left: 5px solid #146EB4;
        padding: 15px;
        border-radius: 5px;
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
def show_disclaimer():
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> This tool provides information only and is NOT medical advice.
    Always consult a qualified healthcare provider. Call 911 in emergencies.
    </div>
    """, unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_symptom_model():
    with open('../src/symptom_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['encoder'], data['symptoms']

@st.cache_resource
def load_forecast():
    return pd.read_csv('../data/processed/bc_wait_forecast.csv', parse_dates=['ds'])

@st.cache_resource
def load_wait_data():
    df = pd.read_excel('../data/raw/2009_2025-quarterly-surgical_wait_times-final.xlsx')
    df['WAITING'] = pd.to_numeric(df['WAITING'], errors='coerce')
    df['COMPLETED'] = pd.to_numeric(df['COMPLETED'], errors='coerce')
    return df

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def load_nlp():
    return spacy.load('en_core_web_sm')

# ── Care Pathway ──────────────────────────────────────────────────────────────
def get_care_pathway(disease):
    urgent    = ['heart attack','stroke','sepsis','meningitis','pulmonary embolism','anaphylaxis']
    specialist= ['cancer','diabetes','arthritis','epilepsy','multiple sclerosis','parkinson']
    gp_tests  = ['pneumonia','tuberculosis','hepatitis','kidney','liver','thyroid']
    pharmacy  = ['cold','flu','allergy','migraine','acid reflux','conjunctivitis',
                 'seasonal allergies','hay fever','eczema','psoriasis']

    specialist_map = {
        'cancer'            : 'Oncologist',
        'diabetes'          : 'Endocrinologist',
        'arthritis'         : 'Rheumatologist',
        'epilepsy'          : 'Neurologist',
        'multiple sclerosis': 'Neurologist',
        'parkinson'         : 'Neurologist',
        'spinal'            : 'Orthopedic Surgeon',
        'knee'              : 'Orthopedic Surgeon',
        'hip'               : 'Orthopedic Surgeon',
        'heart'             : 'Cardiologist',
        'cardiac'           : 'Cardiologist',
        'skin'              : 'Dermatologist',
        'eye'               : 'Ophthalmologist',
        'ear'               : 'ENT Specialist',
        'mental'            : 'Psychiatrist',
        'depression'        : 'Psychiatrist',
        'anxiety'           : 'Psychiatrist',
        'lung'              : 'Pulmonologist',
        'kidney'            : 'Nephrologist',
        'liver'             : 'Gastroenterologist',
    }

    d = disease.lower()

    if any(u in d for u in urgent):
        return '⚠️ Urgent Care', '#E74C3C', 'Call 911 or go to nearest emergency room immediately.', None
    elif any(s in d for s in specialist):
        specialist_type = next((v for k, v in specialist_map.items() if k in d), 'Specialist')
        return '🩺 GP → Specialist', '#8E44AD', f'See your GP first — they will refer you to a {specialist_type}. Do not self-refer.', specialist_type
    elif any(g in d for g in gp_tests):
        return '🧪 GP + Tests', '#2980B9', 'Book a GP appointment. Lab tests or imaging will likely be needed before any referral.', None
    elif any(p in d for p in pharmacy):
        return '⚫ Pharmacy', '#27AE60', 'A pharmacist can recommend over-the-counter treatment. See a GP if symptoms persist over 1 week.', None
    else:
        return '👨‍⚕️ See GP', '#146EB4', 'Book an appointment with your family doctor. They will assess and refer if needed.', None

# ── Medical Terms ─────────────────────────────────────────────────────────────
medical_terms = {
    'hemoglobin'  : 'protein in red blood cells that carries oxygen',
    'hematocrit'  : 'percentage of red blood cells in blood',
    'platelet'    : 'tiny cells that help blood clot',
    'creatinine'  : 'waste product filtered by kidneys — high levels may indicate kidney issues',
    'glucose'     : 'blood sugar level',
    'cholesterol' : 'fatty substance in blood — high levels increase heart disease risk',
    'triglycerides': 'type of fat in blood — high levels linked to heart disease',
    'hdl'         : 'good cholesterol — higher is better',
    'ldl'         : 'bad cholesterol — lower is better',
    'tsh'         : 'thyroid stimulating hormone — checks thyroid function',
    'wbc'         : 'white blood cells — fight infection',
    'rbc'         : 'red blood cells — carry oxygen',
    'bilirubin'   : 'yellow pigment from red blood cell breakdown — high levels may indicate liver issues',
    'albumin'     : 'protein made by liver — low levels may indicate liver or kidney disease',
    'sodium'      : 'electrolyte that controls fluid balance',
    'potassium'   : 'electrolyte important for heart and muscle function',
    'calcium'     : 'mineral important for bones and muscle function',
    'ferritin'    : 'protein that stores iron — indicates iron levels',
    'vitamin d'   : 'vitamin important for bone health and immune function',
    'b12'         : 'vitamin important for nerve function and red blood cell production',
    'alt'         : 'liver enzyme — high levels may indicate liver damage',
    'ast'         : 'enzyme found in liver and heart — high levels may indicate damage',
    'gfr'         : 'measures how well kidneys filter blood',
    'inr'         : 'measures blood clotting time',
    'hba1c'       : 'average blood sugar over 3 months — used to monitor diabetes',
    'esr'         : 'indicates inflammation in the body',
    'crp'         : 'c-reactive protein — marker of inflammation',
    'psa'         : 'prostate specific antigen — screens for prostate issues',
    'mcv'         : 'average size of red blood cells',
    'mch'         : 'amount of hemoglobin per red blood cell'
}

# ── Parse date helper ─────────────────────────────────────────────────────────
def parse_date(row):
    quarter_map = {'Q1':'07','Q2':'10','Q3':'01','Q4':'04'}
    try:
        year = int(row['FISCAL_YEAR'].split('/')[0])
        month = quarter_map.get(row['QUARTER'],'07')
        if month in ['01','04']:
            year += 1
        return pd.Timestamp(f'{year}-{month}-01')
    except:
        return pd.NaT

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("# 🏥 BC Health Helper")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home",
    "🩺 Symptom Helper",
    "⏱️ Wait Time Predictor",
    "🔬 Lab Form Reader",
    "📊 About & Methods"
])
st.sidebar.markdown("---")
st.sidebar.markdown("""
**⚠️ Disclaimer**  
Information only.  
Not medical advice.  
Call 911 in emergencies.
""")

# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-header">🏥 BC Health Helper</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray'>An AI-powered healthcare navigation tool for BC patients</p>", unsafe_allow_html=True)
    st.markdown("---")
    show_disclaimer()
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🩺 Symptom Helper")
        st.write("Enter your symptoms and get a suggested care pathway — pharmacy, GP, or urgent care. Includes likely specialist and pre-visit summary.")

    with col2:
        st.markdown("### ⏱️ Wait Time Predictor")
        st.write("See BC surgical wait times by procedure and health authority. Includes forecast for upcoming quarters.")

    with col3:
        st.markdown("### 🔬 Lab Form Reader")
        st.write("Upload a photo of your lab requisition and get plain English explanations of every test.")

    st.markdown("---")
    st.markdown("### 📊 BC Health Crisis — Key Facts")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients Waiting", "1.2M", "for specialists in BC")
    c2.metric("Median Wait", "32.2 weeks", "GP to treatment")
    c3.metric("No Family Doctor", "900K", "BC residents")
    c4.metric("Longer than 1993", "208%", "wait time increase")

    st.markdown("---")
    st.markdown("*Use the sidebar on the left to navigate between tools.*")

# ══════════════════════════════════════════════════════════════════════════════
# SYMPTOM HELPER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🩺 Symptom Helper":
    st.markdown("## 🩺 Symptom Helper")
    show_disclaimer()
    st.markdown("---")

    try:
        model, le, symptom_list = load_symptom_model()

        st.markdown("### Select your symptoms")
        st.info("Select all symptoms you are currently experiencing. The model will suggest the most appropriate level of care.")

        col1, col2, col3 = st.columns(3)
        selected_symptoms = []

        common_symptoms = [
            'headache', 'fever', 'cough', 'nausea', 'vomiting',
            'back pain', 'sharp chest pain', 'shortness of breath', 'dizziness',
            'insomnia', 'sharp abdominal pain', 'burning abdominal pain',
            'sore throat', 'nasal congestion', 'skin swelling',
            'joint pain', 'leg pain', 'arm pain', 'knee pain',
            'frequent urination', 'blood in urine', 'depression',
            'anxiety and nervousness', 'chills', 'fatigue'
        ]

        available = [s for s in common_symptoms if s in symptom_list]
        chunks = [available[i::3] for i in range(3)]

        for col, chunk in zip([col1, col2, col3], chunks):
            with col:
                for symptom in chunk:
                    if st.checkbox(symptom.title(), key=symptom):
                        selected_symptoms.append(symptom)

        st.markdown("### Additional symptoms (optional)")
        free_text = st.text_input("Type any other symptom:")
        if free_text and free_text.lower() in symptom_list:
            selected_symptoms.append(free_text.lower())

        if st.button("🔍 Get Care Pathway", type="primary"):
            if len(selected_symptoms) == 0:
                st.warning("Please select at least one symptom.")
            else:
                input_vector = pd.DataFrame([{s: 1 if s in selected_symptoms else 0
                                              for s in symptom_list}])
                prediction = model.predict(input_vector)[0]
                disease = le.inverse_transform([prediction])[0]
                pathway, color, advice, specialist_type = get_care_pathway(disease)

                st.markdown("---")
                st.markdown("### Results")
                st.markdown(f"""
                <div class="pathway-box" style="border-left-color: {color};">
                {pathway}
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"**Suggested condition:** {disease.title()}")
                st.markdown(f"**Advice:** {advice}")

                if specialist_type:
                    st.info(f"💡 Likely specialist: **{specialist_type}** — your GP will confirm and refer you. Check the Wait Time Predictor to see expected wait times.")

                st.markdown("---")
                st.markdown("#### 📄 Pre-visit Summary")
                summary = f"""BC Health Helper — Pre-visit Summary
                
Symptoms reported  : {', '.join(selected_symptoms)}
Suggested condition: {disease.title()}
Recommended care   : {pathway}
Advice             : {advice}
Specialist         : {specialist_type if specialist_type else 'N/A — GP will assess'}

⚠️ Disclaimer: Information only. Not medical advice. Always consult a healthcare provider.
"""
                st.code(summary)
                st.download_button("📄 Download Pre-visit Summary",
                    data=summary,
                    file_name="previsit_summary.txt")

    except Exception as e:
        st.error(f"Error loading model: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# WAIT TIME PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⏱️ Wait Time Predictor":
    st.markdown("## ⏱️ BC Surgical Wait Time Predictor")
    show_disclaimer()
    st.markdown("---")

    try:
        df_wait = load_wait_data()
        forecast = load_forecast()
        df_wait['DATE'] = df_wait.apply(parse_date, axis=1)

        # ── BC Wide Trend ─────────────────────────────────────────────────────
        st.markdown("### 📈 BC-wide Wait Time Trend + Forecast")
        st.info("🔵 Blue = actual recorded wait times (weeks) | 🟠 Orange dashed = predicted future wait times | Shaded = possible best to worst case range")

        df_bc = df_wait[
            (df_wait['HOSPITAL_NAME'] == 'All Facilities') &
            (df_wait['HEALTH_AUTHORITY'] == 'All Health Authorities') &
            (df_wait['PROCEDURE_GROUP'] == 'All Procedures')
        ].dropna(subset=['DATE','PERCENTILE_COMP_50TH']).sort_values('DATE')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_bc['DATE'], y=df_bc['PERCENTILE_COMP_50TH'],
                                 name='Actual Wait Time (weeks)', line=dict(color='#146EB4', width=2)))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                 name='Predicted Wait Time', line=dict(color='#FF9900', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                                 fill=None, mode='lines', line_color='rgba(255,153,0,0.1)', showlegend=False))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                                 fill='tonexty', mode='lines', line_color='rgba(255,153,0,0.1)',
                                 name='Possible range (best to worst case)'))
        fig.update_layout(
            title='BC Surgical Wait Times 2009–2026 — All Procedures',
            xaxis_title='Year', 
            yaxis_title='Median Wait Time (weeks)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # ── By Procedure ──────────────────────────────────────────────────────
        st.markdown("### 🔍 Search Wait Time by Procedure")
        st.write("Select your procedure to see how long patients are currently waiting in BC.")

        df_proc = df_wait[
            (df_wait['HOSPITAL_NAME'] == 'All Facilities') &
            (df_wait['HEALTH_AUTHORITY'] == 'All Health Authorities') &
            (df_wait['PROCEDURE_GROUP'] != 'All Procedures') &
            (df_wait['PROCEDURE_GROUP'] != 'All Other Procedures')
        ].dropna(subset=['PERCENTILE_COMP_50TH'])

        procedures = sorted(df_proc['PROCEDURE_GROUP'].unique().tolist())
        selected_proc = st.selectbox("Select procedure:", procedures)

        if selected_proc:
            df_p = df_proc[df_proc['PROCEDURE_GROUP'] == selected_proc].sort_values('DATE')
            latest = df_p.iloc[-1]

            c1, c2 = st.columns(2)
            c1.metric("Current Median Wait", f"{latest['PERCENTILE_COMP_50TH']:.1f} weeks",
                      f"as of {latest['FISCAL_YEAR']} {latest['QUARTER']}")
            c2.metric("90th Percentile Wait", f"{latest['PERCENTILE_COMP_90TH']:.1f} weeks",
                      "worst 10% of patients wait this long")

            fig2 = px.line(df_p, x='DATE', y='PERCENTILE_COMP_50TH',
                           title=f'Wait Time Trend — {selected_proc}',
                           labels={'PERCENTILE_COMP_50TH':'Median Wait (weeks)','DATE':'Year'})
            fig2.update_traces(line_color='#146EB4', line_width=2)
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("---")

        # ── By Health Authority ───────────────────────────────────────────────
        st.markdown("### 🏥 Wait Times by Health Authority")
        st.write("Compare wait times across BC regions.")

        df_ha = df_wait[
            (df_wait['HOSPITAL_NAME'] == 'All Facilities') &
            (df_wait['HEALTH_AUTHORITY'] != 'All Health Authorities') &
            (df_wait['PROCEDURE_GROUP'] == 'All Procedures')
        ].dropna(subset=['DATE','PERCENTILE_COMP_50TH'])

        selected_ha = st.multiselect(
            "Select health authorities to compare:",
            df_ha['HEALTH_AUTHORITY'].unique().tolist(),
            default=df_ha['HEALTH_AUTHORITY'].unique().tolist()[:3]
        )

        if selected_ha:
            df_filtered = df_ha[df_ha['HEALTH_AUTHORITY'].isin(selected_ha)]
            fig3 = px.line(df_filtered, x='DATE', y='PERCENTILE_COMP_50TH',
                           color='HEALTH_AUTHORITY',
                           title='Median Surgical Wait by Health Authority',
                           labels={'PERCENTILE_COMP_50TH':'Median Wait (weeks)',
                                   'DATE':'Year',
                                   'HEALTH_AUTHORITY':'Health Authority'})
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
        st.write(str(e))

# ══════════════════════════════════════════════════════════════════════════════
# LAB FORM READER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Lab Form Reader":
    st.markdown("## 🔬 Lab Form Reader")
    show_disclaimer()
    st.markdown("""
    <div class="disclaimer">
    🔒 <strong>Privacy:</strong> Your image is processed locally on your device.
    Nothing is uploaded to any server. PIPEDA compliant.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.info("📷 Take a clear photo of your printed lab requisition form and upload it below. Works best with LifeLabs, DynaLab, and BC hospital forms.")

    uploaded = st.file_uploader("Upload your lab form photo (JPG or PNG):",
                                type=['jpg','jpeg','png'])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption='Uploaded Lab Form', width=400)

        if st.button("🔍 Read & Explain Lab Form", type="primary"):
            with st.spinner("Reading your lab form using AI... this may take 30 seconds."):
                image.save('/tmp/lab_upload.png')
                reader = load_ocr()
                results = reader.readtext('/tmp/lab_upload.png')
                extracted = [text for (_, text, conf) in results if conf > 0.3]
                full_text = ' '.join(extracted).lower()

            st.markdown("### 📝 Text extracted from your form:")
            st.code(' | '.join(extracted))

            found_terms = {term: exp for term, exp in medical_terms.items()
                          if term in full_text}

            st.markdown(f"### 💊 Plain English Explanations ({len(found_terms)} tests found)")
            if found_terms:
                for term, explanation in found_terms.items():
                    with st.expander(f"📋 {term.upper()}"):
                        st.write(explanation)
            else:
                st.warning("No recognized lab terms found. Try uploading a clearer image.")
    else:
        st.info("👆 Upload a photo of your BC lab requisition form to get started.")
        st.markdown("""
        **Tips for best results:**
        - Take photo in good lighting
        - Keep the form flat and straight
        - Make sure all text is visible and in focus
        - Supported: LifeLabs, DynaLab, BC hospital lab forms
        """)

# ══════════════════════════════════════════════════════════════════════════════
# ABOUT PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 About & Methods":
    st.markdown("## 📊 About & Methods")
    st.markdown("---")

    st.markdown("""
    ### What is BC Health Helper?
    BC Health Helper is an AI-powered healthcare navigation tool built for BC patients.
    It does NOT diagnose — it helps patients understand what level of care they likely need,
    provides information about BC surgical wait times by procedure, and explains lab results
    in plain English.

    ### Components & Methods
    | Component | Method | Dataset | Performance |
    |---|---|---|---|
    | Symptom Helper | Logistic Regression | Disease-Symptom 246k records | **93.16% accuracy** |
    | Wait Time Predictor | Prophet Time Series | BC Gov Wait Times 2009–2025 | **MAE 0.32 weeks** |
    | Lab Form Reader | EasyOCR (CV) + spaCy (NLP) | MTSamples 4,999 notes | **19/30 terms identified** |
    | Patient Clustering | K-Means (K=14) | Disease-Symptom 246k records | **Silhouette 0.119** |

    ### Why Logistic Regression Won
    Logistic Regression (93.16%) outperformed Random Forest (92.04%) and XGBoost (91.91%)
    because all symptom features are binary (0/1) — linear models handle this very efficiently.

    ### Privacy & PIPEDA Compliance
    - All processing runs locally on your device
    - No patient data is stored or transmitted to any server
    - Lab form images are processed in memory only and immediately discarded
    - Fully compliant with BC PIPEDA privacy requirements

    ### Limitations
    - Not a diagnostic tool — for care navigation only
    - Symptom dataset is global, not BC-specific
    - Lab reader works best on printed (not handwritten) forms
    - Wait time forecast is BC-wide aggregate — individual hospital times may vary
    - Specialist mapping is approximate — always confirm with your GP

    ---
    """)
    show_disclaimer()