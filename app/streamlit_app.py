import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import easyocr
from PIL import Image
from pdf2image import convert_from_bytes
import re
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="BC Health Helper",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size:2.5rem; font-weight:700; color:#146EB4; text-align:center; padding:1rem 0; }
    .disclaimer { background-color:#FFF3CD; border-left:5px solid #FF9900; padding:10px 15px; border-radius:5px; font-size:0.9rem; color:#333; margin:10px 0; }
    .pathway-box { padding:20px; border-radius:10px; font-size:1.3rem; font-weight:700; text-align:center; margin:10px 0; }
    .normal { background-color:#EAFAF1; border-left:5px solid #27AE60; padding:10px; border-radius:5px; margin:5px 0; }
    .high { background-color:#FDECEA; border-left:5px solid #E74C3C; padding:10px; border-radius:5px; margin:5px 0; }
    .low { background-color:#FEF9E7; border-left:5px solid #F39C12; padding:10px; border-radius:5px; margin:5px 0; }
</style>
""", unsafe_allow_html=True)

def show_disclaimer():
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong> Information only. NOT medical advice. Always consult a healthcare provider. Call 911 in emergencies.
    </div>
    """, unsafe_allow_html=True)

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_symptom_model():
    with open('../src/symptom_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['encoder'], data['symptoms']

@st.cache_resource
def load_pathway_model():
    with open('../src/pathway_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['encoder'], data['symptoms'], data['red_flags'], data['disease_map']

@st.cache_resource
def load_forecast():
    return pd.read_csv('../data/processed/bc_wait_forecast.csv', parse_dates=['ds'])

@st.cache_data(ttl=604800)  # Refresh every 7 days
def load_wait_data():
    try:
        quarterly_url = "https://catalogue.data.gov.bc.ca/dataset/7c1bf2a8-96bb-4ad5-888d-a90672eb306e/resource/f294562c-a6fd-4d7f-8f99-c51c91891c67/download/2009_2025-quarterly-surgical_wait_times-final.xlsx"
        interim_url = "https://catalogue.data.gov.bc.ca/dataset/7c1bf2a8-96bb-4ad5-888d-a90672eb306e/resource/0c430fa8-043c-48d8-8e61-ecdab63b9ef3/download/2025_2026-quarterly-surgical_wait_times-q2-interim.xlsx"
        df_quarterly = pd.read_excel(quarterly_url)
        df_interim = pd.read_excel(interim_url)
        df = pd.concat([df_quarterly, df_interim], ignore_index=True)
        df['WAITING'] = pd.to_numeric(df['WAITING'], errors='coerce')
        df['COMPLETED'] = pd.to_numeric(df['COMPLETED'], errors='coerce')
        st.success("✅ Live BC Government data loaded")
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not load live BC data: {e}")
        return pd.DataFrame()

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

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

# ── Care Pathway ──────────────────────────────────────────────────────────────
def get_final_pathway(disease, severity, selected_symptoms, red_flags):
    red_flag_hit = [s for s in selected_symptoms if s in red_flags]
    if red_flag_hit:
        return {'label':'🚨 Call 911 — Emergency','color':'#E74C3C','bg':'#FDECEA',
                'action':f'Red flag symptom detected: {red_flag_hit[0].title()}. Call 911 or go to nearest ER immediately.',
                'source':'BC HealthLink 811 (healthlinkbc.ca)'}
    if severity >= 9:
        return {'label':'🚨 Call 911 — Emergency','color':'#E74C3C','bg':'#FDECEA',
                'action':'Severity 9-10/10. Call 911 or go to nearest ER immediately.',
                'source':'CTAS Level 1'}
    if severity >= 7:
        return {'label':'🏥 Go to Urgent Care Today','color':'#E67E22','bg':'#FEF9E7',
                'action':'Severity 7-8/10. Go to an Urgent & Primary Care Centre (UPCC) or walk-in clinic today.',
                'source':'CTAS Level 2-3'}

    disease_map = {
        'acute bronchitis':'pharmacy','conjunctivitis due to allergy':'pharmacy',
        'seasonal allergies (hay fever)':'pharmacy','eczema':'pharmacy',
        'fungal infection of the hair':'pharmacy','nose disorder':'pharmacy',
        'otitis media':'pharmacy','psoriasis':'pharmacy',
        'pyogenic skin infection':'pharmacy','sebaceous cyst':'pharmacy',
        'sprain or strain':'pharmacy','strep throat':'pharmacy',
        'urinary tract infection':'pharmacy','chronic constipation':'pharmacy',
        'esophagitis':'pharmacy','noninfectious gastroenteritis':'pharmacy',
        'infectious gastroenteritis':'pharmacy','gout':'pharmacy',
        'bursitis':'pharmacy','dental caries':'pharmacy',
        'anxiety':'gp','depression':'gp','marijuana abuse':'gp',
        'injury to the arm':'gp','injury to the leg':'gp','vaginal cyst':'gp',
        'acute bronchiolitis':'gp_tests','acute pancreatitis':'gp_tests',
        'cholecystitis':'gp_tests','concussion':'gp_tests',
        'diverticulitis':'gp_tests','hyperemesis gravidarum':'gp_tests',
        'hypoglycemia':'gp_tests','liver disease':'gp_tests',
        'personality disorder':'gp_tests','pneumonia':'gp_tests',
        'problem during pregnancy':'gp_tests',
        'arthritis of the hip':'specialist','benign prostatic hyperplasia (bph)':'specialist',
        'complex regional pain syndrome':'specialist','developmental disability':'specialist',
        'obstructive sleep apnea (osa)':'specialist','peripheral nerve disorder':'specialist',
        'spinal stenosis':'specialist','spondylosis':'specialist','vulvodynia':'specialist',
        'gastrointestinal hemorrhage':'emergency','sickle cell crisis':'emergency',
        'spontaneous abortion':'emergency',
    }

    specialist_map = {
        'arthritis':'Rheumatologist','spinal':'Orthopedic Surgeon',
        'sleep apnea':'Sleep Specialist','nerve':'Neurologist',
        'prostatic':'Urologist','vulvodynia':'Gynecologist',
        'developmental':'Developmental Pediatrician'
    }

    base = disease_map.get(disease.lower(), 'gp')
    if severity >= 5 and base == 'pharmacy':
        base = 'gp'

    specialist_type = next((v for k,v in specialist_map.items() if k in disease.lower()), 'Specialist')

    details = {
        'emergency':{'label':'🚨 Call 911 — Emergency','color':'#E74C3C','bg':'#FDECEA',
                     'action':'Call 911 or go to nearest emergency room immediately.'},
        'urgent':   {'label':'🏥 Urgent Care Today','color':'#E67E22','bg':'#FEF9E7',
                     'action':'Go to an Urgent & Primary Care Centre (UPCC) or walk-in clinic today.'},
        'specialist':{'label':f'🩺 See GP → {specialist_type}','color':'#8E44AD','bg':'#F5EEF8',
                      'action':f'Book a GP appointment. They will refer you to a {specialist_type}. Do not self-refer.'},
        'gp_tests': {'label':'🧪 See GP + Tests Needed','color':'#2980B9','bg':'#EBF5FB',
                     'action':'Book a GP appointment. Tests (blood work or imaging) will likely be needed.'},
        'gp':       {'label':'👨‍⚕️ Book GP Appointment','color':'#146EB4','bg':'#EBF5FB',
                     'action':'Book an appointment with your family doctor for assessment.'},
        'pharmacy': {'label':'💊 Pharmacy / Self-Care First','color':'#27AE60','bg':'#EAFAF1',
                     'action':'Try over-the-counter treatment as appropriate. Ask your pharmacist what is suitable for your symptoms. See GP if no improvement in 1 week. ⚠️  If you are on prescription medications, talk to a pharmacist to confirm whether the OTC medications may interact with your prescriptions.'},
    }
    d = details.get(base, details['gp'])
    d['source'] = 'BC Government Guidelines + BC HealthLink 811'
    return d

# ── Lab Reference Ranges (Medical Council of Canada) ─────────────────────────
# Source: mcc.ca/examinations-assessments/resources-to-help-with-exam-prep/normal-lab-values/
LAB_REFERENCE = {
    'hemoglobin': {
        'unit':'g/L', 'min_male':135, 'max_male':170, 'min_female':115, 'max_female':155,
        'what':'Protein in red blood cells that carries oxygen',
        'high':'May indicate dehydration or a blood disorder. Discuss with your GP.',
        'low':'May indicate anemia (low iron, B12, or other causes). Discuss with your GP.',
        'fasting':False
    },
    'hematocrit': {
        'unit':'%', 'min_male':40, 'max_male':54, 'min_female':37, 'max_female':47,
        'what':'Percentage of blood made up of red blood cells',
        'high':'May indicate dehydration. Discuss with your GP.',
        'low':'May indicate anemia. Discuss with your GP.',
        'fasting':False
    },
    'wbc': {
        'unit':'x10⁹/L', 'min':4.0, 'max':10.0,
        'what':'White blood cells — your immune system cells that fight infection',
        'high':'May indicate infection, inflammation, or immune response. Discuss with your GP.',
        'low':'May indicate immune suppression or bone marrow issues. Discuss with your GP.',
        'fasting':False
    },
    'platelet': {
        'unit':'x10⁹/L', 'min':150, 'max':400,
        'what':'Tiny cells that help your blood clot when you have a cut or injury',
        'high':'May indicate inflammation or bone marrow issues. Discuss with your GP.',
        'low':'May increase bleeding risk. Discuss with your GP.',
        'fasting':False
    },
    'glucose': {
        'unit':'mmol/L', 'min':3.9, 'max':5.6,
        'what':'Blood sugar level (fasting)',
        'high':'May indicate prediabetes or diabetes. Discuss with your GP.',
        'low':'May indicate low blood sugar (hypoglycemia). Discuss with your GP.',
        'fasting':True
    },
    'hba1c': {
        'unit':'%', 'min':0, 'max':6.4,
        'what':'Average blood sugar over the past 3 months — used to monitor diabetes',
        'high':'May indicate diabetes or prediabetes. Discuss with your GP.',
        'low':'Normal — no action needed.',
        'fasting':False
    },
    'creatinine': {
        'unit':'μmol/L', 'min_male':60, 'max_male':110, 'min_female':45, 'max_female':90,
        'what':'Waste product filtered by your kidneys',
        'high':'May indicate kidney issues. Discuss with your GP.',
        'low':'Usually not concerning. Discuss with your GP.',
        'fasting':False
    },
    'sodium': {
        'unit':'mmol/L', 'min':135, 'max':145,
        'what':'Electrolyte that controls fluid balance in your body',
        'high':'May indicate dehydration. Discuss with your GP.',
        'low':'May indicate fluid imbalance. Discuss with your GP.',
        'fasting':False
    },
    'potassium': {
        'unit':'mmol/L', 'min':3.5, 'max':5.0,
        'what':'Electrolyte important for heart and muscle function',
        'high':'May affect heart rhythm. Discuss with your GP promptly.',
        'low':'May cause muscle weakness or heart issues. Discuss with your GP.',
        'fasting':False
    },
    'cholesterol': {
        'unit':'mmol/L', 'min':0, 'max':5.2,
        'what':'Total fat in blood — high levels increase heart disease risk',
        'high':'Increases heart disease risk. Discuss with your GP about lifestyle changes.',
        'low':'Usually not concerning.',
        'fasting':True
    },
    'hdl': {
        'unit':'mmol/L', 'min_male':1.0, 'max_male':99, 'min_female':1.3, 'max_female':99,
        'what':'Good cholesterol — higher levels protect your heart',
        'high':'Good — higher HDL is protective.',
        'low':'Low HDL increases heart disease risk. Discuss with your GP.',
        'fasting':True
    },
    'ldl': {
        'unit':'mmol/L', 'min':0, 'max':3.4,
        'what':'Bad cholesterol — lower is better for heart health',
        'high':'Increases heart disease risk. Discuss with your GP.',
        'low':'Good — low LDL is protective.',
        'fasting':True
    },
    'triglycerides': {
        'unit':'mmol/L', 'min':0, 'max':2.21,
        'what':'Type of fat in blood — high levels linked to heart disease',
        'high':'Increases heart disease risk. Discuss with your GP.',
        'low':'Usually not concerning.',
        'fasting':True
    },
    'tsh': {
        'unit':'mU/L', 'min':0.3, 'max':5.0,
        'what':'Thyroid Stimulating Hormone — checks how well your thyroid is working',
        'high':'May indicate underactive thyroid (hypothyroidism). Discuss with your GP.',
        'low':'May indicate overactive thyroid (hyperthyroidism). Discuss with your GP.',
        'fasting':False
    },
    'alt': {
        'unit':'U/L', 'min_male':0, 'max_male':50, 'min_female':0, 'max_female':36,
        'what':'Liver enzyme — elevated levels may indicate liver stress',
        'high':'May indicate liver issues. Discuss with your GP.',
        'low':'Usually not concerning.',
        'fasting':False
    },
    'ast': {
        'unit':'U/L', 'min':0, 'max':36,
        'what':'Enzyme found in liver and heart — elevated may indicate organ stress',
        'high':'May indicate liver or heart issues. Discuss with your GP.',
        'low':'Usually not concerning.',
        'fasting':False
    },
    'bilirubin': {
        'unit':'μmol/L', 'min':0, 'max':17,
        'what':'Yellow pigment from broken-down red blood cells — processed by liver',
        'high':'May indicate liver issues or bile duct problems. Discuss with your GP.',
        'low':'Usually not concerning.',
        'fasting':False
    },
    'albumin': {
        'unit':'g/L', 'min':35, 'max':50,
        'what':'Protein made by your liver — reflects nutrition and liver health',
        'high':'Usually not concerning.',
        'low':'May indicate liver disease, kidney disease, or malnutrition. Discuss with your GP.',
        'fasting':False
    },
    'calcium': {
        'unit':'mmol/L', 'min':2.12, 'max':2.62,
        'what':'Mineral important for bones, muscles, and nerves',
        'high':'May indicate parathyroid issues or other conditions. Discuss with your GP.',
        'low':'May indicate vitamin D deficiency or other issues. Discuss with your GP.',
        'fasting':False
    },
    'ferritin': {
        'unit':'μg/L', 'min_male':24, 'max_male':444, 'min_female':15, 'max_female':247,
        'what':'Protein that stores iron in your body',
        'high':'May indicate inflammation or iron overload. Discuss with your GP.',
        'low':'May indicate iron deficiency. Discuss with your GP.',
        'fasting':False
    },
    'vitamin d': {
        'unit':'nmol/L', 'min':50, 'max':125,
        'what':'Vitamin important for bone health and immune system',
        'high':'Very high levels can be toxic. Discuss with your GP.',
        'low':'May indicate vitamin D deficiency — common in BC winters. Discuss with your GP.',
        'fasting':False
    },
    'b12': {
        'unit':'pmol/L', 'min':150, 'max':700,
        'what':'Vitamin important for nerve function and red blood cell production',
        'high':'Usually not concerning.',
        'low':'May cause nerve problems or anemia. Discuss with your GP.',
        'fasting':False
    },
    'inr': {
        'unit':'', 'min':0.8, 'max':1.2,
        'what':'Measures how long it takes your blood to clot',
        'high':'Blood takes longer to clot — bleeding risk. Discuss with your GP promptly.',
        'low':'Blood clots faster than normal. Discuss with your GP.',
        'fasting':False
    },
    'crp': {
        'unit':'mg/L', 'min':0, 'max':5.0,
        'what':'C-Reactive Protein — marker of inflammation in your body',
        'high':'Indicates inflammation — may be from infection, injury, or chronic condition. Discuss with your GP.',
        'low':'Normal — no significant inflammation detected.',
        'fasting':False
    },
    'esr': {
        'unit':'mm/hr', 'min_male':0, 'max_male':15, 'min_female':0, 'max_female':20,
        'what':'Erythrocyte Sedimentation Rate — another marker of inflammation',
        'high':'Indicates inflammation. Discuss with your GP.',
        'low':'Normal.',
        'fasting':False
    },
    'psa': {
        'unit':'μg/L', 'min':0, 'max':4.0,
        'what':'Prostate Specific Antigen — screens for prostate gland issues in men',
        'high':'May indicate prostate issues. Discuss with your GP.',
        'low':'Normal.',
        'fasting':False
    },
}

# ── Lab test explanations for requisition forms ───────────────────────────────
LAB_REQUISITION_INFO = {
    'cbc': {
        'full_name': 'Complete Blood Count',
        'what': 'Measures all cells in your blood — red cells, white cells, and platelets',
        'why': 'Ordered to check for anemia, infection, or blood disorders',
        'fasting': 'No fasting required',
        'tests_included': 'WBC, RBC, Hemoglobin, Hematocrit, Platelets'
    },
    'comprehensive metabolic panel': {
        'full_name': 'Comprehensive Metabolic Panel',
        'what': 'Checks your blood chemistry including kidneys, liver, and electrolytes',
        'why': 'Ordered for general health check or to monitor chronic conditions',
        'fasting': 'Fasting for 8-12 hours required',
        'tests_included': 'Glucose, Creatinine, Sodium, Potassium, ALT, AST, Albumin, Bilirubin'
    },
    'lipid panel': {
        'full_name': 'Lipid Panel',
        'what': 'Measures fats in your blood to assess heart disease risk',
        'why': 'Ordered to check cholesterol levels and heart disease risk',
        'fasting': 'Fasting for 9-12 hours required',
        'tests_included': 'Total Cholesterol, HDL, LDL, Triglycerides'
    },
    'thyroid': {
        'full_name': 'Thyroid Function Test',
        'what': 'Checks how well your thyroid gland is working',
        'why': 'Ordered if you have symptoms of thyroid problems (fatigue, weight changes)',
        'fasting': 'No fasting required',
        'tests_included': 'TSH, T3, T4'
    },
    'hba1c': {
        'full_name': 'Glycated Hemoglobin (HbA1c)',
        'what': 'Shows your average blood sugar over the past 3 months',
        'why': 'Ordered to diagnose or monitor diabetes',
        'fasting': 'No fasting required',
        'tests_included': 'HbA1c percentage'
    },
}

# ── Extract number from OCR text ──────────────────────────────────────────────
def extract_number(text, term):
    pattern = rf'{term}[\s:]*([0-9]+\.?[0-9]*)'
    match = re.search(pattern, text.lower())
    if match:
        return float(match.group(1))
    return None

def interpret_result(term, value, sex='unknown'):
    if term not in LAB_REFERENCE:
        return None
    ref = LAB_REFERENCE[term]

    if 'min_male' in ref:
        if sex == 'male':
            low, high = ref['min_male'], ref['max_male']
        elif sex == 'female':
            low, high = ref['min_female'], ref['max_female']
        else:
            low = ref.get('min_male', ref.get('min', 0))
            high = ref.get('max_female', ref.get('max', 999))
    else:
        low, high = ref.get('min', 0), ref.get('max', 999)

    if value < low:
        status = 'LOW'
        meaning = ref['low']
        color = 'low'
    elif value > high:
        status = 'HIGH'
        meaning = ref['high']
        color = 'high'
    else:
        status = 'NORMAL'
        meaning = 'Within normal range — no action needed.'
        color = 'normal'

    return {
        'status': status,
        'meaning': meaning,
        'color': color,
        'range': f"{low}–{high} {ref['unit']}",
        'what': ref['what'],
        'fasting': ref.get('fasting', False)
    }

# ── Medical term explanations ─────────────────────────────────────────────────
medical_terms = {
    'hemoglobin':'Carries oxygen in red blood cells.',
    'hematocrit':'Percentage of blood made up of red blood cells.',
    'platelet':'Cells that help blood clot.',
    'creatinine':'Kidney waste product.',
    'glucose':'Blood sugar level.',
    'cholesterol':'Fat in blood — high levels increase heart disease risk.',
    'triglycerides':'Blood fat linked to heart disease.',
    'hdl':'Good cholesterol — higher is better.',
    'ldl':'Bad cholesterol — lower is better.',
    'tsh':'Thyroid Stimulating Hormone — checks thyroid function.',
    'wbc':'White blood cells — fight infection.',
    'rbc':'Red blood cells — carry oxygen.',
    'bilirubin':'Liver pigment — high may indicate liver issues.',
    'albumin':'Liver protein — low may indicate liver/kidney problems.',
    'sodium':'Electrolyte controlling fluid balance.',
    'potassium':'Electrolyte for heart and muscles.',
    'calcium':'Mineral for bones and muscles.',
    'ferritin':'Iron storage protein.',
    'vitamin d':'Bone health and immunity.',
    'b12':'Nerve function and red blood cells.',
    'alt':'Liver enzyme.',
    'ast':'Liver/heart enzyme.',
    'gfr':'Kidney filtration rate.',
    'inr':'Blood clotting time.',
    'hba1c':'3-month average blood sugar.',
    'esr':'Inflammation marker.',
    'crp':'Inflammation marker.',
    'psa':'Prostate screening (men).',
    'mcv':'Red blood cell size.',
    'mch':'Hemoglobin per red blood cell.'
}

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("# 🏥 BC Health Helper")
st.sidebar.markdown("*Your healthcare navigation guide*")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🏠 Home","🩺 Symptom Helper","⏱️ Wait Time Predictor",
    "🔬 Lab Form Reader","📊 About & Methods"
])
st.sidebar.markdown("---")
st.sidebar.markdown("**⚠️ Disclaimer**\nInformation only.\nNot medical advice.\nCall 911 in emergencies.")

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-header">🏥 BC Health Helper</div>', unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;font-size:1.1rem'>Helping BC patients navigate healthcare — not replace it</p>", unsafe_allow_html=True)
    st.markdown("---")
    show_disclaimer()
    st.markdown("---")
    st.markdown("### 👋 What would you like help with today?")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("### 🩺 I have symptoms\n**Use: Symptom Helper**\n\nTell us your symptoms and severity. We suggest pharmacy, GP, urgent care, or 911 based on official BC guidelines.\n\n*Select from sidebar →*")
    with c2:
        st.markdown("### ⏱️ I'm waiting for surgery\n**Use: Wait Time Predictor**\n\nSee how long BC patients wait for your procedure and check if your wait is normal.\n\n*Select from sidebar →*")
    with c3:
        st.markdown("### 🔬 I have a lab form\n**Use: Lab Form Reader**\n\nUpload your lab requisition or results. We explain every test and flag anything outside normal range.\n\n*Select from sidebar →*")
    st.markdown("---")
    st.markdown("### 📊 BC Health Crisis — Key Facts")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("BC patients waiting","1.2 Million","for a specialist")
    c2.metric("Average wait time","32.2 weeks","GP to treatment")
    c3.metric("Without family doctor","1 in 4","BC residents — BC Family Doctors, June 2025")
    c4.metric("Wait times vs 1993","+208%","longer today")
    st.caption("📌 Statistics source: Doctors of BC (2025) and Fraser Institute (2025). Updated annually — not live data. Surgical wait times data updated quarterly by BC Government.")
    st.markdown("---")
    st.caption("Built by Sai Pramod Podila | Cornerstone College | 2026")

# ══════════════════════════════════════════════════════════════════════════════
# SYMPTOM HELPER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🩺 Symptom Helper":
    st.markdown("## 🩺 Symptom Helper")
    show_disclaimer()
    st.markdown("---")

    st.markdown("### 🚨 Step 1 — Do you have any of these RIGHT NOW?")
    st.error("These are BC HealthLink 811 emergency symptoms. If yes — call 911 immediately.")

    red_flags_display = {
        'Chest pain or pressure':'sharp chest pain',
        'Trouble breathing':'shortness of breath',
        'Weakness/numbness on one side of body':'weakness',
        'Loss of consciousness':'loss of consciousness',
        'Heavy or uncontrolled bleeding':'heavy bleeding',
        'Vomiting blood':'vomiting blood',
        'Sudden severe headache':'sudden severe headache',
        'Seizure':'seizure'
    }

    has_red_flag = False
    rc1,rc2 = st.columns(2)
    for i,(display,_) in enumerate(red_flags_display.items()):
        col = rc1 if i < 4 else rc2
        if col.checkbox(f"🔴 {display}", key=f"rf_{i}"):
            has_red_flag = True

    if has_red_flag:
        st.error("🚨 **Call 911 immediately.** Do not wait. This is a medical emergency.")
        st.stop()

    st.markdown("---")
    st.markdown("### 📊 Step 2 — How severe are your symptoms? (1-10)")
    st.write("1 = Mild | 5 = Affecting daily life | 10 = Unbearable")
    severity = st.slider("Severity:", 1, 10, 3,
                         help="Based on Canadian Triage Acuity Scale (CTAS)")

    if severity >= 9:
        st.error("🚨 Severity 9-10 — Please call 911 or go to Emergency immediately.")
        st.stop()
    elif severity >= 7:
        st.warning("⚠️ Severity 7-8 — Consider going to an Urgent & Primary Care Centre (UPCC) or walk-in clinic today.")

    st.markdown("---")
    st.markdown("### 🩺 Step 3 — Select your symptoms")

    try:
        model, le, symptom_list = load_symptom_model()
        pathway_model, le_pathway, pw_symptoms, red_flags, disease_map = load_pathway_model()

        common_symptoms = [
            'headache','fever','cough','nausea','vomiting',
            'back pain','sharp chest pain','shortness of breath','dizziness',
            'insomnia','sharp abdominal pain','burning abdominal pain',
            'sore throat','nasal congestion','skin swelling',
            'joint pain','leg pain','arm pain','knee pain',
            'frequent urination','blood in urine','depression',
            'anxiety and nervousness','chills','fatigue'
        ]

        available = [s for s in common_symptoms if s in symptom_list]
        selected_symptoms = []
        c1,c2,c3 = st.columns(3)
        chunks = [available[i::3] for i in range(3)]
        for col,chunk in zip([c1,c2,c3], chunks):
            with col:
                for s in chunk:
                    if st.checkbox(s.title(), key=s):
                        selected_symptoms.append(s)

        other = st.text_input("Any other symptom?")
        if other and other.lower() in symptom_list:
            selected_symptoms.append(other.lower())
            st.success(f"✅ Added: {other}")

        if 0 < len(selected_symptoms) < 3:
            st.info(f"💡 {len(selected_symptoms)} symptom(s) selected. More symptoms = more accurate suggestion.")

        st.markdown("---")
        if st.button("🔍 Get My Care Recommendation", type="primary"):
            if not selected_symptoms:
                st.warning("Please select at least one symptom.")
            else:
                input_vector = pd.DataFrame([{s: 1 if s in selected_symptoms else 0 for s in pw_symptoms}])
                pathway_proba = pathway_model.predict_proba(input_vector)[0]
                confidence = round(pathway_proba.max() * 100, 1)

                disease_input = pd.DataFrame([{s: 1 if s in selected_symptoms else 0 for s in symptom_list}])
                disease_pred = model.predict(disease_input)[0]
                disease = le.inverse_transform([disease_pred])[0]

                result = get_final_pathway(disease, severity, selected_symptoms, red_flags)

                st.markdown("---")
                st.markdown("## ✅ Your Recommendation")
                st.markdown(f"""
                <div class="pathway-box" style="background-color:{result['bg']};border:2px solid {result['color']};color:{result['color']}">
                {result['label']}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"**What to do:** {result['action']}")
                st.markdown(f"*Source: {result['source']}*")
                st.markdown(f"*Model confidence: {confidence}%*")
                if confidence < 80:
                    st.warning(f"⚠️ Model confidence is {confidence}% — this recommendation may not be accurate. If you have more symptoms, please select them for a better result. If these are your only symptoms, call 811 (HealthLink BC) free 24/7 to speak with a registered nurse.")
                elif confidence >= 80:
                    st.info(f"ℹ️ Model confidence is {confidence}%. Please note this tool is for navigation only — if your symptoms worsen at any time, call 811 or go to your nearest emergency room.")

                if severity <= 3 and 'pharmacy' in result['label'].lower():
                    st.info("💡 **BC Tip:** Call **811 (HealthLink BC)** free 24/7 to speak with a registered nurse before going anywhere.")

                st.markdown("---")
                st.markdown("#### 📄 Pre-visit Summary")
                summary = f"""BC Health Helper — Pre-visit Summary

Symptoms reported : {', '.join([s.title() for s in selected_symptoms])}
Severity          : {severity}/10
Recommended care  : {result['label']}
What to do        : {result['action']}
Source            : {result['source']}

💡 BC HealthLink 811 — Free 24/7 health advice: Call 8-1-1
⚠️ Information only. Not medical advice. Call 911 in emergencies.
"""
                st.text(summary)
                st.download_button("📄 Download Summary", data=summary,
                                   file_name="bc_health_previsit_summary.txt")
    except Exception as e:
        st.error(f"Error: {e}")

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

        df_proc = df_wait[
            (df_wait['HOSPITAL_NAME'] == 'All Facilities') &
            (df_wait['HEALTH_AUTHORITY'] == 'All Health Authorities') &
            (~df_wait['PROCEDURE_GROUP'].isin(['All Procedures','All Other Procedures']))
        ].dropna(subset=['PERCENTILE_COMP_50TH'])

        # ── Section 1: How long will I wait ──────────────────────────────────
        st.markdown("### 🔍 How long will I wait for my procedure?")
        st.info("""
            📊 **BC Specialist Wait Times**

            BC does not publicly report specialist wait times by specialty.

            According to the Fraser Institute 2025 report, BC patients wait:
            - **15.3 weeks** — GP referral to specialist consultation
            - **12.4 weeks** — specialist consultation to treatment  
            - **32.2 weeks total** — GP referral to treatment

            *Source: Moir & Barua (2025). Waiting Your Turn: Wait Times for Health Care in Canada. Fraser Institute.*
            """)
        st.write("Select your procedure to see current BC wait times.")

        procedures = sorted(df_proc['PROCEDURE_GROUP'].unique().tolist())
        selected_proc = st.selectbox("Select your procedure:", procedures,
                                     help="Ask your GP for the exact procedure name if unsure.")

        if selected_proc:
            df_p = df_proc[df_proc['PROCEDURE_GROUP'] == selected_proc].sort_values('DATE')
            latest = df_p.iloc[-1]
            c1,c2,c3 = st.columns(3)
            c1.metric("Most patients wait", f"{latest['PERCENTILE_COMP_50TH']:.0f} weeks","median")
            c2.metric("Longest waits", f"{latest['PERCENTILE_COMP_90TH']:.0f} weeks","1 in 10 patients")
            c3.metric("Data as of", f"{latest['FISCAL_YEAR']}", latest['QUARTER'])

            fig = px.line(df_p, x='DATE', y='PERCENTILE_COMP_50TH',
                          title=f'Wait Time History — {selected_proc}',
                          labels={'PERCENTILE_COMP_50TH':'Weeks waiting','DATE':'Year'})
            fig.update_traces(line_color='#146EB4', line_width=2)
            fig.update_layout(height=300)
            st.plotly_chart(fig, width="stretch")

        st.markdown("---")

        # ── Section 2: Am I waiting too long ─────────────────────────────────
        st.markdown("### ⏰ Am I waiting too long?")
        st.write("Enter how long you have been waiting to see if it's within normal range.")

        weeks_waiting = st.number_input("How many weeks have you been waiting?",
                                        min_value=0, max_value=200, value=0, step=1)

        if weeks_waiting > 0 and selected_proc:
            median_wait = latest['PERCENTILE_COMP_50TH']
            p90_wait = latest['PERCENTILE_COMP_90TH']

            if weeks_waiting <= median_wait:
                st.success(f"✅ Your wait of **{weeks_waiting} weeks** is within the normal range for {selected_proc} (median: {median_wait:.0f} weeks).")
            elif weeks_waiting <= p90_wait:
                st.warning(f"⚠️ Your wait of **{weeks_waiting} weeks** is longer than average but within the top 10% range ({p90_wait:.0f} weeks).")
                st.markdown("""
                **What you can do:**
                - Call your specialist's office directly and ask your position on the waitlist
                - Ask your GP if your condition has worsened — they can update your referral priority
                - Contact BC Patient Care Quality Office: **1-866-952-2448**
                """)
            else:
                st.error(f"🚨 Your wait of **{weeks_waiting} weeks** is longer than 90% of BC patients for this procedure.")
                st.markdown("""
                **Action steps:**
                - Call your specialist's office immediately to confirm you're still on the waitlist
                - Book an urgent appointment with your GP to review your referral
                - Contact BC Patient Care Quality Office: **1-866-952-2448**
                - Consider asking your GP about referral to a different health authority
                """)

        st.markdown("---")

        # ── Section 3: Which region is fastest ───────────────────────────────
        st.markdown("### 🏥 Which BC region has the shortest wait for my procedure?")
        st.write("Compare current wait times across BC health authorities for your selected procedure.")

        if selected_proc:
            df_region = df_wait[
                (df_wait['HOSPITAL_NAME'] == 'All Facilities') &
                (df_wait['HEALTH_AUTHORITY'] != 'All Health Authorities') &
                (df_wait['PROCEDURE_GROUP'] == selected_proc)
            ].dropna(subset=['PERCENTILE_COMP_50TH','DATE'])

            if not df_region.empty:
                latest_by_region = df_region.sort_values('DATE').groupby('HEALTH_AUTHORITY').last().reset_index()
                latest_by_region = latest_by_region.sort_values('PERCENTILE_COMP_50TH')

                fig2 = px.bar(latest_by_region,
                              x='HEALTH_AUTHORITY', y='PERCENTILE_COMP_50TH',
                              title=f'Current Wait Times by Region — {selected_proc}',
                              labels={'PERCENTILE_COMP_50TH':'Weeks waiting','HEALTH_AUTHORITY':'Region'},
                              color='PERCENTILE_COMP_50TH',
                              color_continuous_scale=['#27AE60','#E74C3C'])
                fig2.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig2, width="stretch")

                fastest = latest_by_region.iloc[0]
                st.info(f"💡 **{fastest['HEALTH_AUTHORITY']}** currently has the shortest wait for {selected_proc} at **{fastest['PERCENTILE_COMP_50TH']:.0f} weeks**. Ask your GP if a cross-region referral is possible.")
            else:
                st.info("Regional data not available for this procedure.")

        st.markdown("---")

        # ── Section 4: Is wait getting better or worse ────────────────────────
        st.markdown("### 📈 Is the BC wait getting better or worse?")
        st.write("BC-wide trend for all procedures — last 3 years + AI forecast.")

        df_bc = df_wait[
            (df_wait['HOSPITAL_NAME'] == 'All Facilities') &
            (df_wait['HEALTH_AUTHORITY'] == 'All Health Authorities') &
            (df_wait['PROCEDURE_GROUP'] == 'All Procedures')
        ].dropna(subset=['DATE','PERCENTILE_COMP_50TH']).sort_values('DATE')
        df_bc_recent = df_bc[df_bc['DATE'] >= '2018-01-01']

        latest_forecast = forecast.tail(4)['yhat'].mean()
        current = df_bc_recent['PERCENTILE_COMP_50TH'].iloc[-1]

        if latest_forecast > current:
            st.warning(f"📈 Our AI forecast suggests BC wait times may **increase slightly** to ~{latest_forecast:.1f} weeks in the next year.")
        else:
            st.success(f"📉 Our AI forecast suggests BC wait times may **improve slightly** to ~{latest_forecast:.1f} weeks in the next year.")

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_bc_recent['DATE'], y=df_bc_recent['PERCENTILE_COMP_50TH'],
                          name='Actual wait time', line=dict(color='#00BFFF',width=3)))
        fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                                  name='AI Forecast', line=dict(color='#FF9900',width=2,dash='dash')))
        fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'],
                                  fill=None, mode='lines', line_color='rgba(255,153,0,0.1)', showlegend=False))
        fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],
                                  fill='tonexty', mode='lines', line_color='rgba(255,153,0,0.1)',
                                  name='Possible range'))
        fig3.update_layout(xaxis_title='Year', yaxis_title='Weeks waiting', height=320,
                           legend=dict(orientation='h', y=-0.3))
        st.plotly_chart(fig3, width="stretch")
        st.caption("📌 Forecast uses Prophet ML model trained on BC government data 2009–2025. COVID spike (2020) visible in historical data.")

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
    🔒 <strong>Privacy:</strong> Remove your name, DOB and health number before uploading. The app only needs test names and values. For full PIPEDA compliance run locally from GitHub.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="disclaimer">
    ⚠️ <strong>Important:</strong> Reference ranges are from the LifeLabs BC Burnaby Reference Laboratory (March 2024). Your lab report includes its own reference ranges — always use those first. These results are for information only. Always discuss your results with your GP. Call 811 (HealthLink BC) free 24/7 if you have concerns.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    form_type = st.radio("What are you uploading?", [
        "📋 Lab Requisition Form (form my GP gave me to take to LifeLabs)",
        "📊 Lab Results (results I received back from LifeLabs)"
    ])

    sex = st.selectbox("Your biological sex (for accurate reference ranges):",
                       ["Unknown", "Male", "Female"]).lower()

    st.markdown("### 📸 Option 1 — Take a Photo")
    st.info("💡 Cover your name, DOB and health number with your finger before capturing.")
    camera_image = st.camera_input("Take a photo of your lab form")

    st.markdown("### 📁 Option 2 — Upload a File")
    uploaded_files = st.file_uploader(
        "Upload your lab form(s) — JPG, PNG, or PDF:",
        type=['jpg','jpeg','png','pdf'],
        accept_multiple_files=True
    )

    all_images = []

    if camera_image:
        all_images.append(Image.open(camera_image))

    if uploaded_files:
        for uploaded in uploaded_files:
            if uploaded.type == 'application/pdf':
                try:
                    images = convert_from_bytes(uploaded.read())
                    all_images.extend(images)
                    st.success(f"✅ PDF converted: {uploaded.name} ({len(images)} page(s))")
                except Exception as e:
                    st.error(f"PDF error: {e}")
            else:
                all_images.append(Image.open(uploaded))

    if all_images:
        st.write(f"**{len(all_images)} page(s) loaded**")

        if st.button("🔍 Read & Explain", type="primary"):
            with st.spinner("Reading your form(s)... this may take 30-60 seconds."):
                reader = load_ocr()
                full_text = ''
                for i, img in enumerate(all_images):
                    img.save(f'/tmp/lab_page_{i}.png')
                    results = reader.readtext(f'/tmp/lab_page_{i}.png')
                    extracted = [t for (_,t,c) in results if c > 0.3]
                    full_text += ' '.join(extracted) + ' '

            full_text_lower = full_text.lower()

            if '📋 Lab Requisition' in form_type:
                # ── Requisition mode ──────────────────────────────────────────
                st.markdown("### 📋 Tests on Your Requisition Form")
                st.write("Here's what each test means and what to expect:")

                found_panels = []
                for key, info in LAB_REQUISITION_INFO.items():
                    if key in full_text_lower:
                        found_panels.append((key, info))

                found_terms = {term: exp for term, exp in medical_terms.items()
                              if term in full_text_lower}

                if found_panels:
                    for key, info in found_panels:
                        with st.expander(f"🧪 {info['full_name']}"):
                            st.markdown(f"**What it is:** {info['what']}")
                            st.markdown(f"**Why your doctor ordered it:** {info['why']}")
                            st.markdown(f"**Tests included:** {info['tests_included']}")
                            if 'fasting' in info['fasting'].lower() and 'no' not in info['fasting'].lower():
                                st.warning(f"⚠️ **{info['fasting']}**")
                            else:
                                st.success(f"✅ {info['fasting']}")

                if found_terms:
                    st.markdown("### Individual Tests Found:")
                    for term, exp in found_terms.items():
                        ref = LAB_REFERENCE.get(term, {})
                        fasting = ref.get('fasting', False)
                        with st.expander(f"📋 {term.upper()}"):
                            st.write(exp)
                            if fasting:
                                st.warning("⚠️ Fasting required for this test (8-12 hours, water only)")

                if not found_panels and not found_terms:
                    st.warning("No recognized tests found. Try uploading a clearer photo.")

                st.info("💡 Bring this form to any LifeLabs location. Show your BC Services Card.")

            else:
                # ── Results mode ──────────────────────────────────────────────
                st.markdown("### 📊 Your Lab Results")

                results_found = []
                for term, ref in LAB_REFERENCE.items():
                    value = extract_number(full_text_lower, term)
                    if value is not None:
                        interpretation = interpret_result(term, value, sex)
                        if interpretation:
                            results_found.append((term, value, interpretation))

                if results_found:
                    normal_count = sum(1 for _,_,r in results_found if r['status'] == 'NORMAL')
                    high_count = sum(1 for _,_,r in results_found if r['status'] == 'HIGH')
                    low_count = sum(1 for _,_,r in results_found if r['status'] == 'LOW')

                    c1,c2,c3 = st.columns(3)
                    c1.metric("✅ Normal", normal_count)
                    c2.metric("⬆️ High", high_count)
                    c3.metric("⬇️ Low", low_count)

                    if high_count > 0 or low_count > 0:
                        st.error(f"⚠️ {high_count + low_count} result(s) are outside normal range. Please discuss with your GP.")

                    st.markdown("### Detailed Results:")
                    for term, value, interp in results_found:
                        status_emoji = "✅" if interp['status'] == 'NORMAL' else ("⬆️" if interp['status'] == 'HIGH' else "⬇️")
                        with st.expander(f"{status_emoji} {term.upper()} — {value} {LAB_REFERENCE[term]['unit']} — {interp['status']}"):
                            st.markdown(f"**What it measures:** {interp['what']}")
                            st.markdown(f"**Normal range:** {interp['range']}")
                            st.markdown(f"**Your value:** {value} {LAB_REFERENCE[term]['unit']}")
                            st.markdown(f"**What this means:** {interp['meaning']}")
                            if interp['status'] != 'NORMAL':
                                st.warning("⚠️ Please discuss this result with your GP.")

                    st.markdown("---")
                    st.markdown("""
                    **📞 Next Steps:**
                    - ✅ Normal results — mention to your GP at next visit
                    - ⬆️⬇️ Flagged results — book a GP appointment to discuss
                    - 🆘 Urgent concerns — call **811 (HealthLink BC)** free 24/7
                    - 📋 **Reference ranges source:** LifeLabs BC Burnaby Reference Laboratory (March 2024) and Medical Council of Canada (MCC) 2024
                    """)

                    summary_lines = ["BC Health Helper — Lab Results Summary\n"]
                    for term, value, interp in results_found:
                        summary_lines.append(f"{term.upper()}: {value} {LAB_REFERENCE[term]['unit']} — {interp['status']}")
                        if interp['status'] != 'NORMAL':
                            summary_lines.append(f"  → {interp['meaning']}")
                    summary_lines.append("\n⚠️ Information only. Always discuss with your GP.")
                    summary_lines.append("Source: LifeLabs BC Burnaby Reference Laboratory (March 2024) and Medical Council of Canada (MCC) 2024")

                    st.download_button("📄 Download Results Summary",
                                      data='\n'.join(summary_lines),
                                      file_name="bc_health_lab_results.txt")
                else:
                    found_terms2 = {term: exp for term, exp in medical_terms.items()
                                   if term in full_text_lower}
                    if found_terms2:
                        st.markdown("### Tests found on your form:")
                        for term, exp in found_terms2.items():
                            with st.expander(f"📋 {term.upper()}"):
                                st.write(exp)
                        st.info("💡 Could not extract exact values from this image. Try uploading a clearer photo for full results interpretation.")
                    else:
                        st.warning("No lab terms found. Try uploading a clearer, better-lit photo.")
    else:
        st.info("👆 Upload your lab form above to get started.")
        st.markdown("""
        **Tips for best results:**
        - ✅ Good lighting — no shadows
        - ✅ Form flat and straight
        - ✅ All text clearly visible
        - ✅ Works with LifeLabs, DynaLab, BC hospital forms
        - ✅ Supports JPG, PNG, and PDF
        - ✅ Multiple pages supported
        """)

# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 About & Methods":
    st.markdown("## 📊 About & Methods")
    st.markdown("---")
    st.markdown("""
    ### What is BC Health Helper?
    An AI-powered healthcare navigation tool for BC patients.
    Does NOT diagnose — helps patients navigate to the right level of care.

    ### Components & Performance
    | Component | Method | Dataset | Performance |
    |---|---|---|---|
    | Symptom Helper | Logistic Regression | Disease-Symptom 246k | 93.16% accuracy |
    | Care Pathway Classifier | Logistic Regression | Disease-Symptom 246k | 95.89% accuracy |
    | Wait Time Predictor | Prophet Time Series | BC Gov 2009–2025 | MAE 0.32 weeks |
    | Lab Form Reader | EasyOCR (CV) + NLP | MTSamples 4,999 | 19/30 terms |
    | Patient Clustering | K-Means K=14 | Disease-Symptom 246k | Silhouette 0.119 |

    ### Official Sources Used
    | Source | Used For |
    |---|---|
    | BC HealthLink 811 — healthlinkbc.ca (Jan 2025) | Emergency red flag symptoms |
    | BC Government Pharmacy Services — gov.bc.ca (Feb 2026) | 21 official minor ailments |
    | Canadian Triage Acuity Scale (CTAS) | Severity 1-10 routing |
    | LifeLabs BC Burnaby Reference Laboratory (March 2024) + Medical Council of Canada (MCC) 2024 | Lab reference ranges |

    ### Privacy & PIPEDA
- 🌐 Cloud version: Remove personal identifiers before uploading lab forms
    - 💻 Local version: Full PIPEDA compliance — data never leaves your device
    - No patient data stored or transmitted in either version
    - Production deployment would use Canadian cloud infrastructure for full PIPEDA compliance
    - Clone from GitHub to run locally: github.com/pramodkumarpodila-source/bc_health_helper

    ### Limitations
    - Not a diagnostic tool — navigation only
    - Severity is self-reported — no vital signs
    - Lab reader works best on printed forms
    - Reference ranges are general — your lab's ranges may differ slightly
    - Clinical validation recommended before real deployment

    ---
    **Built by:** Sai Pramod Podila | Cornerstone College | 2026
    **GitHub:** github.com/pramodkumarpodila-source/bc_health_helper
    """)
    show_disclaimer()