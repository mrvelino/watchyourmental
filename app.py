
# ============================================================
# SISTEM PAKAR DIAGNOSIS PENYAKIT MENTAL
# Forward Chaining + Certainty Factor + Hybrid Dataset
# Kelompok 3 - Sistem Pakar 2025
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from enum import Enum
import json

# Set page config
st.set_page_config(
    page_title="WATCHYOURMENTAL!",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS CUSTOM UNTUK UI BAHASA INDONESIA
# ============================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .cluster-a { color: #e74c3c; font-weight: bold; }
    .cluster-b { color: #f39c12; font-weight: bold; }
    .cluster-c { color: #27ae60; font-weight: bold; }
    .cf-high { color: #e74c3c; font-weight: bold; font-size: 1.2rem; }
    .cf-mid { color: #f39c12; font-weight: bold; font-size: 1.2rem; }
    .cf-low { color: #27ae60; font-weight: bold; font-size: 1.2rem; }
    .rule-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f4e79;
        padding: 10px;
        margin: 5px 0;
        border-radius: 0 5px 5px 0;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f4e79;
        color: white;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #2c6fad;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA STRUCTURES
# ============================================================

class Cluster(Enum):
    A = "A: Aneh/Eksentrik"
    B = "B: Dramatis/Emosional"
    C = "C: Cemas/Takut"

@dataclass
class Rule:
    id: str
    cluster: Cluster
    disorder: str
    disorder_code: str
    conditions: List[Dict[str, Any]]
    base_cf: float
    description: str
    page_ref: str

@dataclass
class DiagnosisResult:
    disorder: str
    disorder_code: str
    cluster: str
    cf: float
    triggered_rules: List[Dict]
    recommendation: str

# ============================================================
# KNOWLEDGE BASE - 50 RULES
# ============================================================

def build_knowledge_base():
    rules = []

    # CLUSTER A: Paranoid PD (R01-R05)
    rules.extend([
        Rule("R01", Cluster.A, "Paranoid Personality Disorder", "Paranoid PD",
             [{"dataset_map": "paranoid_distrust"}, {"dataset_map": "paranoid_hypervigilance"}], 0.75,
             "Curiga berlebihan terhadap motif orang lain & menginterpretasi tindakan sebagai ancaman", "Hal. 113"),
        Rule("R02", Cluster.A, "Paranoid Personality Disorder", "Paranoid PD",
             [{"dataset_map": "paranoid_distrust"}, {"dataset_map": "paranoid_grudge"}], 0.70,
             "Enggan berbagi informasi pribadi & takut informasi digunakan untuk menyerang", "Hal. 113"),
        Rule("R03", Cluster.A, "Paranoid Personality Disorder", "Paranoid PD",
             [{"dataset_map": "paranoid_grudge"}], 0.65,
             "Menyimpan dendam lama & tidak mau memaafkan kesalahan kecil", "Hal. 113"),
        Rule("R04", Cluster.A, "Paranoid Personality Disorder", "Paranoid PD",
             [{"dataset_map": "paranoid_counterattack"}], 0.70,
             "Mudah marah & menyerang balik saat merasa dihina", "Hal. 113"),
        Rule("R05", Cluster.A, "Paranoid Personality Disorder", "Paranoid PD",
             [{"dataset_map": "paranoid_family_distrust"}], 0.72,
             "Meragukan loyalitas orang terdekat termasuk keluarga", "Hal. 113"),
    ])

    # CLUSTER A: Schizoid PD (R06-R10)
    rules.extend([
        Rule("R06", Cluster.A, "Schizoid Personality Disorder", "Schizoid PD",
             [{"dataset_map": "schizoid_avoid_social"}], 0.75,
             "Menghindari hubungan sosial & lebih memilih aktivitas soliter", "Hal. 113"),
        Rule("R07", Cluster.A, "Schizoid Personality Disorder", "Schizoid PD",
             [{"dataset_map": "schizoid_no_desire"}], 0.72,
             "Tidak ada keinginan memulai/mempertahankan pertemanan", "Hal. 113"),
        Rule("R08", Cluster.A, "Schizoid Personality Disorder", "Schizoid PD",
             [{"dataset_map": "schizoid_flat_affect"}], 0.70,
             "Jarang ekspresi emosi & tidak terpengaruh pujian/kritik", "Hal. 113"),
        Rule("R09", Cluster.A, "Schizoid Personality Disorder", "Schizoid PD",
             [{"dataset_map": "schizoid_no_sexual"}], 0.68,
             "Tidak tertarik pengalaman seksual dengan orang lain", "Hal. 113"),
        Rule("R10", Cluster.A, "Schizoid Personality Disorder", "Schizoid PD",
             [{"dataset_map": "schizoid_no_attention"}], 0.68,
             "Tidak membutuhkan perhatian/penerimaan dari orang lain", "Hal. 113"),
    ])

    # CLUSTER A: Schizotypal PD (R11-R15)
    rules.extend([
        Rule("R11", Cluster.A, "Schizotypal Personality Disorder", "Schizotypal PD",
             [{"dataset_map": "schizotypal_ideas_ref"}], 0.75,
             "Percaya kejadian tidak berkaitan punya makna khusus (ideas of reference)", "Hal. 113"),
        Rule("R12", Cluster.A, "Schizotypal Personality Disorder", "Schizotypal PD",
             [{"dataset_map": "schizotypal_magical"}], 0.72,
             "Percaya punya kekuatan magis atau kemampuan kontrol pikiran", "Hal. 113-114"),
        Rule("R13", Cluster.A, "Schizotypal Personality Disorder", "Schizotypal PD",
             [{"dataset_map": "schizotypal_speech"}], 0.70,
             "Pola bicara tidak biasa (melantur/inkoheren)", "Hal. 114"),
        Rule("R14", Cluster.A, "Schizotypal Personality Disorder", "Schizotypal PD",
             [{"dataset_map": "schizotypal_social_anxiety"}], 0.72,
             "Kecemasan sosial ekstrem terutama dengan orang asing", "Hal. 114"),
        Rule("R15", Cluster.A, "Schizotypal Personality Disorder", "Schizotypal PD",
             [{"dataset_map": "schizotypal_perception"}], 0.70,
             "Persepsi tidak biasa (merasakan kehadiran/mendengar nama dipanggil)", "Hal. 114"),
    ])

    # CLUSTER B: Antisocial PD (R16-R20)
    rules.extend([
        Rule("R16", Cluster.B, "Antisocial Personality Disorder", "Antisocial PD",
             [{"dataset_map": "antisocial_rule_break"}], 0.75,
             "Melanggar norma sosial/hukum & tidak peduli hak orang lain", "Hal. 115"),
        Rule("R17", Cluster.B, "Antisocial Personality Disorder", "Antisocial PD",
             [{"dataset_map": "antisocial_deceit"}], 0.72,
             "Sering berbohong untuk keuntungan pribadi & impulsif", "Hal. 115"),
        Rule("R18", Cluster.B, "Antisocial Personality Disorder", "Antisocial PD",
             [{"dataset_map": "antisocial_aggression"}], 0.72,
             "Agresif & reckless terhadap keselamatan diri/orang lain", "Hal. 115"),
        Rule("R19", Cluster.B, "Antisocial Personality Disorder", "Antisocial PD",
             [{"dataset_map": "antisocial_no_remorse"}], 0.70,
             "Tidak menyesal & menyalahkan korban", "Hal. 115"),
        Rule("R20", Cluster.B, "Antisocial Personality Disorder", "Antisocial PD",
             [{"dataset_map": "antisocial_irresponsible"}], 0.68,
             "Tidak mampu mempertahankan pekerjaan/keuangan", "Hal. 115"),
    ])

    # CLUSTER B: Borderline PD (R21-R25)
    rules.extend([
        Rule("R21", Cluster.B, "Borderline Personality Disorder", "Borderline PD",
             [{"dataset_map": "borderline_unstable_rel"}], 0.70,
             "Hubungan interpersonal tidak stabil (idealisasi → devaluasi)", "Hal. 116"),
        Rule("R22", Cluster.B, "Borderline Personality Disorder", "Borderline PD",
             [{"dataset_map": "borderline_mood_swing"}], 0.65,
             "Perubahan mood drastis & interaksi volatile", "Hal. 116"),
        Rule("R23", Cluster.B, "Borderline Personality Disorder", "Borderline PD",
             [{"dataset_map": "borderline_abandonment"}], 0.70,
             "Takut abandonment ekstrem & perilaku impulsif", "Hal. 116"),
        Rule("R24", Cluster.B, "Borderline Personality Disorder", "Borderline PD",
             [{"dataset_map": "borderline_self_harm"}], 0.80,
             "Self-harm / perilaku suicidal", "Hal. 116"),
        Rule("R25", Cluster.B, "Borderline Personality Disorder", "Borderline PD",
             [{"dataset_map": "borderline_identity"}], 0.72,
             "Citra diri tidak stabil & paranoia/halusinasi transien", "Hal. 116"),
    ])

    # CLUSTER B: Histrionic PD (R26-R30)
    rules.extend([
        Rule("R26", Cluster.B, "Histrionic Personality Disorder", "Histrionic PD",
             [{"dataset_map": "histrionic_attention"}], 0.72,
             "Tidak nyaman bukan pusat perhatian & berperilaku dramatis", "Hal. 117"),
        Rule("R27", Cluster.B, "Histrionic Personality Disorder", "Histrionic PD",
             [{"dataset_map": "histrionic_provocative"}], 0.70,
             "Berperilaku provokatif seksual di situasi tidak tepat", "Hal. 117"),
        Rule("R28", Cluster.B, "Histrionic Personality Disorder", "Histrionic PD",
             [{"dataset_map": "histrionic_suggestible"}], 0.68,
             "Mudah dipengaruhi pendapat orang lain/tren", "Hal. 117"),
        Rule("R29", Cluster.B, "Histrionic Personality Disorder", "Histrionic PD",
             [{"dataset_map": "histrionic_shallow"}], 0.70,
             "Menganggap hubungan biasa lebih intim dari kenyataan", "Hal. 117"),
        Rule("R30", Cluster.B, "Histrionic Personality Disorder", "Histrionic PD",
             [{"dataset_map": "histrionic_appearance"}], 0.68,
             "Menghabiskan waktu berlebihan untuk penampilan fisik", "Hal. 117"),
    ])

    # CLUSTER B: Narcissistic PD (R31-R35)
    rules.extend([
        Rule("R31", Cluster.B, "Narcissistic Personality Disorder", "Narcissistic PD",
             [{"dataset_map": "narcissistic_grandiosity"}], 0.75,
             "Rasa kebesaran diri & membanggakan pencapaian", "Hal. 117"),
        Rule("R32", Cluster.B, "Narcissistic Personality Disorder", "Narcissistic PD",
             [{"dataset_map": "narcissistic_admiration"}], 0.72,
             "Butuh kekaguman berlebihan & marah tidak diistimewakan", "Hal. 117"),
        Rule("R33", Cluster.B, "Narcissistic Personality Disorder", "Narcissistic PD",
             [{"dataset_map": "narcissistic_no_empathy"}], 0.72,
             "Tidak mampu berempati terhadap perasaan/kebutuhan orang", "Hal. 117"),
        Rule("R34", Cluster.B, "Narcissistic Personality Disorder", "Narcissistic PD",
             [{"dataset_map": "narcissistic_exploitative"}], 0.70,
             "Mengeksploitasi hubungan untuk kepentingan pribadi", "Hal. 117"),
        Rule("R35", Cluster.B, "Narcissistic Personality Disorder", "Narcissistic PD",
             [{"dataset_map": "narcissistic_envious"}], 0.70,
             "Iri terhadap orang sukses & percaya semua orang iri padanya", "Hal. 118"),
    ])

    # CLUSTER C: Avoidant PD (R36-R39)
    rules.extend([
        Rule("R36", Cluster.C, "Avoidant Personality Disorder", "Avoidant PD",
             [{"dataset_map": "avoidant_social"}], 0.75,
             "Menghindari situasi sosial karena takut dikritik/ditolak", "Hal. 119"),
        Rule("R37", Cluster.C, "Avoidant Personality Disorder", "Avoidant PD",
             [{"dataset_map": "avoidant_fear_rejection"}], 0.72,
             "Menginginkan hubungan tapi takut tidak disukai", "Hal. 119"),
        Rule("R38", Cluster.C, "Avoidant Personality Disorder", "Avoidant PD",
             [{"dataset_map": "avoidant_inadequacy"}], 0.70,
             "Melebih-lebihkan konsekuensi negatif situasi baru", "Hal. 119"),
        Rule("R39", Cluster.C, "Avoidant Personality Disorder", "Avoidant PD",
             [{"dataset_map": "avoidant_occupational"}], 0.70,
             "Menghindari pekerjaan dengan kontak sosial karena takut evaluasi", "Hal. 119"),
    ])

    # CLUSTER C: Dependent PD (R40-R44)
    rules.extend([
        Rule("R40", Cluster.C, "Dependent Personality Disorder", "Dependent PD",
             [{"dataset_map": "dependent_decisions"}], 0.72,
             "Tidak bisa membuat keputusan tanpa persetujuan/jaminan orang", "Hal. 119"),
        Rule("R41", Cluster.C, "Dependent Personality Disorder", "Dependent PD",
             [{"dataset_map": "dependent_submissive"}], 0.72,
             "Orang lain mengontrol hidupnya & takut ekspresi pendapat", "Hal. 119"),
        Rule("R42", Cluster.C, "Dependent Personality Disorder", "Dependent PD",
             [{"dataset_map": "dependent_abandonment"}], 0.70,
             "Takut ditinggalkan & segera cari pengganti hubungan", "Hal. 119"),
        Rule("R43", Cluster.C, "Dependent Personality Disorder", "Dependent PD",
             [{"dataset_map": "dependent_disagree"}], 0.70,
             "Rela tidak menyenangkan demi dukungan orang lain", "Hal. 120"),
        Rule("R44", Cluster.C, "Dependent Personality Disorder", "Dependent PD",
             [{"dataset_map": "dependent_helpless"}], 0.70,
             "Merasa tidak berdaya & tidak percaya diri sendiri", "Hal. 120"),
    ])

    # CLUSTER C: OCPD (R45-R50)
    rules.extend([
        Rule("R45", Cluster.C, "Obsessive-Compulsive Personality Disorder", "OCPD",
             [{"dataset_map": "ocpd_order"}], 0.72,
             "Terobsesi keteraturan & detail hingga mengabaikan tujuan", "Hal. 120"),
        Rule("R46", Cluster.C, "Obsessive-Compulsive Personality Disorder", "OCPD",
             [{"dataset_map": "ocpd_perfectionism"}], 0.72,
             "Perfeksionisme tinggi menghambat penyelesaian tugas", "Hal. 120"),
        Rule("R47", Cluster.C, "Obsessive-Compulsive Personality Disorder", "OCPD",
             [{"dataset_map": "ocpd_rigid"}], 0.70,
             "Kaku & keras kepala dalam moral/etika/nilai", "Hal. 120"),
        Rule("R48", Cluster.C, "Obsessive-Compulsive Personality Disorder", "OCPD",
             [{"dataset_map": "ocpd_delegation"}], 0.70,
             "Tidak mau mendelegasikan karena tidak percaya orang lain", "Hal. 120"),
        Rule("R49", Cluster.C, "Obsessive-Compulsive Personality Disorder", "OCPD",
             [{"dataset_map": "ocpd_hoarding"}], 0.68,
             "Hidup di bawah kemampuan finansial & menabung berlebihan", "Hal. 120"),
        Rule("R50", Cluster.C, "Obsessive-Compulsive Personality Disorder", "OCPD",
             [{"dataset_map": "ocpd_workaholic"}], 0.68,
             "Sulit membuang barang tidak berguna", "Hal. 120"),
    ])

    return rules

# ============================================================
# DATASET MAPPER
# ============================================================

class DatasetMapper:
    def __init__(self, df=None):
        if df is not None:
            self.stats = self._compute_stats(df)
        else:
            # Default stats berdasarkan dataset asli
            self.stats = {
                'anxiety_q75': 16, 'anxiety_q50': 10, 'anxiety_q25': 5,
                'depression_q75': 15, 'depression_q50': 11, 'depression_q25': 6,
                'stress_q75': 7, 'stress_q50': 5, 'stress_q25': 3,
                'social_q25': 3, 'social_q75': 7,
                'esteem_q25': 3, 'esteem_q75': 7,
                'life_q25': 3,
                'lonely_q75': 7,
                'sleep_q25': 5.4, 'sleep_q50': 6.4,
                'financial_q75': 7, 'work_q75': 7,
            }

    def _compute_stats(self, df):
        return {
            'anxiety_q75': df['Anxiety_Score'].quantile(0.75),
            'anxiety_q50': df['Anxiety_Score'].median(),
            'anxiety_q25': df['Anxiety_Score'].quantile(0.25),
            'depression_q75': df['Depression_Score'].quantile(0.75),
            'depression_q50': df['Depression_Score'].median(),
            'depression_q25': df['Depression_Score'].quantile(0.25),
            'stress_q75': df['Stress_Level'].quantile(0.75),
            'stress_q50': df['Stress_Level'].median(),
            'stress_q25': df['Stress_Level'].quantile(0.25),
            'social_q25': df['Social_Support_Score'].quantile(0.25),
            'social_q75': df['Social_Support_Score'].quantile(0.75),
            'esteem_q25': df['Self_Esteem_Score'].quantile(0.25),
            'esteem_q75': df['Self_Esteem_Score'].quantile(0.75),
            'life_q25': df['Life_Satisfaction_Score'].quantile(0.25),
            'lonely_q75': df['Loneliness_Score'].quantile(0.75),
            'sleep_q25': df['Sleep_Hours'].quantile(0.25),
            'sleep_q50': df['Sleep_Hours'].median(),
            'financial_q75': df['Financial_Stress'].quantile(0.75),
            'work_q75': df['Work_Stress'].quantile(0.75),
        }

    def map_patient(self, p):
        s = self.stats

        conditions = {}

        # CLUSTER A
        conditions['paranoid_distrust'] = p.get('Stress_Level', 0) >= s['stress_q75'] and p.get('Social_Support_Score', 0) <= s['social_q25']
        conditions['paranoid_hypervigilance'] = p.get('Anxiety_Score', 0) >= s['anxiety_q50'] and p.get('Stress_Level', 0) >= s['stress_q75']
        conditions['paranoid_grudge'] = p.get('Loneliness_Score', 0) >= s['lonely_q75'] and p.get('Life_Satisfaction_Score', 0) <= s['life_q25']
        conditions['paranoid_counterattack'] = p.get('Stress_Level', 0) >= s['stress_q75'] and p.get('Anxiety_Score', 0) >= s['anxiety_q50']
        conditions['paranoid_family_distrust'] = p.get('Social_Support_Score', 0) <= s['social_q25'] and p.get('Family_History_Mental_Illness', 0) == 1

        conditions['schizoid_avoid_social'] = p.get('Social_Support_Score', 0) <= 2 and p.get('Loneliness_Score', 0) <= 3
        conditions['schizoid_no_desire'] = p.get('Social_Support_Score', 0) <= s['social_q25'] and p.get('Loneliness_Score', 0) <= s['lonely_q75']
        conditions['schizoid_flat_affect'] = p.get('Depression_Score', 0) <= s['depression_q50'] and p.get('Anxiety_Score', 0) <= s['anxiety_q50']
        conditions['schizoid_no_sexual'] = p.get('Social_Support_Score', 0) <= 2 and p.get('Age', 0) >= 25
        conditions['schizoid_no_attention'] = p.get('Social_Support_Score', 0) <= 2 and p.get('Self_Esteem_Score', 0) >= 5

        conditions['schizotypal_ideas_ref'] = p.get('Anxiety_Score', 0) >= s['anxiety_q75'] and p.get('Stress_Level', 0) >= s['stress_q75']
        conditions['schizotypal_magical'] = p.get('Anxiety_Score', 0) >= s['anxiety_q75'] and p.get('Meditation', 0) == 1
        conditions['schizotypal_speech'] = p.get('Anxiety_Score', 0) >= s['anxiety_q75'] and p.get('Stress_Level', 0) >= s['stress_q75']
        conditions['schizotypal_social_anxiety'] = p.get('Anxiety_Score', 0) >= s['anxiety_q75'] and p.get('Social_Support_Score', 0) <= s['social_q25']
        conditions['schizotypal_perception'] = p.get('Anxiety_Score', 0) >= s['anxiety_q75'] and p.get('Sleep_Hours', 0) < s['sleep_q50']

        # CLUSTER B
        conditions['antisocial_rule_break'] = p.get('Substance_Use', 'None') not in ['None', None, 'nan', ''] and p.get('Stress_Level', 0) >= s['stress_q75']
        conditions['antisocial_deceit'] = p.get('Stress_Level', 0) >= s['stress_q75'] and p.get('Financial_Stress', 0) >= s['financial_q75']
        conditions['antisocial_aggression'] = p.get('Stress_Level', 0) >= s['stress_q75'] and p.get('Physical_Activity_Hrs', 0) >= 5
        conditions['antisocial_no_remorse'] = p.get('Life_Satisfaction_Score', 0) >= 7 and p.get('Social_Support_Score', 0) <= s['social_q25']
        conditions['antisocial_irresponsible'] = p.get('Employment_Status', '') == 'Unemployed' and p.get('Financial_Stress', 0) >= s['financial_q75']

        conditions['borderline_unstable_rel'] = p.get('Social_Support_Score', 0) <= s['social_q25'] and p.get('Life_Satisfaction_Score', 0) <= s['life_q25']
        conditions['borderline_mood_swing'] = p.get('Anxiety_Score', 0) >= s['anxiety_q75'] and p.get('Depression_Score', 0) >= s['depression_q75']
        conditions['borderline_abandonment'] = p.get('Loneliness_Score', 0) >= s['lonely_q75'] and p.get('Anxiety_Score', 0) >= s['anxiety_q75']
        # Dataset tidak memuat laporan self-harm; R24 tidak dapat dievaluasi dari skor.
        conditions['borderline_self_harm'] = False
        conditions['borderline_identity'] = p.get('Self_Esteem_Score', 0) <= s['esteem_q25'] and p.get('Life_Satisfaction_Score', 0) <= s['life_q25']

        conditions['histrionic_attention'] = p.get('Social_Support_Score', 0) >= s['social_q75'] and p.get('Loneliness_Score', 0) >= s['lonely_q75']
        conditions['histrionic_provocative'] = p.get('Social_Support_Score', 0) >= s['social_q75'] and p.get('Age', 0) < 40
        conditions['histrionic_suggestible'] = p.get('Self_Esteem_Score', 0) <= s['esteem_q25'] and p.get('Social_Support_Score', 0) >= s['social_q75']
        conditions['histrionic_shallow'] = p.get('Social_Support_Score', 0) >= s['social_q75'] and p.get('Life_Satisfaction_Score', 0) <= s['life_q25']
        conditions['histrionic_appearance'] = p.get('Social_Support_Score', 0) >= s['social_q75'] and p.get('Physical_Activity_Hrs', 0) >= 3

        conditions['narcissistic_grandiosity'] = p.get('Self_Esteem_Score', 0) >= s['esteem_q75'] and p.get('Life_Satisfaction_Score', 0) >= 7
        conditions['narcissistic_admiration'] = p.get('Self_Esteem_Score', 0) >= s['esteem_q75'] and p.get('Social_Support_Score', 0) >= s['social_q75']
        conditions['narcissistic_no_empathy'] = p.get('Self_Esteem_Score', 0) >= s['esteem_q75'] and p.get('Social_Support_Score', 0) <= s['social_q25']
        conditions['narcissistic_exploitative'] = p.get('Self_Esteem_Score', 0) >= s['esteem_q75'] and p.get('Financial_Stress', 0) >= s['financial_q75']
        conditions['narcissistic_envious'] = p.get('Self_Esteem_Score', 0) >= s['esteem_q75'] and p.get('Life_Satisfaction_Score', 0) <= s['life_q25']

        # CLUSTER C
        conditions['avoidant_social'] = p.get('Anxiety_Score', 0) >= s['anxiety_q75'] and p.get('Social_Support_Score', 0) <= s['social_q25']
        conditions['avoidant_fear_rejection'] = p.get('Self_Esteem_Score', 0) <= s['esteem_q25'] and p.get('Anxiety_Score', 0) >= s['anxiety_q75']
        conditions['avoidant_inadequacy'] = p.get('Self_Esteem_Score', 0) <= s['esteem_q25'] and p.get('Life_Satisfaction_Score', 0) <= s['life_q25']
        conditions['avoidant_occupational'] = p.get('Employment_Status', '') == 'Unemployed' and p.get('Anxiety_Score', 0) >= s['anxiety_q75']

        conditions['dependent_decisions'] = p.get('Self_Esteem_Score', 0) <= s['esteem_q25'] and p.get('Social_Support_Score', 0) >= s['social_q75']
        conditions['dependent_submissive'] = p.get('Self_Esteem_Score', 0) <= s['esteem_q25'] and p.get('Social_Support_Score', 0) >= 5
        conditions['dependent_abandonment'] = p.get('Loneliness_Score', 0) >= s['lonely_q75'] and p.get('Social_Support_Score', 0) <= s['social_q25']
        conditions['dependent_disagree'] = p.get('Self_Esteem_Score', 0) <= 2 and p.get('Social_Support_Score', 0) >= 5
        conditions['dependent_helpless'] = p.get('Self_Esteem_Score', 0) <= s['esteem_q25'] and p.get('Employment_Status', '') == 'Unemployed'

        conditions['ocpd_order'] = p.get('Stress_Level', 0) >= s['stress_q75'] and p.get('Work_Stress', 0) >= s['work_q75']
        conditions['ocpd_perfectionism'] = p.get('Work_Stress', 0) >= s['work_q75'] and p.get('Anxiety_Score', 0) >= s['anxiety_q50']
        conditions['ocpd_rigid'] = p.get('Stress_Level', 0) >= s['stress_q75'] and p.get('Meditation', 0) == 0
        conditions['ocpd_delegation'] = p.get('Work_Stress', 0) >= s['work_q75'] and p.get('Self_Esteem_Score', 0) >= s['esteem_q75']
        conditions['ocpd_hoarding'] = p.get('Stress_Level', 0) >= s['stress_q75'] and p.get('Financial_Stress', 0) >= s['financial_q75']
        conditions['ocpd_workaholic'] = p.get('Work_Stress', 0) >= s['work_q75'] and p.get('Physical_Activity_Hrs', 0) <= 2

        # General
        conditions['therapy_active'] = p.get('Therapy', 0) == 1
        conditions['medication_use'] = p.get('Medication_Use', 'None') not in ['None', None, 'nan', '']
        conditions['substance_use'] = p.get('Substance_Use', 'None') not in ['None', None, 'nan', '']
        conditions['family_history'] = p.get('Family_History_Mental_Illness', 0) == 1
        conditions['chronic_illness'] = p.get('Chronic_Illnesses', 0) == 1
        conditions['sleep_poor'] = p.get('Sleep_Hours', 0) < 6
        conditions['sleep_good'] = p.get('Sleep_Hours', 0) > 7.5
        conditions['meditation_active'] = p.get('Meditation', 0) == 1

        return conditions

# ============================================================
# CF ENGINE
# ============================================================

class CFEngine:
    @staticmethod
    def combine_cf(cf1, cf2):
        if cf1 >= 0 and cf2 >= 0:
            return cf1 + cf2 * (1 - cf1)
        elif cf1 < 0 and cf2 < 0:
            return cf1 + cf2 * (1 + cf1)
        else:
            return (cf1 + cf2) / (1 - min(abs(cf1), abs(cf2)))

    @staticmethod
    def apply_modulation(base_cf, patient_conditions):
        cf = base_cf
        risk_mods = {
            'family_history': 0.05, 'chronic_illness': 0.03, 'substance_use': 0.05,
            'sleep_poor': 0.05, 'loneliness_high': 0.05,
        }
        protective_mods = {
            'therapy_active': -0.10, 'meditation_active': -0.05,
            'sleep_good': -0.05,
        }
        for factor, mod in risk_mods.items():
            if patient_conditions.get(factor, False):
                cf = min(0.99, cf + mod)
        for factor, mod in protective_mods.items():
            if patient_conditions.get(factor, False):
                cf = max(0.0, cf + mod)
        return round(cf, 4)

    @staticmethod
    def get_recommendation(cf):
        if cf < 0.45:
            return "🟢 Keyakinan rendah: indikasi lemah. Tetap jaga kesehatan mental dan amati gejala Anda."
        elif cf < 0.70:
            return "🟡 Keyakinan sedang: ada indikasi yang perlu diperhatikan. Disarankan konsultasi dengan profesional bila gejala mengganggu aktivitas sehari-hari."
        else:
            return "🟠 Keyakinan tinggi: indikasi kuat terdeteksi. Disarankan konsultasi dengan psikolog/psikiater untuk penilaian lebih lanjut."

# ============================================================
# FORWARD CHAINING ENGINE
# ============================================================

class ForwardChainingEngine:
    def __init__(self, rules, mapper, cf_engine):
        self.rules = rules
        self.mapper = mapper
        self.cf_engine = cf_engine

    def evaluate_rule(self, rule, patient_conditions):
        conditions = rule.conditions
        total = len(conditions)
        matched = sum(1 for c in conditions if patient_conditions.get(c.get('dataset_map', ''), False))

        # Kondisi majemuk pada basis aturan menggunakan AND secara eksplisit.
        triggered = total > 0 and matched == total

        if triggered:
            return True, rule.base_cf
        return False, 0.0

    def diagnose(self, patient_data=None, conditions=None):
        # conditions (dict kondisi boolean) boleh diberikan langsung (mode gejala Ya/Tidak).
        # Jika tidak ada, turunkan dari data pasien numerik lewat mapper (mode dataset).
        if conditions is None:
            conditions = self.mapper.map_patient(patient_data)
        patient_conditions = conditions
        disorder_cfs = {}

        for rule in self.rules:
            triggered, cf = self.evaluate_rule(rule, patient_conditions)
            if triggered and cf > 0.3:
                modulated = self.cf_engine.apply_modulation(cf, patient_conditions)
                if rule.disorder_code not in disorder_cfs:
                    disorder_cfs[rule.disorder_code] = {
                        'cf': modulated, 'rules': [{'id': rule.id, 'cf': modulated, 'desc': rule.description}],
                        'cluster': rule.cluster.value, 'disorder': rule.disorder
                    }
                else:
                    existing = disorder_cfs[rule.disorder_code]['cf']
                    disorder_cfs[rule.disorder_code]['cf'] = self.cf_engine.combine_cf(existing, modulated)
                    disorder_cfs[rule.disorder_code]['rules'].append({'id': rule.id, 'cf': modulated, 'desc': rule.description})

        # Komorbiditas calibration
        if len(disorder_cfs) > 3:
            for code in disorder_cfs:
                disorder_cfs[code]['cf'] = max(0.3, disorder_cfs[code]['cf'] * 0.85)

        results = []
        for code, data in disorder_cfs.items():
            final_cf = round(data['cf'], 4)
            if final_cf >= 0.30:
                results.append(DiagnosisResult(
                    disorder=data['disorder'], disorder_code=code, cluster=data['cluster'],
                    cf=final_cf, triggered_rules=data['rules'],
                    recommendation=self.cf_engine.get_recommendation(final_cf)
                ))
        results.sort(key=lambda x: x.cf, reverse=True)
        return results

# ============================================================
# KUESIONER GEJALA (Ya/Tidak) — SISTEM PAKAR
# ============================================================
# Setiap pertanyaan gejala memetakan langsung ke satu kondisi rule
# (dataset_map). Jawaban "Ya" = kondisi bernilai True untuk rule terkait.

CRISIS_INFO = (
    "Jika Anda sedang mengalami pikiran untuk menyakiti diri sendiri atau bunuh diri, "
    "hubungi **119 ekstensi 8** atau buka [Healing119.id](https://healing119.id). "
    "Jika Anda dalam bahaya segera, sudah terluka, atau layanan tidak tersambung, "
    "segera ke IGD terdekat dan minta orang tepercaya menemani. "
    "Informasi bantuan ini tidak bergantung pada nilai CF."
)

SYMPTOM_QUESTIONS = [
    ("Cluster A · Paranoid PD", [
        ("paranoid_distrust", "Curiga berlebihan terhadap motif orang lain & menganggap tindakan biasa sebagai ancaman terhadap diri"),
        ("paranoid_hypervigilance", "Selalu waspada berlebihan dan merasa mudah terancam sehingga sulit rileks"),
        ("paranoid_grudge", "Menyimpan dendam lama dan sulit memaafkan kesalahan kecil"),
        ("paranoid_counterattack", "Mudah marah dan menyerang balik saat merasa dihina atau dikritik"),
        ("paranoid_family_distrust", "Meragukan loyalitas orang terdekat, termasuk keluarga/pasangan"),
    ]),
    ("Cluster A · Schizoid PD", [
        ("schizoid_avoid_social", "Menghindari hubungan sosial dan lebih memilih aktivitas sendirian"),
        ("schizoid_no_desire", "Tidak ada keinginan untuk memulai atau mempertahankan pertemanan"),
        ("schizoid_flat_affect", "Jarang menunjukkan ekspresi emosi, tidak terpengaruh pujian maupun kritik"),
        ("schizoid_no_sexual", "Tidak tertarik pada pengalaman seksual dengan orang lain"),
        ("schizoid_no_attention", "Tidak membutuhkan perhatian atau penerimaan dari orang lain"),
    ]),
    ("Cluster A · Schizotypal PD", [
        ("schizotypal_ideas_ref", "Percaya kejadian yang tidak berkaitan punya makna khusus untuk diri sendiri (ideas of reference)"),
        ("schizotypal_magical", "Percaya memiliki kekuatan magis atau kemampuan memengaruhi orang lain lewat pikiran"),
        ("schizotypal_speech", "Cara bicara tidak biasa, sering melantur atau sulit dipahami orang lain"),
        ("schizotypal_social_anxiety", "Cemas berlebihan saat berinteraksi, terutama dengan orang asing"),
        ("schizotypal_perception", "Merasakan hal tidak biasa (misal merasakan kehadiran atau mendengar nama dipanggil)"),
    ]),
    ("Cluster B · Antisocial PD", [
        ("antisocial_rule_break", "Sering melanggar norma/hukum dan tidak peduli hak orang lain"),
        ("antisocial_deceit", "Sering berbohong atau berbuat curang demi keuntungan pribadi"),
        ("antisocial_aggression", "Mudah agresif dan bertindak sembrono terhadap keselamatan diri/orang lain"),
        ("antisocial_no_remorse", "Tidak merasa bersalah atas perbuatan yang merugikan orang lain"),
        ("antisocial_irresponsible", "Tidak mampu mempertahankan pekerjaan atau kewajiban finansial"),
    ]),
    ("Cluster B · Borderline PD", [
        ("borderline_unstable_rel", "Hubungan dengan orang lain sering berubah ekstrem (idealisasi lalu devaluasi)"),
        ("borderline_mood_swing", "Perubahan suasana hati yang drastis dalam waktu singkat"),
        ("borderline_abandonment", "Takut berlebihan ditinggalkan dan melakukan tindakan impulsif untuk mencegahnya"),
        ("borderline_self_harm", "Pernah atau sedang memiliki pikiran untuk menyakiti diri sendiri / bunuh diri"),
        ("borderline_identity", "Citra diri tidak stabil, sering merasa kosong atau paranoid sesaat"),
    ]),
    ("Cluster B · Histrionic PD", [
        ("histrionic_attention", "Tidak nyaman bila bukan pusat perhatian dan cenderung berperilaku dramatis"),
        ("histrionic_provocative", "Berperilaku provokatif/menggoda secara tidak pantas"),
        ("histrionic_suggestible", "Sangat mudah dipengaruhi pendapat orang lain atau tren"),
        ("histrionic_shallow", "Menganggap hubungan yang biasa saja sudah intim/akrab"),
        ("histrionic_appearance", "Menghabiskan waktu berlebihan untuk penampilan fisik"),
    ]),
    ("Cluster B · Narcissistic PD", [
        ("narcissistic_grandiosity", "Merasa diri hebat dan berlebihan membanggakan pencapaian"),
        ("narcissistic_admiration", "Butuh pujian/kekaguman berlebihan, marah bila tidak diistimewakan"),
        ("narcissistic_no_empathy", "Sulit memahami atau merasakan perasaan dan kebutuhan orang lain"),
        ("narcissistic_exploitative", "Memanfaatkan hubungan/orang lain demi kepentingan pribadi"),
        ("narcissistic_envious", "Sering iri pada orang lain atau percaya orang lain iri pada diri Anda"),
    ]),
    ("Cluster C · Avoidant PD", [
        ("avoidant_social", "Menghindari situasi sosial karena takut dikritik atau ditolak"),
        ("avoidant_fear_rejection", "Menginginkan hubungan dekat tetapi takut tidak disukai atau ditolak"),
        ("avoidant_inadequacy", "Merasa tidak mampu dan melebih-lebihkan risiko buruk dari hal baru"),
        ("avoidant_occupational", "Menghindari pekerjaan yang menuntut banyak interaksi karena takut dinilai"),
    ]),
    ("Cluster C · Dependent PD", [
        ("dependent_decisions", "Tidak bisa mengambil keputusan tanpa pendapat atau jaminan orang lain"),
        ("dependent_submissive", "Membiarkan orang lain mengambil alih kendali atas hidup Anda"),
        ("dependent_abandonment", "Takut ditinggalkan dan segera mencari pengganti saat sendiri"),
        ("dependent_disagree", "Rela melakukan hal yang tidak menyenangkan demi mempertahankan dukungan orang lain"),
        ("dependent_helpless", "Merasa tidak berdaya dan tidak percaya diri mengurus diri sendiri"),
    ]),
    ("Cluster C · OCPD", [
        ("ocpd_order", "Sangat terobsesi pada keteraturan/daftar/detail hingga menghambat tujuan utama"),
        ("ocpd_perfectionism", "Perfeksionis berlebihan sampai tugas tidak kunjung selesai"),
        ("ocpd_rigid", "Kaku dan keras kepala soal moral, aturan, atau nilai"),
        ("ocpd_delegation", "Tidak mau mendelegasikan tugas karena tidak percaya orang lain"),
        ("ocpd_hoarding", "Hidup di bawah kemampuan finansial dan menabung secara berlebihan"),
        ("ocpd_workaholic", "Sulit membuang barang yang tidak berguna"),
    ]),
]

# ============================================================
# STREAMLIT UI
# ============================================================

def main():
    # Header
    st.markdown('<div class="main-header">Sistem Pakar Pemetaan Gejala Mental</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Forward Chaining + Certainty Factor — Pemetaan Berbasis Aturan (Rule-Based)</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### Menu")
        mode = st.radio("Pilih Mode:", ["📝 Kuesioner (Input Manual)", "📊 Dataset (Upload CSV)", " Tentang Sistem"])

        st.markdown("---")
        st.markdown("### Cluster Gangguan")
        st.markdown('<span class="cluster-a">🔴 Cluster A: Aneh/Eksentrik</span><br>Paranoid, Schizoid, Schizotypal', unsafe_allow_html=True)
        st.markdown('<span class="cluster-b">🟡 Cluster B: Dramatis/Emosional</span><br>Antisocial, Borderline, Histrionic, Narcissistic', unsafe_allow_html=True)
        st.markdown('<span class="cluster-c">🟢 Cluster C: Cemas/Takut</span><br>Avoidant, Dependent, OCPD', unsafe_allow_html=True)

    st.warning("Prototipe edukasi berbasis aturan, bukan diagnosis medis. CF bukan probabilitas penyakit, akurasi, atau tingkat keparahan. Bobot dan pemetaan belum divalidasi klinis.")
    st.info(CRISIS_INFO)

    # Initialize engine
    rules = build_knowledge_base()

    if mode == "📝 Kuesioner (Input Manual)":
        run_kuesioner_mode(rules)
    elif mode == "📊 Dataset (Upload CSV)":
        run_dataset_mode(rules)
    else:
        run_about_mode()


def run_kuesioner_mode(rules):
    st.markdown("### 📝 Kuesioner — Gejala (opsi, Ya/Tidak)")
    st.info("Centang gejala yang Anda alami. Gejala yang tidak dicentang dianggap tidak ada. "
            "Jawaban Anda dipetakan langsung ke 50 rule pada knowledge base (forward chaining).")

    with st.expander("ℹ️ Panduan pengisian", expanded=False):
        st.markdown(
            "Sistem pakar bekerja dengan mencocokkan **gejala** terhadap **aturan IF–THEN**, "
            "bukan skor numerik. Cukup centang pernyataan yang sesuai kondisi Anda, lalu tekan "
            "**Jalankan Pemetaan**."
        )

    answers = {}
    for group_title, items in SYMPTOM_QUESTIONS:
        with st.expander(group_title, expanded=False):
            for key, question in items:
                answers[key] = st.checkbox(question, key=key)

    if answers.get('borderline_self_harm'):
        st.warning("Anda menandai pengalaman terkait menyakiti diri. Ini tidak menetapkan diagnosis gangguan kepribadian.")
        st.info(CRISIS_INFO)

    st.markdown("#### Faktor Pendukung (bobot asumsi, belum tervalidasi)")
    col1, col2, col3 = st.columns(3)
    with col1:
        family_history = st.checkbox("Riwayat keluarga gangguan mental", key="fh")
        chronic = st.checkbox("Penyakit kronis", key="ch")
    with col2:
        therapy = st.checkbox("Sedang menjalani terapi", key="th")
        meditation = st.checkbox("Rutin meditasi/relaksasi", key="md")
    with col3:
        substance = st.checkbox("Menggunakan zat (alkohol/obat terlarang)", key="su")
        sleep_poor = st.checkbox("Gangguan tidur (sulit tidur / tidak nyenyak)", key="sp")

    if st.button("🔍 JALANKAN PEMETAAN", type="primary"):
        conditions = dict(answers)
        conditions.update({
            'family_history': family_history,
            'chronic_illness': chronic,
            'therapy_active': therapy,
            'meditation_active': meditation,
            'substance_use': substance,
            'sleep_poor': sleep_poor,
        })

        # Ringkasan pasien untuk trace/explainability
        patient = {
            'mode': 'kuesioner',
            'gejala_terkonfirmasi': [k for k, v in conditions.items() if v],
        }

        mapper = DatasetMapper()
        cf_engine = CFEngine()
        engine = ForwardChainingEngine(rules, mapper, cf_engine)
        results = engine.diagnose(conditions=conditions)

        display_results(results, patient)


def run_dataset_mode(rules):
    st.markdown("### 📊 Mode Dataset - Upload CSV")
    st.info("Upload file CSV dengan format yang sesuai dataset 'Anxiety and Depression Mental Health Factors' "
            "(cara mengunduhnya tercantum di README).")
    st.caption(
        "Catatan: pemetaan 21 fitur dataset ke 50 rule adalah asumsi kelompok dan belum divalidasi, "
        "karena dataset ini tidak memiliki label gangguan kepribadian. Mode ini untuk eksplorasi batch, "
        "bukan pengukuran akurasi. R24 tidak dievaluasi: dataset tidak memuat laporan self-harm."
    )

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:
        try:
            df = pd.read_csv(uploaded, keep_default_na=False)
            required = "Age Gender Education_Level Employment_Status Sleep_Hours Physical_Activity_Hrs Social_Support_Score Anxiety_Score Depression_Score Stress_Level Family_History_Mental_Illness Chronic_Illnesses Medication_Use Therapy Meditation Substance_Use Financial_Stress Work_Stress Self_Esteem_Score Life_Satisfaction_Score Loneliness_Score".split()
            missing = sorted(set(required) - set(df.columns))
            if missing:
                raise ValueError("Kolom wajib tidak tersedia: " + ", ".join(missing))
            if df.empty:
                raise ValueError("CSV tidak memiliki baris data.")
            categorical = {'Gender', 'Education_Level', 'Employment_Status', 'Medication_Use', 'Substance_Use'}
            for column in set(required) - categorical:
                df[column] = pd.to_numeric(df[column], errors='raise')
                if not np.isfinite(df[column]).all():
                    raise ValueError(f"Kolom {column} memuat nilai kosong/tidak terbatas.")
        except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
            st.error(f"CSV tidak dapat diproses: {exc}")
            return
        st.success(f"✅ Dataset loaded: {df.shape[0]} baris, {df.shape[1]} kolom")

        st.markdown("#### Preview Data:")
        st.dataframe(df.head())

        # Stats
        mapper = DatasetMapper(df)
        cf_engine = CFEngine()
        engine = ForwardChainingEngine(rules, mapper, cf_engine)

        # Run diagnosis on all
        all_results = []
        for idx, row in df.iterrows():
            patient = row.to_dict()
            res = engine.diagnose(patient)
            all_results.append({
                'index': idx,
                'top_disorder': res[0].disorder_code if res else "None",
                'top_cf': res[0].cf if res else 0,
                'num_disorders': len(res),
                'all_results': res
            })

        results_df = pd.DataFrame([{
            'Pasien': r['index'] + 1,
            'Hipotesis Aturan': r['top_disorder'],
            'CF': f"{r['top_cf']:.1%}",
            'Jumlah Hipotesis': r['num_disorders']
        } for r in all_results])

        st.markdown("#### 📋 Hasil Simulasi Pemetaan Dataset:")
        st.dataframe(results_df)

        # Visualisasi
        st.markdown("#### 📊 Visualisasi Distribusi")

        col_v1, col_v2 = st.columns(2)

        with col_v1:
            disorder_counts = results_df['Hipotesis Aturan'].value_counts()
            fig = px.pie(values=disorder_counts.values, names=disorder_counts.index,
                        title="Distribusi Hipotesis Aturan")
            st.plotly_chart(fig, use_container_width=True)

        with col_v2:
            cf_values = [r['top_cf'] for r in all_results if r['top_cf'] > 0]
            fig2 = px.histogram(x=cf_values, nbins=20, title="Distribusi Certainty Factor",
                               labels={'x': 'CF Value'})
            st.plotly_chart(fig2, use_container_width=True)

        # Detail per pasien
        st.markdown("#### 🔍 Detail Per Pasien")
        selected = st.selectbox("Pilih Pasien:", range(1, min(len(df)+1, 51)), 
                               format_func=lambda x: f"Pasien #{x}")

        patient = df.iloc[selected-1].to_dict()
        res = all_results[selected-1]['all_results']

        display_results(res, patient)


def display_results(results, patient):
    st.markdown("---")
    st.markdown("### HASIL PEMETAAN ATURAN")
    st.caption("CF adalah skor internal prototipe, bukan peluang penyakit atau keputusan klinis.")

    if not results:
        st.info("Tidak ada aturan yang cocok dengan input. Ini tidak membuktikan ada atau tidaknya gangguan; konsultasikan keluhan yang mengganggu kepada profesional.")
        return

    # Deteksi indikasi self-harm / suicidal (rule R24)
    self_harm_triggered = any(
        rule['id'] == 'R24' for r in results for rule in r.triggered_rules
    )
    if self_harm_triggered:
        st.warning("R24 cocok dengan jawaban yang dilaporkan, bukan diagnosis. Pengalaman self-harm tidak spesifik terhadap satu gangguan.")
        st.info(CRISIS_INFO)

    # Summary cards
    cols = st.columns(min(len(results), 4))
    for i, r in enumerate(results[:4]):
        with cols[i]:
            color = "#e74c3c" if r.cf >= 0.70 else "#f39c12" if r.cf >= 0.45 else "#27ae60"
            st.markdown(f"""
            <div style="background: {color}15; border: 2px solid {color}; border-radius: 10px; padding: 15px; text-align: center;">
                <div style="font-size: 0.9rem; color: #555;">{r.disorder_code}</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: {color};">{r.cf:.1%}</div>
                <div style="font-size: 0.8rem; color: {color};">{'🟠 Keyakinan Tinggi' if r.cf >= 0.70 else '🟡 Keyakinan Sedang' if r.cf >= 0.45 else '🟢 Keyakinan Rendah'}</div>
            </div>
            """, unsafe_allow_html=True)

    # Detailed table
    st.markdown("#### Detail per Gangguan")

    for r in results:
        cluster_color = "cluster-a" if "Aneh" in r.cluster else "cluster-b" if "Dramatis" in r.cluster else "cluster-c"
        cf_class = "cf-high" if r.cf >= 0.70 else "cf-mid" if r.cf >= 0.45 else "cf-low"

        with st.expander(f"{r.disorder_code} — CF: {r.cf:.1%}"):
            st.markdown(f"""
            <div class="rule-box">
                <b>Cluster:</b> <span class="{cluster_color}">{r.cluster}</span><br>
                <b>CF Final:</b> <span class="{cf_class}">{r.cf:.2%}</span><br>
                <b>Rekomendasi:</b> {r.recommendation}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Rules yang Terpicu:**")
            for rule in r.triggered_rules:
                st.markdown(f"- `{rule['id']}`: {rule['desc']} (CF: {rule['cf']:.1%})")

    # Ringkasan rule terpicu per gangguan (explainability)
    st.markdown("#### 📊 Jumlah Rule Terpicu per Gangguan")
    rule_counts = {r.disorder_code: len(r.triggered_rules) for r in results}
    fig_rules = go.Figure(go.Bar(
        x=list(rule_counts.values()),
        y=list(rule_counts.keys()),
        orientation='h',
        marker_color='#1f4e79',
    ))
    fig_rules.update_layout(
        xaxis_title="Jumlah rule terpicu",
        yaxis_title="Gangguan",
        yaxis=dict(autorange="reversed"),
        height=max(300, 40 * len(rule_counts)),
        margin=dict(l=160),
    )
    st.plotly_chart(fig_rules, use_container_width=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <b>⚠️ PERINGATAN:</b> Sistem ini adalah <b>pemetaan berbasis aturan (rule-based)</b> untuk tujuan edukasi,
        BUKAN alat diagnostik klinis dan BUKAN pengganti penilaian profesional.
        Hasil harus dikonfirmasi oleh psikolog atau psikiater.
    </div>
    """, unsafe_allow_html=True)
    st.info(CRISIS_INFO)


def run_about_mode():
    st.markdown("""
    ### Tentang Sistem Pakar Ini

    **Sistem Pakar Diagnosis Penyakit Mental** adalah sistem **pemetaan berbasis aturan (rule-based)**,
    bukan model machine learning yang divalidasi. Sistem mencocokkan gejala yang dilaporkan pengguna
    terhadap 50 aturan IF–THEN untuk menghasilkan hipotesis gangguan beserta tingkat keyakinannya (Certainty Factor).

    #### 🔧 Arsitektur Sistem
    1. **Knowledge Base (50 Rules)**: Berbasis DSM-5 & "Essentials of Abnormal Psychology" (Bridley & Daffin, 2019)
    2. **Forward Chaining**: Inferensi dari fakta (gejala Ya/Tidak) → evaluasi rules → hipotesis gangguan
    3. **Certainty Factor (CF)**: Mengelola ketidakpastian dengan formula kombinasi CF
    4. **Dataset Pendukung**: Dataset 'Anxiety and Depression Mental Health Factors' hanya dipakai untuk
       perhitungan kuartil eksploratif pada mode dataset — bukan untuk melatih maupun memvalidasi model

    #### 10 Gangguan yang Dipetakan
    | Cluster | Gangguan | Rules |
    |---------|----------|-------|
    | **A: Aneh/Eksentrik** | Paranoid PD, Schizoid PD, Schizotypal PD | R01-R15 |
    | **B: Dramatis/Emosional** | Antisocial PD, Borderline PD, Histrionic PD, Narcissistic PD | R16-R35 |
    | **C: Cemas/Takut** | Avoidant PD, Dependent PD, OCPD | R36-R50 |

    #### ⚠️ Keterbatasan (Penting)
    - **Tidak divalidasi secara klinis** dan **tidak ada klaim akurasi**. Akurasi tidak dapat dihitung
      karena dataset yang digunakan tidak memiliki label diagnosis gangguan kepribadian.
    - **Pemetaan 21 fitur dataset → 50 rule adalah asumsi kelompok**, belum diuji terhadap ground truth
      psikiater/psikolog.
    - **Bukan alat diagnostik.** Hasil hanyalah indikasi awal dan wajib dikonfirmasi oleh profesional.
    - **Input self-report** bergantung pada kejujuran pengguna dalam mengisi gejala.

    #### Referensi
    - Bridley, A., & Daffin, L. W. (2019). *Essentials of Abnormal Psychology*. Washington State University.
    - Dataset: Anxiety and Depression Mental Health Factors (1200 sampel, 21 fitur)
    """)
    st.info(CRISIS_INFO)


if __name__ == "__main__":
    main()
