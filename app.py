import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import shap
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="Piyasa Değeri Tahminleyicisi", page_icon="⚽", layout="wide")

st.title("⚽ Futbolcu Piyasa Değeri Tahminleyicisi (Canlı Demo)")
st.markdown("Bu demo, makine öğrenmesi (LightGBM) kullanılarak bir futbolcunun tahmini piyasa değerini hesaplar. Yalnızca temel metrikleri değiştirerek modelin değerleri nasıl güncellediğini canlı olarak görebilirsiniz.")

# --- PATHS ---
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "results" / "saved_models"
DATA_DIR = BASE_DIR / "dataset" / "model_input"

# --- CACHE DATA & MODELS ---
@st.cache_resource(show_spinner="Model ve Scaler yükleniyor...")
def load_models():
    # Load Model
    model_path = MODELS_DIR / "lightgbm.pkl"
    if not model_path.exists():
        st.error(f"Model dosyası bulunamadı: {model_path}")
        st.stop()
    model = joblib.load(model_path)
    
    # Load Scaler
    scaler_path = DATA_DIR / "standard_scaler.pkl"
    if not scaler_path.exists():
        st.error(f"Scaler dosyası bulunamadı: {scaler_path}")
        st.stop()
    scaler = joblib.load(scaler_path)
    
    return model, scaler

@st.cache_data(show_spinner="Ortalama oyuncu profili hesaplanıyor...")
def load_base_profile():
    # X_train.csv üzerinden medyan profil çıkartalım
    train_path = DATA_DIR / "X_train.csv"
    if not train_path.exists():
        st.error(f"Eğitim verisi bulunamadı: {train_path}")
        st.stop()
    
    # Read a sample to get median efficiently (or read full if memory allows)
    # Using full data since Streamlit caching handles the load
    df = pd.read_csv(train_path)
    median_profile = df.median().to_dict()
    column_order = df.columns.tolist()
    return median_profile, column_order

try:
    model, scaler = load_models()
    base_profile, column_order = load_base_profile()
except Exception as e:
    st.error(f"Yükleme sırasında hata oluştu: {e}")
    st.stop()

# --- SIDEBAR: USER INPUTS ---
st.sidebar.header("Oyuncu Özellikleri")

age = st.sidebar.slider("Yaş (age)", min_value=15, max_value=45, value=int(base_profile.get('age', 25)), step=1)
contract_days = st.sidebar.slider("Sözleşme Kalan Gün", min_value=0, max_value=1825, value=int(base_profile.get('contract_days_remaining', 365)), step=30)
career_minutes = st.sidebar.number_input("Kariyer Oynama Süresi (Dk)", min_value=0, max_value=50000, value=int(base_profile.get('career_minutes_played', 5000)), step=100)

position_options = ["Hücum (Attack)", "Orta Saha (Midfield)", "Defans (Defender)", "Kaleci (Goalkeeper)"]
selected_pos = st.sidebar.selectbox("Ana Pozisyon", options=position_options)

if "Goalkeeper" in selected_pos:
    with st.sidebar.expander("🧤 Kaleci İstatistikleri", expanded=True):
        clean_sheets = st.number_input("Gol Yemediği Maç (Clean Sheet)", min_value=0, max_value=500, value=int(base_profile.get('career_clean_sheets', 15)), step=1)
        goals_conceded = st.number_input("Yenilen Toplam Gol", min_value=0, max_value=1000, value=int(base_profile.get('career_goals_conceded', 40)), step=1)
        career_goals = 0
        recent_goals = 0
        career_assists = 0
        recent_assists = 0
else:
    career_goals = st.sidebar.number_input("Kariyer Golleri", min_value=0, max_value=500, value=int(base_profile.get('career_goals', 10)), step=1)
    career_assists = st.sidebar.number_input("Kariyer Asistleri", min_value=0, max_value=300, value=int(base_profile.get('career_assists', 5)), step=1)
    recent_goals = st.sidebar.number_input("Son Dönem Golleri", min_value=0, max_value=50, value=int(base_profile.get('recent_goals', 2)), step=1)
    recent_assists = st.sidebar.number_input("Son Dönem Asistleri", min_value=0, max_value=50, value=int(base_profile.get('recent_assists', 1)), step=1)
    clean_sheets = int(base_profile.get('career_clean_sheets', 0))
    goals_conceded = int(base_profile.get('career_goals_conceded', 0))

with st.sidebar.expander("Disiplin Geçmişi", expanded=True):
    yellow_cards = st.number_input("Sarı Kart (Kariyer)", min_value=0, max_value=200, value=int(base_profile.get('career_yellow_cards', 5)), step=1)
    red_cards = st.number_input("Kırmızı Kart (Kariyer)", min_value=0, max_value=50, value=int(base_profile.get('career_direct_red_cards', 0)), step=1)

with st.sidebar.expander("💸 Transfer Geçmişi", expanded=True):
    last_transfer_fee = st.number_input("Son Transfer Bedeli (€)", min_value=0, max_value=300000000, value=int(base_profile.get('last_transfer_fee', 1000000)), step=100000)
    max_value_at_transfer = st.number_input("Transfer Anındaki Max Değeri (€)", min_value=0, max_value=300000000, value=int(base_profile.get('max_value_at_transfer', 2000000)), step=100000)

with st.sidebar.expander("🌍 Milli Takım", expanded=True):
    national_matches = st.number_input("Milli Maça Çıkma Sayısı", min_value=0, max_value=200, value=int(base_profile.get('national_matches', 0)), step=1)
    national_goals = st.number_input("Milli Takım Golleri", min_value=0, max_value=150, value=int(base_profile.get('national_goals', 0)), step=1)

# --- PREPARE INPUT FEATURE VECTOR ---
# Start with base profile
current_profile = base_profile.copy()

# Override with user inputs
current_profile['age'] = age
current_profile['contract_days_remaining'] = contract_days
current_profile['career_goals'] = career_goals
current_profile['career_assists'] = career_assists
current_profile['recent_goals'] = recent_goals
current_profile['recent_assists'] = recent_assists
current_profile['career_minutes_played'] = career_minutes
current_profile['career_yellow_cards'] = yellow_cards
current_profile['career_direct_red_cards'] = red_cards
current_profile['career_clean_sheets'] = clean_sheets
current_profile['career_goals_conceded'] = goals_conceded
current_profile['last_transfer_fee'] = last_transfer_fee
current_profile['max_value_at_transfer'] = max_value_at_transfer
current_profile['national_matches'] = national_matches
current_profile['national_goals'] = national_goals

# Re-calculate derived features if necessary
nb_in_group = max(1, current_profile.get('career_nb_in_group', 1))
nb_on_pitch = max(1, current_profile.get('career_nb_on_pitch', 1))

current_profile['career_goals_per_game'] = career_goals / nb_in_group
current_profile['career_assists_per_game'] = career_assists / nb_in_group
current_profile['goal_contribution_ratio'] = (career_goals + career_assists) / nb_in_group
current_profile['recent_goals_per_game'] = recent_goals / max(1, current_profile.get('recent_nb_in_group', 1))
current_profile['recent_assists_per_game'] = recent_assists / max(1, current_profile.get('recent_nb_in_group', 1))

current_profile['yellow_cards_per_game'] = yellow_cards / nb_in_group
current_profile['red_cards_per_game'] = (red_cards + current_profile.get('career_second_yellow_cards', 0)) / nb_in_group
current_profile['clean_sheet_ratio'] = clean_sheets / nb_on_pitch
current_profile['fee_to_max_value_ratio'] = last_transfer_fee / max(1, max_value_at_transfer)
current_profile['national_goals_per_match'] = national_goals / max(1, national_matches)

# Handle One-Hot Position
current_profile['main_position_Attack'] = 0
current_profile['main_position_Defender'] = 0
current_profile['main_position_Goalkeeper'] = 0
current_profile['main_position_Midfield'] = 0

if "Attack" in selected_pos:
    current_profile['main_position_Attack'] = 1
elif "Defender" in selected_pos:
    current_profile['main_position_Defender'] = 1
elif "Goalkeeper" in selected_pos:
    current_profile['main_position_Goalkeeper'] = 1
elif "Midfield" in selected_pos:
    current_profile['main_position_Midfield'] = 1

# Create DataFrame matching the exact column order
input_df = pd.DataFrame([current_profile], columns=column_order)

# --- INFERENCE ---
# Scale inputs
try:
    input_scaled = scaler.transform(input_df)
    # Predict
    prediction = model.predict(input_scaled)[0]
    
    # Ensure value isn't negative
    market_value = max(0, prediction)
except Exception as e:
    st.error(f"Tahminleme sırasında hata: {e}")
    market_value = 0

# --- DISPLAY RESULTS ---
st.write("---")
st.subheader("💡 Modelin Tahmini")

# Format as currency
formatted_value = f"€ {market_value:,.0f}"

col1, col2 = st.columns([1, 2])
with col1:
    st.metric(label="Tahmini Piyasa Değeri", value=formatted_value, delta="Yapay Zeka Tahmini")

with col2:
    st.info("""
    **Nasıl Çalışır?**
    Sol taraftaki özellikleri değiştirdiğinizde:
    1. Arka planda 80 sütunluk bir "Ortalama Oyuncu" profili oluşturulur.
    2. Seçtiğiniz değerler bu profilin üzerine yazılır.
    3. Veriler eğitilmiş Scaler ile ölçeklendirilir.
    4. LightGBM modeli milisaniyeler içerisinde yeni piyasa değerini hesaplar.
    """)

st.write("---")
st.subheader("🔍 Neden Bu Değer? (Açıklanabilir Yapay Zeka - SHAP)")
st.markdown("Aşağıdaki **Şelale (Waterfall) Grafiği**, oyuncunun değerini **hangi istatistiğin ne kadar artırdığını veya düşürdüğünü** şeffaf bir şekilde açıklar. Mavi barlar değeri düşüren, kırmızı barlar ise artıran etkenlerdir.")

with st.spinner("SHAP değerleri hesaplanıyor..."):
    try:
        # Create explainer and compute SHAP values
        explainer = shap.TreeExplainer(model)
        # We must pass the scaled input
        shap_values = explainer(input_scaled)
        
        # Manually attach feature names since input_scaled is a numpy array
        shap_values.feature_names = column_order
        
        # Draw waterfall plot
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_values[0], max_display=12, show=False)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"SHAP grafiği oluşturulurken bir hata oluştu: {e}")

st.write("---")
st.caption("Eğitim seti medyan değerleri kullanılarak hesaplanmıştır.")
