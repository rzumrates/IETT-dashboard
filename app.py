from streamlit_folium import st_folium
import folium
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from twilio.rest import Client

@st.cache_resource
def train_model():
    df = pd.read_csv("data/sensors_train.csv")
    X = df.drop(columns=["label"])
    y = df["label"]
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X, y)
    return model

def load_data():
    sensors = pd.read_csv("data/sensors_simulated.csv")
    lines = pd.read_csv("data/iett_lines.csv")
    faults = pd.read_csv("data/iett_faults.csv")
    passengers = pd.read_csv("data/iett_passenger.csv")
    return sensors, lines, faults, passengers

def compute_priority(row, passenger_df, faults_df):
    hat = row['hat_kodu']
    bus = row['bus_id']
    
 
    recent = passenger_df[passenger_df['SHATKODU'] == hat]
    if recent.empty:
        density_score = 0.0
    else:
        density_score = recent['YOLCU_SAYISI'].mean() / passenger_df['YOLCU_SAYISI'].max()

    fault_score = 1.0 if bus in faults_df['SKAPINUMARA'].values else 0.0

    return 0.6 * density_score + 0.4 * fault_score

def send_sms(bus_id):
    try:
        account_sid = st.secrets["TWILIO_SID"]
        auth_token = st.secrets["TWILIO_TOKEN"]
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=f"[Otobüs {bus_id}] Bakım gerektiği tespit edildi. Lütfen teknik servise başvurun.",
            from_='+18382501727',
            to='+905438289912'
        )
        return message.sid
    except Exception as e:
        return str(e)

st.set_page_config("Otobüs Bakım Dashboard", layout="wide")
st.title("🔧 Otobüs Bakım Tahmin ve Bildirim Sistemi")

model = train_model()
sensors_df, lines_df, faults_df, passenger_df = load_data()

X_sim = sensors_df.drop(columns=["bus_id", "hat_kodu", "label"])
sensors_df['prediction'] = model.predict(X_sim)
sensors_df['prediction_prob'] = model.predict_proba(X_sim)[:, 1]

sensors_df['priority_score'] = sensors_df.apply(lambda row: compute_priority(row, passenger_df, faults_df), axis=1)
sorted_df = sensors_df[sensors_df['prediction'] == 1].sort_values(by="priority_score", ascending=False)

st.subheader("🛠️ Bakım Gereken Otobüsler")
st.dataframe(sorted_df[['bus_id', 'hat_kodu', 'priority_score']])

for _, row in sorted_df.iterrows():
    if st.button(f"📩 {row['bus_id']} - Mesaj Gönder"):
        result = send_sms(row['bus_id'])
        st.success(f"Mesaj gönderildi: {result}")

st.subheader("🗺️ Bakım Gereken Otobüslerin Konumları")

# Harita oluştur
m = folium.Map(location=[41.015137, 28.979530], zoom_start=11)  # İstanbul ortalama koordinat

# Sadece bakım gereken otobüslerin konumları
for _, row in sorted_df.iterrows():
    fault_row = faults_df[faults_df["SKAPINUMARA"] == row["bus_id"]]
    if not fault_row.empty:
        lat = fault_row.iloc[0]["NENLEM"]
        lon = fault_row.iloc[0]["NBOYLAM"]
        folium.Marker(
            location=[lat, lon],
            popup=f"{row['bus_id']} - {row['hat_kodu']}",
            icon=folium.Icon(color='red', icon='wrench', prefix='fa')
        ).add_to(m)

# Haritayı Streamlit'e göm
st_folium(m, width=700)
