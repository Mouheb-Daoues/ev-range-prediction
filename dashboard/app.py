import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# 1. Configuration de la page
st.set_page_config(page_title="EV Range Predictor Pro", page_icon="⚡", layout="wide")

st.title("⚡ Simulateur & Prédiction d'Autonomie VE")
st.write("Modifiez les paramètres du véhicule dans la barre latérale pour observer les prédictions en temps réel.")

# 2. Chargement des modèles d'IA
@st.cache_resource
def load_ai_model(model_name):
    if model_name == "Random Forest":
        return joblib.load('models/random_forest_model.pkl')
    elif model_name == "XGBoost":
        return joblib.load('models/xgboost_model.pkl')

st.sidebar.header("🧠 Configuration de l'IA")
chosen_model_name = st.sidebar.selectbox("Sélectionnez le modèle d'IA", options=["Random Forest", "XGBoost"])

try:
    model = load_ai_model(chosen_model_name)
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# 3. Curseurs (Widgets Réseau CAN)
st.sidebar.header("🕹️ Signaux du Véhicule (Réseau CAN)")
speed = st.sidebar.slider("Vitesse Moyenne (km/h)", min_value=10.0, max_value=150.0, value=50.0, step=1.0)
speed_std = st.sidebar.slider("Variabilité Vitesse (Écart-type)", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
consumption = st.sidebar.slider("Consommation Moyenne (kWh/100km)", min_value=5.0, max_value=40.0, value=15.0, step=0.5)
throttle_std = st.sidebar.slider("Variabilité Pédale Accélérateur", min_value=0.0, max_value=30.0, value=5.0, step=0.5)
energy = st.sidebar.slider("Énergie Restante Batterie (kWh)", min_value=0.0, max_value=60.0, value=45.0, step=0.5)

st.sidebar.markdown("---")
agility = st.sidebar.selectbox("Mode de conduite", options=[0, 1, 2], format_func=lambda x: ["Éco", "Normal", "Sport"][x])
recup = st.sidebar.selectbox("Niveau de Récupération", options=[0, 1, 2], format_func=lambda x: ["Désactivé", "Faible", "Fort"][x])

# 4. Préparation et normalisation des données
input_data = pd.DataFrame([{
    'speed_mean_30s': speed, 'speed_std_30s': speed_std, 'consumption_mean_30s': consumption,
    'throttle_std_30s': throttle_std, 'sigRemainingEnergy': energy, 'sigAgility': agility, 'sigRecuperationLevel': recup
}])

means = np.array([66.5, 6.2, 17.8, 14.3, 31.2, 1.0, 1.0])
stds = np.array([32.4, 4.8,  7.1, 14.8, 17.5, 0.8, 0.8])
input_data_scaled = pd.DataFrame((input_data.values - means) / stds, columns=input_data.columns)

# 5. Calcul de la prédiction brute
prediction = model.predict(input_data_scaled)[0]
if energy == 0:
    prediction = 0.0
else:
    prediction = np.clip(prediction, 0, (energy / (consumption / 100)))

# ==========================================================
# INTERFACE VISUELLE : Organisation en 2 colonnes majeures
# ==========================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Métriques Clés")
    st.metric(label="Distance estimée avant recharge", value=f"{prediction:.1f} km")
    
    # Alertes contextuelles dynamiques
    soc_percentage = (energy / 60.0) * 100
    if soc_percentage < 20:
        st.warning("⚠️ Batterie faible ! Pensez à recharger.")
    elif agility == 2:
        st.info("🏃 Mode Sport activé : forte réactivité demandée.")
    else:
        st.success("🌱 Conduite optimale détectée.")

    # Graphique 1 : Jauge circulaire (Compteur SoC %)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = soc_percentage,
        number = {'suffix': "%", 'font': {'size': 32}},
        title = {'text': "État de Charge Batterie (SoC)", 'font': {'size': 18}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': "#2ca02c" if soc_percentage > 20 else "#d62728"},
            'steps': [
                {'range': [0, 20], 'color': "rgba(214, 39, 40, 0.2)"},
                {'range': [20, 100], 'color': "rgba(44, 160, 44, 0.1)"}
            ],
        }
    ))
    fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col2:
    st.subheader("📉 Simulation de Décharge Linéaire")
    
    # Graphique 2 : Courbe dynamique prédisant l'autonomie restante au fil du trajet
    # On crée 20 points fictifs du départ (0km parcouru) jusqu'à la fin de la batterie
    distance_steps = np.linspace(0, prediction, 20)
    remaining_battery_steps = np.linspace(energy, 0, 20)
    
    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=distance_steps, 
        y=remaining_battery_steps, 
        mode='lines+markers',
        name='Énergie disponible',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=5)
    ))
    
    fig_line.update_layout(
        xaxis_title="Distance parcourue simulée (km)",
        yaxis_title="Énergie Restante en Batterie (kWh)",
        height=320,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig_line, use_container_width=True)