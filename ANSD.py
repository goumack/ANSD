import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prévision Régionale", layout="wide")

# Ajouter une icône locale à côté du titre
col1, col2 = st.columns([1, 8])
with col1:
    st.image("senegal icone.jpg", width=50)
with col2:
    st.title("📊 Application de Prévision par Région")

# Texte descriptif sous le titre
st.markdown(
    """
    ### Prévision de l'évolution de la population du Sénégal par région  
    *en utilisant le Deep Learning (Temporal Convolutional Neural Network)*  
    """
)

API_URL = "https://ansdpoc1-dgid.apps.ocp.heritage.africa/v2/models/ansdpoc1/infer"

@st.cache_data
def load_data():
    df = pd.read_csv("DATA.csv")
    df.columns = ['indicateur', 'region', 'sexe', 'unit', 'date', 'value']
    df = df[['region', 'date', 'value']]
    df['region'] = df['region'].str.upper().str.strip()
    df = df.sort_values(['region', 'date'])
    return df

df = load_data()
regions = sorted(df['region'].unique())

st.sidebar.header("Filtres Régionaux")
default_selection = ["DAKAR"] if "DAKAR" in regions else ([regions[0]] if regions else [])
selected_regions = st.sidebar.multiselect(
    "Sélectionnez une ou plusieurs régions",
    options=regions,
    default=default_selection,
)

st.sidebar.header("Période de prédiction")
start_year = st.sidebar.number_input("Année de début", min_value=2024, max_value=2100, value=2025)
end_year = st.sidebar.number_input("Année de fin", min_value=start_year, max_value=2100, value=2030)

if not selected_regions:
    st.warning("Veuillez sélectionner au moins une région dans la sidebar.")
    st.stop()

def predict_for_region(region_name):
    region_df = df[df['region'] == region_name]
    if region_df.empty:
        st.warning(f"Aucune donnée historique pour {region_name}")
        return None, None

    seq_length = 5
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(region_df['value'].values.reshape(-1, 1)).flatten()

    region_map = {region: idx for idx, region in enumerate(regions)}
    region_code = region_map[region_name]

    input_seq = scaled_values[-seq_length:].astype(np.float32).reshape(1, seq_length)
    predictions = []

    try:
        for year in range(start_year, end_year + 1):
            payload = {
                "inputs": [
                    {
                        "name": "sequence",
                        "shape": list(input_seq.shape),
                        "datatype": "FP32",
                        "data": input_seq.flatten().tolist()
                    },
                    {
                        "name": "region_code",
                        "shape": [1],
                        "datatype": "INT64",
                        "data": [int(region_code)]
                    }
                ]
            }
            response = requests.post(API_URL, json=payload, timeout=20, verify=False)
            response.raise_for_status()

            result = response.json()
            pred_norm = result["outputs"][0]["data"][0]
            pred = scaler.inverse_transform([[pred_norm]])[0][0]
            predictions.append((year, pred))

            input_seq = np.roll(input_seq, -1, axis=1)
            input_seq[0, -1] = pred_norm

        df_pred = pd.DataFrame(predictions, columns=["Année", "Prédiction (valeur)"])
        return region_df, df_pred

    except Exception as e:
        st.error(f"Erreur lors de la prédiction pour {region_name} : {e}")
        return region_df, None

with st.spinner("⏳ Calcul des prévisions..."):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for region in selected_regions:
        hist_df, pred_df = predict_for_region(region)
        if hist_df is not None:
            ax.plot(hist_df['date'], hist_df['value'], marker='o', label=f"{region} - Historique")
        if pred_df is not None:
            ax.plot(pred_df['Année'], pred_df['Prédiction (valeur)'], marker='x', linestyle='--', label=f"{region} - Prédiction")

    ax.set_xlabel("Année")
    ax.set_ylabel("Valeur")
    ax.set_title("Données historiques et prévisions par région")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
