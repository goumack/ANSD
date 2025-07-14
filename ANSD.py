import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prévision Régionale", layout="wide")

# === ENTÊTE AVEC IMAGE ET TITRE ===
col1, col2 = st.columns([1, 8])
with col1:
    st.image("senegal icone.jpg", width=50)
with col2:
    st.title("Application de Prévision par Région")

st.markdown(
    """
    ### Prévisions de l'évolution au Sénégal par région  
    *avec des modèles de Deep Learning - ACCEL 2025*
    """
)

# === ONGLET PRINCIPAL ===
tab1, tab2, tab3 = st.tabs([" Population", " Structure nombre", " Prévision temporelle"])

# ==================================================================================
# === TAB 1 : PRÉVISION DE LA POPULATION PAR RÉGION ===============================
# ==================================================================================
with tab1:
    st.subheader(" Prévision de la Population par Région")
    
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
    selected_regions = st.sidebar.multiselect("Sélectionnez une ou plusieurs régions", regions, default=default_selection)

    st.sidebar.header("Période de prédiction")
    start_year = st.sidebar.number_input("Année de début", min_value=2024, max_value=2100, value=2025)
    end_year = st.sidebar.number_input("Année de fin", min_value=start_year, max_value=2100, value=2030)

    if not selected_regions:
        st.warning("Veuillez sélectionner au moins une région.")
        st.stop()

    def predict_for_region(region_name):
        region_df = df[df['region'] == region_name]
        if region_df.empty:
            st.warning(f"Aucune donnée pour {region_name}")
            return None, None

        seq_length = 5
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(region_df['value'].values.reshape(-1, 1)).flatten()
        region_code = {region: idx for idx, region in enumerate(regions)}[region_name]

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
            st.error(f"Erreur pour {region_name} : {e}")
            return region_df, None

    with st.spinner("Calcul des prévisions..."):
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

# ==================================================================================
# === TAB 2 : STRUCTURE NOMBRE =====================================================
# ==================================================================================
with tab2:
    st.subheader(" Structure Nombre")

    SEQ_LENGTH2 = 10
    DATA_PATH2 = "couverturedf2.csv"
    API_URL2 = "https://structurenombre-dgid.apps.ocp.heritage.africa/v2/models/structurenombre/infer"

    @st.cache_data
    def load_data_struct():
        df = pd.read_csv(DATA_PATH2, sep=';')
        df.columns = ['region', 'date', 'unit', 'value', 'indicateur', 'sexe']
        df = df[['region', 'date', 'value']]
        df['region'] = df['region'].str.upper().str.strip()
        df = df.sort_values(['region', 'date'])
        return df

    df_struct = load_data_struct()
    region_map2 = {region: idx for idx, region in enumerate(sorted(df_struct['region'].unique()))}
    region_selected2 = st.selectbox("Sélectionnez une région", list(region_map2.keys()), key="region_struct")
    start_year2 = st.number_input("Année de début", 2018, 2100, 2020, key="start_struct")
    end_year2 = st.number_input("Année de fin", start_year2, 2100, 2030, key="end_struct")

    def predict_struct(scaled_values, scaler, region_code):
        input_seq = scaled_values[-SEQ_LENGTH2:].astype(np.float32).reshape(1, SEQ_LENGTH2)
        predictions = []
        for year in range(start_year2, end_year2 + 1):
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
            response = requests.post(API_URL2, json=payload)
            result = response.json()
            pred_norm = result["outputs"][0]["data"][0]
            pred = scaler.inverse_transform([[pred_norm]])[0][0]
            predictions.append((year, pred))
            input_seq = np.roll(input_seq, -1, axis=1)
            input_seq[0, -1] = pred_norm
        return pd.DataFrame(predictions, columns=["Année", "Prédiction (valeur)"])

    region_df2 = df_struct[df_struct['region'] == region_selected2]
    scaler2 = MinMaxScaler()
    scaled2 = scaler2.fit_transform(region_df2['value'].values.reshape(-1, 1)).flatten()

    try:
        df_pred2 = predict_struct(scaled2, scaler2, region_map2[region_selected2])
        st.dataframe(df_pred2)
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(df_pred2['Année'], df_pred2['Prédiction (valeur)'], marker="x", linestyle="--", label="Prédiction")
        ax2.set_title(f"Structure nombre - {region_selected2}")
        ax2.set_xlabel("Année")
        ax2.set_ylabel("Valeur")
        ax2.grid()
        ax2.legend()
        st.pyplot(fig2)
    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")

# ==================================================================================
# === TAB 3 : PRÉVISION TEMPORELLE ================================================
# ==================================================================================
with tab3:
    st.subheader(" Prévision Temporelle par Région")

    SEQ_LENGTH3 = 10
    DATA_PATH3 = "couverturedf1.csv"
    API_URL3 = "https://temporeans-dgid.apps.ocp.heritage.africa/v2/models/temporeans/infer"

    @st.cache_data
    def load_temporelle_data():
        df = pd.read_csv(DATA_PATH3, sep=';')
        df.columns = ['region', 'date', 'value', 'unit', 'indicateur', 'sexe']
        df = df[['region', 'date', 'value']]
        df['region'] = df['region'].str.upper().str.strip()
        df = df.sort_values(['region', 'date'])
        return df

    df_temp = load_temporelle_data()
    region_map3 = {region: idx for idx, region in enumerate(sorted(df_temp['region'].unique()))}

    region_selected3 = st.selectbox("Choisissez une région", list(region_map3.keys()), key="region_temp")
    start_year3 = st.number_input("Année de début", 2018, 2100, 2020, key="start_temp")
    end_year3 = st.number_input("Année de fin", start_year3, 2100, 2030, key="end_temp")

    def predict_temporelle(scaled_values, scaler, region_code):
        input_seq = scaled_values[-SEQ_LENGTH3:].astype(np.float32).reshape(1, SEQ_LENGTH3)
        predictions = []
        for year in range(start_year3, end_year3 + 1):
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
            response = requests.post(API_URL3, json=payload)
            result = response.json()
            pred_norm = result["outputs"][0]["data"][0]
            pred = scaler.inverse_transform([[pred_norm]])[0][0]
            predictions.append((year, pred))
            input_seq = np.roll(input_seq, -1, axis=1)
            input_seq[0, -1] = pred_norm
        return pd.DataFrame(predictions, columns=["Année", "Prédiction (valeur)"])

    region_df3 = df_temp[df_temp['region'] == region_selected3]
    scaler3 = MinMaxScaler()
    scaled3 = scaler3.fit_transform(region_df3['value'].values.reshape(-1, 1)).flatten()

    try:
        df_pred3 = predict_temporelle(scaled3, scaler3, region_map3[region_selected3])
        st.dataframe(df_pred3)
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(df_pred3['Année'], df_pred3['Prédiction (valeur)'], marker="x", linestyle="--", label="Prédiction")
        ax3.set_title(f"Prévision temporelle - {region_selected3}")
        ax3.set_xlabel("Année")
        ax3.set_ylabel("Valeur")
        ax3.grid()
        ax3.legend()
        st.pyplot(fig3)
    except Exception as e:
        st.error(f"Erreur de prédiction : {e}")
