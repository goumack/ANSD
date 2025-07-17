# ...existing code...
        if df_pred_struct.empty:
            st.info("Aucune prévision de structures existantes possible pour cette région (données insuffisantes).")
        else:
            df_pred_struct["Structures existantes"] = np.ceil(df_pred_struct["Prédiction (valeur)"]).clip(lower=0).astype(int)
            df_pred_struct = df_pred_struct[["Année", "Structures existantes"]]

            df_pred_omvs = pd.merge(df_pred_omvs, df_pred_struct, on="Année", how="left")
            df_pred_omvs["Structures existantes"] = df_pred_omvs["Structures existantes"].fillna(0).astype(int)
            df_pred_omvs["À ajouter"] = (df_pred_omvs["Structures recommandées"] - df_pred_omvs["Structures existantes"]).clip(lower=0).astype(int)

            fig_omvs, ax_omvs = plt.subplots(figsize=(10, 5))
            ax_omvs.plot(
                df_pred_omvs["Année"],
                y_values,
                marker="o",
                markersize=8,
                linestyle="-",
                color="green",
                label="Structures recommandées"
            )
            ax_omvs.set_title(f"Structures recommandées par OMS - {region_selected_omvs}")
            ax_omvs.set_xlabel("Année")
            ax_omvs.set_ylabel("Nombre de structures")
            ax_omvs.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
            y_max = y_values.max() if len(y_values) > 0 else 1
            ax_omvs.set_ylim(bottom=0, top=y_max + max(2, int(0.1 * y_max)))
            ax_omvs.grid()
            ax_omvs.legend()
            st.pyplot(fig_omvs)
            st.dataframe(df_pred_omvs[["Année", "Structures recommandées", "Structures existantes", "À ajouter"]])
            csv_omvs = df_pred_omvs[["Année", "Structures recommandées", "Structures existantes", "À ajouter"]].to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Télécharger les recommandations en CSV",
                data=csv_omvs,
                file_name=f"recommandations_oms_{region_selected_omvs}.csv",
                mime="text/csv"
            )
    else:
        st.warning("Impossible de calculer la recommandation pour cette region.")
# ...existing code...
