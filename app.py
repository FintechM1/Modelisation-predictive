import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime

st.set_page_config(page_title="ISM Credit Scoring", page_icon="🏦", layout="wide")

# ── CSS ISM (Marron & Or) ──────────────────────────────────────────
st.markdown("""
<style>
:root { --marron: #5C2D0E; --or: #C9A84C; --or-light: #F0D080; --beige: #FAF6F0; }
html, body, [class*="css"] { background-color: var(--beige) !important; }

.header {
    background: linear-gradient(135deg, #3B1A06, #5C2D0E);
    border-radius: 14px; padding: 2rem; text-align: center; margin-bottom: 2rem;
    box-shadow: 0 6px 24px rgba(92,45,14,0.3);
}
.header h1 { color: #F0D080 !important; font-size: 1.9rem; margin: 0.3rem 0; }
.header p  { color: rgba(255,255,255,0.7); margin: 0; font-size: 0.9rem; }
.badge {
    background: #C9A84C; color: #3B1A06; font-weight: 700;
    font-size: 0.7rem; padding: 3px 12px; border-radius: 20px;
    letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 0.6rem; display: inline-block;
}
.card {
    background: white; border-radius: 12px; padding: 1.4rem; margin-bottom: 1rem;
    border: 1px solid rgba(201,168,76,0.3); box-shadow: 0 2px 10px rgba(92,45,14,0.07);
}
.card-title { color: #5C2D0E; font-weight: 700; font-size: 1rem; border-bottom: 2px solid #C9A84C; padding-bottom: 0.4rem; margin-bottom: 1rem; }
.result-ok { background: #E8F5EE; border: 2px solid #1A7A3C; border-radius: 12px; padding: 1.5rem; text-align: center; }
.result-ok h2 { color: #1A7A3C; font-size: 1.7rem; margin: 0.3rem 0; }
.result-ko { background: #FEE2E2; border: 2px solid #9B1C1C; border-radius: 12px; padding: 1.5rem; text-align: center; }
.result-ko h2 { color: #9B1C1C; font-size: 1.7rem; margin: 0.3rem 0; }
.score-box {
    background: linear-gradient(135deg, #5C2D0E, #7A3B10);
    border-radius: 12px; padding: 1.2rem; text-align: center; color: white; margin-top: 1rem;
}
.score-value { font-size: 3rem; color: #F0D080; line-height: 1; font-weight: 700; }
.score-label { font-size: 0.8rem; opacity: 0.8; margin-top: 4px; }
.gauge-bar {
    height: 18px; border-radius: 9px; margin: 0.5rem 0;
    background: linear-gradient(90deg, #1A7A3C 0%, #D4A017 50%, #9B1C1C 100%);
    position: relative;
}
.gauge-cursor {
    position: absolute; top: -6px; width: 30px; height: 30px;
    background: white; border: 3px solid #5C2D0E; border-radius: 50%;
    transform: translateX(-50%); box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.gauge-labels { display: flex; justify-content: space-between; font-size: 0.72rem; color: #888; }
.hist-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.6rem 1rem; border-radius: 8px; margin-bottom: 5px;
    background: #F2EDE6; border-left: 4px solid #C9A84C; font-size: 0.85rem;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #5C2D0E, #7A3B10) !important;
    color: #F0D080 !important; border: none !important; font-weight: 700 !important;
    font-size: 1rem !important; padding: 0.7rem 2rem !important;
    border-radius: 50px !important; width: 100%;
    box-shadow: 0 4px 14px rgba(92,45,14,0.3) !important;
}
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #3B1A06, #5C2D0E) !important; }
section[data-testid="stSidebar"] * { color: rgba(255,255,255,0.9) !important; }
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label { color: #5C2D0E !important; font-weight: 600 !important; font-size: 0.87rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
    <div class="badge">ISM Dakar · MBA1 Finance Digitale</div>
    <h1>🏦 Système de Scoring Crédit</h1>
    <p>Régression Logistique · AUC = 0.9999 · Année 2025-2026</p>
</div>
""", unsafe_allow_html=True)

# ── Navigation sidebar ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 ISM Credit AI")
    st.markdown("---")
    page = st.radio("Navigation", ["📊 Prédiction", "📋 Historique", "ℹ️ À propos"])
    st.markdown("---")
    st.caption("Prof. M. K. CHOKKI · ISM Dakar · 2025-2026")

# ── Session state ──────────────────────────────────────────────────
if "historique" not in st.session_state:
    st.session_state.historique = []

# ── Chargement modèle ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists("credit_scoring_model.pkl"):
        return joblib.load("credit_scoring_model.pkl")
    return None

model = load_model()

# ══════════════════════════════════════════════════════════════════
#  PAGE 1 — PRÉDICTION
# ══════════════════════════════════════════════════════════════════
if page == "📊 Prédiction":

    if model is None:
        st.error("Modèle introuvable. Placez `credit_scoring_model.pkl` à la racine du projet.")
        st.stop()

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown('<div class="card"><div class="card-title">👤 Profil du Client</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            genre = st.selectbox("Genre", ["M", "F"])
            situation_matrimoniale = st.selectbox("Situation Matrimoniale", ["Célibataire", "Marié(e)", "Divorcé(e)", "Veuf/Veuve"])
        with c2:
            region = st.selectbox("Région", ["Dakar", "Thiès", "Saint-Louis", "Ziguinchor", "Kaolack", "Diourbel", "Tambacounda", "Louga", "Fatick", "Kolda", "Matam", "Kaffrine", "Kédougou", "Sédhiou"])
            type_logement = st.selectbox("Type de Logement", ["Propriétaire", "Locataire", "Hébergé", "Autre"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">💰 Informations Financières</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            ratio_endettement = st.number_input("Ratio d'Endettement (%)", min_value=0.0, max_value=100.0, value=30.0, step=0.5)
            secteur_activite = st.selectbox("Secteur d'Activité", ["Commerce", "Agriculture", "BTP", "Finance & Banque", "Santé", "Éducation", "Technologie", "Transport", "Industrie", "Autre"])
        with c4:
            taille_entreprise = st.selectbox("Taille de l'Entreprise", ["TPE", "PME", "Grande entreprise", "Multinationale"])
            garantie = st.selectbox("Garantie", ["Oui", "Non"])
        c5, c6 = st.columns(2)
        with c5:
            type_carte = st.selectbox("Type de Carte", ["Visa", "Mastercard", "American Express", "Aucune"])
        with c6:
            type_compte = st.selectbox("Type de Compte", ["Courant", "Épargne", "Joint", "Professionnel"])
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button("🔍 Prédire le Risque de Crédit")

    with col_result:
        if predict_btn:
            input_data = pd.DataFrame([{
                "RATIO_ENDETTEMENT":      ratio_endettement / 100,
                "SECTEUR_ACTIVITE":       secteur_activite,
                "GENRE":                  genre,
                "REGION":                 region,
                "TAILLE_ENTREPRISE":      taille_entreprise,
                "TYPE_CARTE":             type_carte,
                "GARANTIE":               garantie,
                "SITUATION_MATRIMONIALE": situation_matrimoniale,
                "TYPE_LOGEMENT":          type_logement,
                "TYPE_COMPTE":            type_compte,
            }])

            try:
                proba      = model.predict_proba(input_data)[0][1]
                prediction = model.predict(input_data)[0]
            except Exception as e:
                st.error(f"Erreur : {e}")
                st.stop()

            score     = int((1 - proba) * 1000)
            proba_pct = round(proba * 100, 1)
            decision  = "Accordé" if prediction == 0 else "Refusé"

            if decision == "Accordé":
                st.markdown(f'<div class="result-ok"><div style="font-size:2.5rem">✅</div><h2>Crédit Accordé</h2><p style="color:#166534;margin:0">Probabilité de défaut : <strong>{proba_pct}%</strong></p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-ko"><div style="font-size:2.5rem">❌</div><h2>Crédit Refusé</h2><p style="color:#7f1d1d;margin:0">Probabilité de défaut : <strong>{proba_pct}%</strong></p></div>', unsafe_allow_html=True)

            st.markdown(f'<div class="score-box"><div class="score-value">{score}</div><div class="score-label">SCORE DE RISQUE SUR 1 000</div></div>', unsafe_allow_html=True)

            cursor = max(2, min(98, 100 - (score / 10)))
            niveau = "🟢 Faible" if score >= 700 else "🟡 Modéré" if score >= 400 else "🔴 Élevé"
            st.markdown(f"""
            <div style="margin:1rem 0">
                <div style="font-size:0.85rem;font-weight:600;color:#5C2D0E;margin-bottom:4px">Niveau de risque : {niveau}</div>
                <div class="gauge-bar"><div class="gauge-cursor" style="left:{cursor}%"></div></div>
                <div class="gauge-labels"><span>Faible</span><span>Modéré</span><span>Élevé</span></div>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Prob. défaut", f"{proba_pct}%")
            c2.metric("Score", f"{score}/1000")
            c3.metric("Décision", decision)

            st.session_state.historique.insert(0, {
                "heure": datetime.now().strftime("%H:%M:%S"),
                "score": score, "proba": proba_pct, "decision": decision,
                "secteur": secteur_activite, "region": region,
            })
        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:3rem;border:2px dashed rgba(201,168,76,0.4)">
                <div style="font-size:3rem">🏦</div>
                <p style="color:#5C2D0E;font-weight:600;font-size:1.05rem;margin-top:1rem">
                    Renseignez le formulaire<br>et cliquez sur « Prédire »
                </p>
            </div>""", unsafe_allow_html=True)


#  PAGE 2 — HISTORIQUE
elif page == "📋 Historique":

    if not st.session_state.historique:
        st.info("Aucune prédiction effectuée. Allez sur la page **Prédiction** pour commencer.")
    else:
        df = pd.DataFrame(st.session_state.historique)

        st.markdown('<div class="card"><div class="card-title">📊 Résumé</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", len(df))
        c2.metric("✅ Accordés", (df["decision"] == "Accordé").sum())
        c3.metric("❌ Refusés", (df["decision"] == "Refusé").sum())
        c4.metric("Score moyen", f"{int(df['score'].mean())}/1000")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🗂️ Prédictions</div>', unsafe_allow_html=True)
        for h in st.session_state.historique:
            color = "#1A7A3C" if h["decision"] == "Accordé" else "#9B1C1C"
            icon  = "✅" if h["decision"] == "Accordé" else "❌"
            st.markdown(f"""
            <div class="hist-row">
                <span>{icon} <strong style="color:{color}">{h['decision']}</strong></span>
                <span>Score : <strong>{h['score']}/1000</strong></span>
                <span>Défaut : {h['proba']}%</span>
                <span>{h.get('secteur','—')}</span>
                <span style="color:#888">{h['heure']}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Exporter CSV", df.to_csv(index=False).encode("utf-8"), "historique.csv", "text/csv")
        with col2:
            if st.button("🗑️ Effacer l'historique"):
                st.session_state.historique = []
                st.rerun()



#  PAGE 3 — À PROPOS
elif page == "ℹ️ À propos":
 
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-title">🤖 Modèle</div>
            <p><strong>Algorithme :</strong> Régression Logistique</p>
            <p><strong>AUC-ROC :</strong> <span style="color:#1A7A3C;font-weight:700">0.9999</span></p>
            <p><strong>Pondération :</strong> class_weight='balanced'</p>
            <p><strong>Prétraitement :</strong> StandardScaler + OneHotEncoder</p>
            <p><strong>Sauvegarde :</strong> joblib (.pkl)</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-title">📂 Dataset</div>
            <p><strong>Lignes :</strong> 900 000</p>
            <p><strong>Variables :</strong> 47 colonnes</p>
            <p><strong>Cible :</strong> Défaut de paiement (0 / 1)</p>
            <p><strong>Split :</strong> 80% train / 20% test</p>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("""
    <div class="card">
        <div class="card-title">📊 Top 10 Variables du Modèle</div>
        <table width="100%" style="border-collapse:collapse;font-size:0.9rem">
            <tr style="background:#F2EDE6">
                <th style="padding:8px;text-align:left">#</th>
                <th style="padding:8px;text-align:left">Variable</th>
                <th style="padding:8px">Type</th>
                <th style="padding:8px">Importance</th>
            </tr>
            <tr><td style="padding:8px">1</td><td style="padding:8px">RATIO_ENDETTEMENT</td><td style="padding:8px;text-align:center"><code>Numérique</code></td><td style="padding:8px;text-align:center">⭐⭐⭐⭐⭐</td></tr>
            <tr style="background:#FAF6F0"><td style="padding:8px">2</td><td style="padding:8px">SECTEUR_ACTIVITE</td><td style="padding:8px;text-align:center"><code>Catégorielle</code></td><td style="padding:8px;text-align:center">⭐⭐⭐⭐⭐</td></tr>
            <tr><td style="padding:8px">3</td><td style="padding:8px">GENRE</td><td style="padding:8px;text-align:center"><code>Catégorielle</code></td><td style="padding:8px;text-align:center">⭐⭐⭐⭐</td></tr>
            <tr style="background:#FAF6F0"><td style="padding:8px">4</td><td style="padding:8px">REGION</td><td style="padding:8px;text-align:center"><code>Catégorielle</code></td><td style="padding:8px;text-align:center">⭐⭐⭐⭐</td></tr>
            <tr><td style="padding:8px">5</td><td style="padding:8px">TAILLE_ENTREPRISE</td><td style="padding:8px;text-align:center"><code>Catégorielle</code></td><td style="padding:8px;text-align:center">⭐⭐⭐</td></tr>
            <tr style="background:#FAF6F0"><td style="padding:8px">6</td><td style="padding:8px">TYPE_CARTE</td><td style="padding:8px;text-align:center"><code>Catégorielle</code></td><td style="padding:8px;text-align:center">⭐⭐⭐</td></tr>
            <tr><td style="padding:8px">7</td><td style="padding:8px">GARANTIE</td><td style="padding:8px;text-align:center"><code>Catégorielle</code></td><td style="padding:8px;text-align:center">⭐⭐⭐</td></tr>
            <tr style="background:#FAF6F0"><td style="padding:8px">8</td><td style="padding:8px">SITUATION_MATRIMONIALE</td><td style="padding:8px;text-align:center"><code>Catégorielle</code></td><td style="padding:8px;text-align:center">⭐⭐</td></tr>
            <tr><td style="padding:8px">9</td><td style="padding:8px">TYPE_LOGEMENT</td><td style="padding:8px;text-align:center"><code>Catégorielle</code></td><td style="padding:8px;text-align:center">⭐⭐</td></tr>
            <tr style="background:#FAF6F0"><td style="padding:8px">10</td><td style="padding:8px">TYPE_COMPTE</td><td style="padding:8px;text-align:center"><code>Catégorielle</code></td><td style="padding:8px;text-align:center">⭐</td></tr>
        </table>
    </div>
    <div class="card">
        <div class="card-title">📐 Interprétation du Score</div>
        <p>Formule : <code>Score = (1 − probabilité_défaut) × 1000</code></p>
        <table width="100%" style="border-collapse:collapse;font-size:0.9rem">
            <tr style="background:#F2EDE6"><th style="padding:8px;text-align:left">Score</th><th style="padding:8px">Risque</th><th style="padding:8px">Décision</th></tr>
            <tr><td style="padding:8px">700 – 1000</td><td style="padding:8px;text-align:center;color:#1A7A3C;font-weight:700">🟢 Faible</td><td style="padding:8px;text-align:center">Accordé</td></tr>
            <tr style="background:#FAF6F0"><td style="padding:8px">400 – 699</td><td style="padding:8px;text-align:center;color:#B45309;font-weight:700">🟡 Modéré</td><td style="padding:8px;text-align:center">Analyse</td></tr>
            <tr><td style="padding:8px">0 – 399</td><td style="padding:8px;text-align:center;color:#9B1C1C;font-weight:700">🔴 Élevé</td><td style="padding:8px;text-align:center">Refusé</td></tr>
        </table>
    </div>
    <div class="card">
        <div class="card-title">👥 Contexte Académique</div>
        <p><strong>Cours :</strong> Modélisation Prédictive et Intelligence Artificielle en Finance</p>
        <p><strong>Professeur :</strong> M. Komla Martin CHOKKI — Lead Data Strategist</p>
        <p><strong>Institution :</strong> ISM Dakar · Année 2025-2026</p>
        <p><strong>Technologies :</strong> Python · Scikit-Learn · Streamlit · GitHub</p>
    </div>
    """, unsafe_allow_html=True)