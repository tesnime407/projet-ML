import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prédiction de la sortie du patient", page_icon="🏥")


@st.cache_resource
def charger():
    modele = joblib.load("model.joblib")
    with open("meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    return modele, meta


modele, meta = charger()
num, cat = meta["numeriques"], meta["categorielles"]

st.title("🏥 Prédiction de la sortie du patient")
st.write(
    "Cet outil estime si un patient diabétique hospitalisé rentrera **directement à domicile** "
    "ou aura besoin **d'une autre prise en charge**. "
    "⚠️ Domicile ne veut pas dire « guéri » : c'est la destination après l'hôpital."
)

valeurs = {}


def choix(nom, label):
    opts = cat[nom]["valeurs"]
    return st.selectbox(label, opts, index=opts.index(cat[nom]["defaut"]))


def nombre(nom, label):
    d = num[nom]
    return st.number_input(label, min_value=int(d["min"]), max_value=int(d["max"]),
                           value=int(d["defaut"]), step=1)


with st.form("formulaire"):
    st.subheader("Profil et admission")
    c1, c2 = st.columns(2)
    with c1:
        valeurs["age"] = choix("age", "Tranche d'âge")
        valeurs["gender"] = choix("gender", "Sexe")
        valeurs["race"] = choix("race", "Origine")
    with c2:
        valeurs["admission_type_id"] = choix("admission_type_id", "Type d'admission (code)")
        valeurs["admission_source_id"] = choix("admission_source_id", "Provenance (code)")
        valeurs["medical_specialty"] = choix("medical_specialty", "Spécialité médicale")

    st.subheader("Séjour")
    c3, c4 = st.columns(2)
    with c3:
        valeurs["time_in_hospital"] = nombre("time_in_hospital", "Durée du séjour (jours)")
        valeurs["num_lab_procedures"] = nombre("num_lab_procedures", "Nombre d'examens de labo")
        valeurs["num_procedures"] = nombre("num_procedures", "Nombre d'actes")
    with c4:
        valeurs["num_medications"] = nombre("num_medications", "Nombre de médicaments")
        valeurs["number_diagnoses"] = nombre("number_diagnoses", "Nombre de diagnostics")
        valeurs["diag_1_group"] = choix("diag_1_group", "Groupe du diagnostic principal")

    st.subheader("Historique (année précédente)")
    c5, c6, c7 = st.columns(3)
    with c5:
        valeurs["number_outpatient"] = nombre("number_outpatient", "Consultations")
    with c6:
        valeurs["number_emergency"] = nombre("number_emergency", "Urgences")
    with c7:
        valeurs["number_inpatient"] = nombre("number_inpatient", "Hospitalisations")

    st.subheader("Diabète")
    c8, c9 = st.columns(2)
    with c8:
        valeurs["A1Cresult"] = choix("A1Cresult", "Résultat HbA1c")
        valeurs["insulin"] = choix("insulin", "Insuline")
    with c9:
        valeurs["max_glu_serum"] = choix("max_glu_serum", "Glycémie sérique")
        valeurs["diabetesMed"] = choix("diabetesMed", "Médicament antidiabétique")

    envoyer = st.form_submit_button("Prédire")

if envoyer:
    ligne = {}
    for col in meta["columns"]:
        if col in valeurs:
            ligne[col] = valeurs[col]
        elif col in num:
            ligne[col] = num[col]["defaut"]
        else:
            ligne[col] = cat[col]["defaut"]
    X_new = pd.DataFrame([ligne], columns=meta["columns"])

    proba = modele.predict_proba(X_new)[0]
    classes = list(modele.classes_)
    prediction = classes[proba.argmax()]
    p_dom = float(proba[classes.index("Domicile")])

    if prediction == "Domicile":
        st.success(f"Sortie prédite : **Domicile** (probabilité {p_dom:.0%})")
    else:
        st.warning(f"Sortie prédite : **Autre prise en charge** (probabilité {1 - p_dom:.0%})")
    st.progress(p_dom, text=f"Probabilité de retour à domicile : {p_dom:.0%}")
    st.caption("Outil d'aide à la planification issu d'un projet de Machine Learning. "
               "Il ne remplace pas une décision médicale. Les champs non affichés "
               "prennent la valeur la plus fréquente du jeu de données.")
