import io
import streamlit as st
import base64
from pdfminer.high_level import extract_text

st.set_page_config(
    page_title="MUSALA", layout="wide", page_icon="nest_logo-transformed.png"
)
from header import header, accueil
from matching import job_offer_parser


##### BARRE DE SELECTION
selected = header()


if selected == "Accueil":
    accueil()

if selected == "MUSALA":
    job_offer_parser(selected)

# ########### PAGE DE GESTION ########################

# if selected == "Talent Nest":
#     talent_nest(st.session_state["email"])
#     if "get_all_cv" not in st.session_state:
#         st.session_state["get_all_cv"] = get_all_cv(st.session_state["email"])

#     talent_nest(st.session_state["email"])
# ########### ANALYSE DE TALENT ########################
# if selected == "Analyse de Talent":
#     df_vizualization = df_analyse_cv_from_users(st.session_state["email"])
#     st.write(df_vizualization)
# ########### MATCHING ########################


# if selected == "Se déconnecter":
#     run_modal()
