import io
import streamlit as st
import base64
from pdfminer.high_level import extract_text

st.set_page_config(
    page_title="NEST", layout="wide", page_icon="nest_logo-transformed.png"
)
from header import header, accueil
from matching import job_offer_parser

from PIL import Image

##### BARRE DE SELECTION
selected = header()


def resize_image(image_path, base_width):
    try:
        img = Image.open(image_path)
        w_percent = base_width / float(img.size[0])
        h_size = int((float(img.size[1]) * float(w_percent)))
        img = img.resize((base_width, h_size), Image.ANTIALIAS)
        return img
    except IOError:
        st.error(f"Erreur lors du chargement de l'image : {image_path}")
        return None


def page_presentation():
    st.markdown(
        """
        <style>
            .theme-nest1 {
                padding: 10px;
                border-radius: 10px;
            }
            .theme-nest2 {
                background-color: #F5F5DC; /* Changer selon le thème de NEST */
                padding: 10px;
                border-radius: 10px;
            }
            .img-description {
                text-align: center; /* Alignement du texte pour le sous-titre */
                color: #34495E; /* Couleur plus douce pour le sous-titre */
                font-size: 3.8vw; /* Taille de la police pour le sous-titre */
            }
            .img-descriptif{
                text-align: center; /* Alignement du texte pour le sous-titre */
                color: #34495E; /* Couleur plus douce pour le sous-titre */
                font-size: 3.4vw; /* Taille de la police pour le sous-titre */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Section logo et descriptio

    col1, col2 = st.columns([2, 3])
    with col1:
        st.image("nest_logo.png", use_column_width=True)
    with col2:
        st.markdown("#")
        st.markdown("#")

        st.markdown(
            """
           <div class="img-descriptif">
    <p><b>Bienvenue chez NEST,</b> votre partenaire novateur pour une gestion de carrière révolutionnée. NEST n'est pas simplement une plateforme ; c'est une révolution dans le monde du recrutement et du développement professionnel, propulsée par l'intelligence artificielle.</p>
    <p>En s'appuyant sur des algorithmes avancés, <b>NEST personnifie votre expérience</b> de recherche d'opportunités, en alignant précisément vos compétences et aspirations avec les besoins du marché. Notre objectif ? Vous guider vers votre prochain défi professionnel avec une précision inégalée.</p>
    <p><b>Construit en collaboration avec des experts du recrutement tech</b> et des leaders d'Entreprises de Services du Numérique (ESN), NEST apporte une connaissance approfondie du secteur pour transformer les processus de recrutement et de gestion de carrière. Nous associons expertise humaine et puissance de l'IA pour offrir des solutions sur mesure.</p>
    <p><b>Prochaines étapes :</b> Nous ne nous arrêtons pas là. L'innovation est au cœur de notre ADN. Nous continuons à développer des fonctionnalités avancées, comme l'amélioration du scoring et l'analyse prédictive, pour mieux anticiper les tendances du marché et affiner nos recommandations.</p>
    <p>Rejoignez NEST pour vivre une expérience unique, où technologie et expertise humaine se conjuguent pour façonner l'avenir de votre carrière. L'aventure ne fait que commencer.</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns([2, 3])
    with col1:
        st.markdown(
            """
            <h1 style="text-align:center;font-size: 2.4vw;color: #34495E;">
            <em>Lead by the Data</em>
            </h1>
            """,
            unsafe_allow_html=True,
        )

        st.image("photo.jpeg", use_column_width=True)
    with col2:
        st.markdown("#")
        st.markdown("#")
        st.markdown("#")
        st.markdown("#")
        st.markdown("#")
        st.markdown("#")

        st.markdown(
            """
        <style>
            .img-descriptif p {
                font-size: 18px;
            }
            .img-descriptif b {
                font-weight: bold;
            }
        </style>
        <div class="img-descriptif">
            <p> <b>Data Leader</b> dans la création de  <b>NEST</b><br>
            Le recrutement est un enjeu crucial dans un univers de travail en constante évolution.</p>
            <p><b>Expertise :</b> Ma spécialisation en <b>science des données</b> et <b>intelligence artificielle</b> me permet d'apporter une valeur ajoutée significative à travers des solutions innovantes pour optimiser les processus de recrutement.</p>
            <p><b>Prochaines étapes :</b> Je travaille sur l'intégration de l'intelligence artificielle avancée pour personnaliser encore plus l'expérience de nos clients et fournir des recommandations sur mesure basées sur l'analyse prédictive et le traitement du langage naturel.</p>
            <p>Pour plus d'informations ou pour discuter d'opportunités, n'hésitez pas à me contacter :</p>
            <ul>
                <li>Email : <a href="mailto:lemniscatedata@gmail.com">lemniscatedata@gmail.com</a></li>
                <li>LinkedIn : <a href="https://www.linkedin.com/in/geoffret-tuyindi-mafuta-40801a150/" target="_blank">Contact LinkedIn</a></li>
            </ul>
        </div>
            """,
            unsafe_allow_html=True,
        )


# Appeler la fonction pour afficher la page
# ########### PAGE DE GESTION ########################
if selected == "Accueil":
    accueil()

if selected == "NEST IA":
    job_offer_parser(selected)

if selected == "Qui sommes nous?":

    page_presentation()
