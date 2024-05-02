import io
import streamlit as st
import base64
from pdfminer.high_level import extract_text

import matplotlib.pyplot as plt

st.set_page_config(
    page_title="NEST", layout="wide", page_icon="nest_logo-transformed.png"
)
from header import header, accueil
from matching import job_offer_parser, test_cv_page, affichage_analyse

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


def page_presentation_2():
    col1, col2, col3 = st.columns([1, 2, 1])
    st.markdown("#")
    st.markdown("#")
    with col1:
        st.image("nest_logo.png", use_column_width=True)
    st.markdown(
        """
        <style>
            .container {
                display: flex;
                justify-content: space-around;
                align-items: center;
                padding: 20px;
                background-color: #f5f5f5; /* Couleur de fond légère */
                border-radius: 10px; /* Bords arrondis */
                margin-bottom: 20px;
            }
            .profile {
                text-align: center; /* Centrer le texte et l'image dans le div */
                width: 30%; /* Définir la largeur pour chaque profil */
            }
            .profile img {
                border-radius: 50%; /* Rendre les images rondes */
                width: 100px; /* Taille des images */
                height: 100px; /* Hauteur fixe pour les images */
            }
            .title {
                text-align: center;
                font-size: 24px;
                color: #34495e; /* Couleur du texte */
            }
        </style>

      

        <div class="title">Rencontrer l'équipe</div>
        <div class="container">
            <div class="profile">
                <img src="path_to_member1_image.jpg" alt="Member 1">
                <h3>Nom du Membre 1</h3>
                <p>Position du Membre 1</p>
                <p>Email : <a href="mailto:email1@example.com">email1@example.com</a></p>
                <p>LinkedIn : <a href="https://www.linkedin.com/in/link1" target="_blank">Profil LinkedIn</a></p>
            </div>
            <div class="profile">
                <img src="path_to_member2_image.jpg" alt="Member 2">
                <h3>Nom du Membre 2</h3>
                <p>Position du Membre 2</p>
                <p>Email : <a href="mailto:email2@example.com">email2@example.com</a></p>
                <p>LinkedIn : <a href="https://www.linkedin.com/in/link2" target="_blank">Profil LinkedIn</a></p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_presentation():
    st.markdown("#")
    st.markdown("#")
    st.markdown("#")
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
                background-color: #F5F5F5; /* Couleur de fond gris beige */
                padding: 20px;
                border-radius: 10px;
                margin: 20px;
            }
            .side-column {
               background-color: #2980B9; /* Dark blue color */
               height: 100vh; /* Full height of the viewport */
           }

        </style>
        """,
        unsafe_allow_html=True,
    )

    # Section logo et descriptio

    col1, col2, col3, col4, col5, col6, col7 = st.columns(
        [1, 1, 4, 1, 4, 1, 1], gap="large"
    )

    with col3:
        st.image("nest_logo.png", use_column_width=True)

    st.markdown("#")
    with col5:
        st.markdown(
            """
        <style>
            .custom-container {
                background-color: #fff; /* Fond blanc */
                border-left: 5px solid rgb(113,224,203); /* Bordure violette à gauche */
                padding: 20px;
                margin: 10px 0;
                font-family: Arial, sans-serif; /* Police moderne */
            }
            .custom-header {
                color:  #2874A6; /* Couleur du titre */
                font-size: 24px; /* Taille du titre */
                font-weight: bold; /* Gras pour le titre */
                margin-bottom: 10px; /* Espacement après le titre */
            }
            .custom-list {
                color: #34495e;
                padding-left: 20px; /* Padding pour aligner avec le titre */
                list-style: none; /* Pas de puces classiques */
                margin-bottom: 0; /* Ajuster selon besoin */
            }
            .custom-list li {
                margin-bottom: 10px; /* Espacement entre les éléments */
                font-size: 1.0vw;/* Taille de police pour les éléments de liste */
                padding-bottom: 10px; /* Padding au bas de chaque élément */
                border-bottom: 1px solid #ddd; /* Ligne de séparation */
            }
            .custom-list li:last-child {
                border-bottom: none; /* Pas de bordure au dernier élément */
            }
            .custom-list li:before {
                content: "✨"; /* Utilisation d'émojis ou autre symbole comme puce */
                color: #9b59b6; /* Couleur des puces */
                font-size: 18px; /* Taille des puces */
                line-height: 18px;
                margin-right: 10px; /* Espacement après la puce */
                vertical-align: middle; /* Alignement vertical des puces */
            }
        </style>
        
        <div class="custom-container">
            <div class="custom-header">Talent meets tech</div>
            <ul class="custom-list">
                <li>Connaissance <b>approfondie du marché </b>et <b>détection des tendances </b>en matière de recrutement technologiques</li>
                <li>Valorisation <b>claire des profils</b> grâce à des outils de présélection et de sélection efficaces dans le processus de recrutement.</li>
                <li>Optimiser le temps consacré au recrutement ou au placement de profils techniques. </li>
                <li>Optimisation continue des <b>processus de recrutement</b> pour mieux répondre aux évolutions du marché.</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )
    st.markdown(
        """
        <style>
            .custom-title2 {
                font-family: 'Helvetica Neue', Arial, sans-serif; /* Police moderne et propre */
                color: #2874A6; /* Bleu moderne, mais vous pouvez choisir n'importe quelle couleur */
                font-size: 2.2vw; /* Taille de la police grande pour plus d'impact */
                text-align: center; /* Centrer le titre */
                margin-top: 20px; /* Espacement du haut pour aérer */
                margin-bottom: 20px; /* Espacement du bas pour aérer */
            }
        </style>
        <h1 class="custom-title2">Equipe</h1> <!-- Titre personnalisé ici -->
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5, col6, col7 = st.columns(
        [1, 1, 4, 1, 4, 1, 1], gap="large"
    )
    st.markdown("#")
    with col3:
        st.image("photo.jpeg", use_column_width=True)
    with col3:
        st.markdown(
            """
        <style>
            .img-descriptif p {
                font-size: 1.9vw;
            }
            .img-descriptif b {
                font-weight: bold;
            }
        </style>
        
        <div class="img-descriptif">
            <p style='font-size:1.3vw;'> <b>TUYINDI MAFUTA Geoffret</b> </p>
            <p style='font-size:1.3vw;font-weight:lighter;'> Data scientist</p>
            <ul>
                <li>Email : <a href="mailto:lemniscatedata@gmail.com">lemniscatedata@gmail.com</a></li>
                <li>LinkedIn : <a href="https://www.linkedin.com/in/geoffret-tuyindi-mafuta-40801a150/" target="_blank">Contact LinkedIn</a></li>
            </ul>
        </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.image("aravinth.jpeg", use_column_width=True)
    with col5:

        st.markdown(
            """
        <style>
            .img-descriptif p {
                font-size: 1.9vw;
            }
            .img-descriptif b {
                font-weight: bold;
            }
        </style>
        
        <div class="img-descriptif">
            <p style='font-size:1.3vw;'> <b>BALAKRISHNAN Aravinth </b> </p>
            <p style='font-size:1.3vw;font-weight:lighter;'> Data engineer</p>
            <ul>
                <li>Email : <a href="mailto:lemniscatedata@gmail.com">lemniscatedata@gmail.com</a></li>
                <li>LinkedIn : <a href="https://www.linkedin.com/in/aravinth-balakrishnan/" target="_blank">Contact LinkedIn</a></li>
            </ul>
        </div>
            """,
            unsafe_allow_html=True,
        )


# Appeler la fonction pour afficher la page
# ########### PAGE DE GESTION ########################
if selected == "Accueil":
    accueil()

if selected == "Job Finder":
    job_offer_parser()

if selected == "Job Fit":
    test_cv_page()
if selected == "About Nest":
    page_presentation()

if selected == "The Market":
    affichage_analyse("JOB_ANALYSIS.json")
