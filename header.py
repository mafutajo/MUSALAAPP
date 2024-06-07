import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_modal import Modal
import extra_streamlit_components as stx

import streamlit as st
from PIL import Image

from streamlit.components.v1 import html

import yaml
import pandas as pd
from yaml.loader import SafeLoader

import altair as alt

import itertools
import docxpy
import pandas as pd
import re
import unidecode
import string
import streamlit as st
import io
import random
import streamlit.components.v1 as components
import base64
import json

import hashlib
import numpy as np
import datetime
from numpy.linalg import norm
from simpletransformers.ner import NERModel, NERArgs
from spacy.lang.fr import French
from fuzzywuzzy import fuzz
from gensim.models import Word2Vec
from annotated_text import annotated_text
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from dateparser.search import search_dates
import streamlit as st
import math
import joblib

import fitz
import os


# Créer un dégradé de gris clair à blanc cassé avec Altair


def header() -> str:
    header_style = """
    <style>
        .musala-headerx {
            font-size: 5vw; /* Taille de la police */
            font-weight: 600; /* Épaisseur de la police */
            color: #302838; /* Couleur de la police */
            text-align: center; /* Alignement du texte */
        }
        .subtitle {
            text-align: center; /* Alignement du texte pour le sous-titre */
            color: #34495E; /* Couleur plus douce pour le sous-titre */
            font-size: 24px; /* Taille de la police pour le sous-titre */
        }
    </style>
    """

    # Utilisation du style défini pour l'en-tête et un sous-titre
    st.markdown(header_style, unsafe_allow_html=True)
    # st.logo("nest_logo-transformed.png")
    st.markdown(
        '<div class="musala-headerx">NEST</div>',
        unsafe_allow_html=True,
    )
    if "menu_option_integer" not in st.session_state:
        st.session_state["menu_option_integer"] = 0

    st.markdown(
        """
    <style>
        body {
            background: linear-gradient(to bottom, #f0f0f0, #ffffff);
        }
    </style>
    """,
        unsafe_allow_html=True,
    )

    hide_streamlit_style = """
                    <style>
                    zoom: 0.8;
                    div[data-testid="stToolbar"] {
                    visibility: hidden;
                    height: 0%;
                    position: fixed;
                    }
                    div[data-testid="stDecoration"] {
                    visibility: hidden;
                    height: 0%;
                    position: fixed;
                    }
                    div[data-testid="stStatusWidget"] {
                    visibility: hidden;
                    height: 0%;
                    position: fixed;
                    }
                    #MainMenu {
                    visibility: hidden;
                    height: 0%;
                    }
                    header {
                    visibility: hidden;
                    height: 0%;
                    }
                    footer {
                    visibility: hidden;
                    height: 0%;
                    }
                    </style>
                    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    ############  SUPRESSION DES CHOSES VISIBLES ###########################

    ##### BARRE DE SELECTION
    col1, col2, col3 = st.columns((0.2, 0.6, 0.2))
    with col2:
        selected = option_menu(
            None,
            [
                "Accueil",
                "The Market",
                "Job Finder",
                "About Nest",
            ],
            icons=[
                "houe",
                "clou",
                "lik",
                "gr",
                "c",
            ],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "#fbebe3",
                    "width": "100%",
                    "margin": "0",
                    "padding": "0",
                },
                "icon": {"color": "orange", "font-size": "4vw"},
                "nav-link": {
                    "font-size": "2.5vw",
                    "font-weight": "lighter",
                    "background-color": "#fbebe3",
                    "text-align": "center",
                    "margin": "0px !important",
                    "--hover-color": "gray",
                },
                "nav-link-selected": {
                    "background-color": "#302838",
                    "text-align": "center",
                    "font-size": "2.3vw",
                },
            },
            manual_select=st.session_state["menu_option_integer"],
        )

    # st.markdown(
    #     "<h1 style='text-align: center; font-size:90px;color:rgba(56,181,165,255);'> <b> NEST </b> <span style='font-size:18px;color:rgb(167, 199, 231);'> by Lemniscate </span></h1>",
    #     unsafe_allow_html=True,
    # )
    return selected


def set_menu_option_integer(i: int):
    st.session_state["menu_option_integer"] = i


def action_left():
    st.session_state["menu_option_integer"] = 1
    st.write("Plus d'informations sur MISALA NETWORK.")


def action_right():
    st.write("Plus de détails sur le disclaimer.")
    st.session_state["menu_option_integer"] = 1


def set_background_color():
    css = """
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #fde9e5;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def accueil():
    set_background_color()
    st.markdown("#")
    st.markdown("#")
    st.markdown("#")

    def get_base64_of_bin_file(bin_file):
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    # Styles CSS pour aligner et styliser le contenu
    css = """
    <style>
        .container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 50px;
        }
        .text {
            text-align: left;
            width: 50%;
        }
        .image {
            width: 40%;
        }
        .title {
            font-size: 4vw;
            font-weight: bold;
            margin-bottom: 0.5em;
            color:#302838; 
        }
        .description {
            font-size: 4vw;
            margin-bottom: 2em;
            transition: transform 0.3s, background-color 0.1s;
            padding: 10px;
            border-radius: 5px;
            color:#302838; 
        }
        .description:hover {
            transform: scale(1.15);
            background-color: #d0c1c7; /* Couleur pastel */
            color:#302838; 
        }
        
        .bullet-points {
            list-style-type: disc;
            padding-left: 20px;
            font-size: 3vw;
            color:#302838; 
        }
    </style>
    """

    # Inclure les styles CSS dans Streamlit
    st.markdown(css, unsafe_allow_html=True)

    # Chemin de l'image téléchargée
    image_path = "./static/acceuil.png"
    image_base64 = get_base64_of_bin_file(image_path)

    # Contenu de l'analyse et du matching
    bullet_points = """
    <ul class="bullet-points">
        <li class="description">Outil d'analyse et de matching de CV basé sur les compétences.</li>
        <li class="description">Analyse des profils pour différents métiers.</li>
        <li class="description">Recommandations personnalisées pour les candidats et les recruteurs.</li>
    </ul>
    """

    # HTML pour structurer la page avec l'image et les bullet points
    html_content = f"""
    <div class="container">
        <div class="image">
            <img src="data:image/png;base64,{image_base64}" alt="Illustration" style="width: 100%;">
        </div>
        <div class="text">
            <div class="title">Recrutons le futur <span style='color:#47855b;'> ensemble</span></div>
            {bullet_points}
        
    </div>
    """

    # Afficher le contenu HTML dans Streamlit
    st.markdown(html_content, unsafe_allow_html=True)
