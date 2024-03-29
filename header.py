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
            font-size: 4vw; /* Taille de la police */
            font-weight: 600; /* Épaisseur de la police */
            color: #1B4F72; /* Couleur de la police */
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
                "NEST IA",
                "Qui sommes nous?",
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
                    "background-color": "white",
                    "width": "100%",
                    "margin": "0",
                    "padding": "0",
                },
                "icon": {"color": "orange", "font-size": "4vw"},
                "nav-link": {
                    "font-size": "2.5vw",
                    "font-weight": "lighter",
                    "text-align": "center",
                    "margin": "0px !important",
                    "--hover-color": "gray",
                },
                "nav-link-selected": {
                    "background-color": "#1B4F72",
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


def accueil():
    st.markdown(
        """
        <style>
            body {
                background: linear-gradient(45deg, #ffffff, #f5f5f5); /* Dégradé de blanc à blanc cassé */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    voxlone, colonne1, colonne2 = st.columns([0.2, 0.8, 0.2], gap="small")
    with colonne1:
        # HTML/CSS personnalisé pour le reste de la page d'accueil
        st.markdown(
            """
                <style>
                    body {
                        background-color: #F4F4F4; /* Couleur de fond légère */
                        text-align: center;
                    }
                    .container {
                        max-width: 1000px;
                        margin: 25px auto;
                        padding: 20px;
                        background-color: #FFFFFF; /* Fond blanc pour le contenu */
                        border-radius: 10px;
                        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); /* Ombre douce */
                    }
                    .header, .explication, h1, h2, summary {
                        color: #34495E; /* Gris Anthracite pour les textes */
                    }
                    .explication {
                        background-color: #DAE8FC; /* Couleur subtile pour les sections d'explication */
                        padding: 20px;
                        border-radius: 10px;
                        margin-bottom: 20px;
                    }
                    .axe-analyse-explication {
                        display: flex;
                        flex-wrap: wrap;
                        
                        justify-content: space-between;
                    }
                    .axe-analyse {
                        flex: 1;
                        background-color: #DAE8FC;
                        margin: 10px;
                        font-size: 0.9em;
                        text-align: center;
                        /* Vert clair pour les axes d'analyse */
                        padding: 20px;
                        border-radius: 10px;
                    }
                    p {
                        color: #000000; /* Texte noir pour une meilleure lisibilité */
                    }
                </style>
                <div class="container">
                    <div class="header">
                        <h1>Optimisez votre parcours professionnel avec NEST</h1>
                        <p>Découvrez comment NEST valorise chaque aspect de votre profil pour une opportunité sur mesure.</p>
                    </div>
                    <div class="explication">
                        <h2>Les axes d'analyse NEST</h2>
                        <p>Chez NEST, nous utilisons des axes d'analyse spécifiques pour aligner vos compétences et aspirations avec les opportunités de carrière idéales.</p>
                    </div>
                    <div class="expander">
                        <details>
                            <summary>Explorez les axes d'analyse clés selon NEST</summary>
                            <div class="axe-analyse-explication">
                                <div class="axe-analyse">
                                    <h3>Formation Académique</h3>
                                    <p>Une mise en valeur de votre parcours éducatif souligne la base de vos connaissances et compétences.</p>
                                </div>
                                <div class="axe-analyse">
                                    <h3>Compétences Comportementales</h3>
                                    <p>Les soft skills sont cruciales pour démontrer votre capacité à évoluer dans divers environnements de travail.</p>
                                </div>
                                <div class="axe-analyse">
                                    <h3>Compétences Techniques</h3>
                                    <p>Vos hard skills et outils technologiques maîtrisés reflètent votre aptitude à répondre aux exigences spécifiques du poste.</p>
                                </div>
                                  <div class="axe-analyse">
                            <h3>Langues Parlées</h3>
                            <p>L'aptitude à communiquer dans plusieurs langues est un atout précieux dans un contexte professionnel globalisé.</p>
                        </div>
                        <div class="axe-analyse">
                            <h3>Localisation & Mobilité</h3>
                            <p>La flexibilité géographique et la disposition à la mobilité peuvent ouvrir des portes à des opportunités uniques.</p>
                        </div>
                        <div class="axe-analyse">
                                    <h3>Expérience Professionnelle</h3>
                                    <p>Votre historique de travail et les réussites professionnelles montrent votre évolution et l'impact dans vos rôles précédents.</p>
                                </div>
                            </div>
                        </details>
                    </div>
                </div>
                """,
            unsafe_allow_html=True,
        )
    with colonne2:
        st.write("#")
        st.write("#")
        st.write("#")
        image = Image.open("nest_logo.png")
        st.image(image)
        # st.markdown(
        #     """
        #     <h4 style='text-align: center;font-weight: bolder;font-size: 2vw;font-weight: 600;color:rgba(56,181,165,255);'> <bold> Hire with Ease </bold>  </h4>

        #     """,
        #     unsafe_allow_html=True,
        # )
