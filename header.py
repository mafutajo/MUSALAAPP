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
        '<div class="musala-headerx">MUSALA</div>',
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
                "MUSALA",
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
    colonne1, colonne2 = st.columns((0.8, 0.2))
    with colonne1:
        style = """
            <style>
                .musala-content {
                    display: flex;
                    justify-content: space-between;
                    padding: 2vw;
                    font-family: 'Roboto', sans-serif;
                }
                .musala-column {
                    flex-basis: calc(50% - 4vw); /* Prendre la moitié de l'espace disponible, moins l'espacement */
                    background-image: linear-gradient(to bottom, #1B4F72, #88bdbc);
                    padding: 2vw;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 1vw;
                }
                .musala-header {
                    color: white;
                    font-size: 2vw;
                    margin-bottom: 1vw;
                    text-align: center;
                }
               
                .musala-text2 {
                    font-size: 1.0vw;
                    color: white;
                    line-height: 1.5;
                }
                .musala-button {
                    display: block; /* Les boutons s'affichent comme des blocs */
                    width: 100%; /* Les boutons prennent toute la largeur */
                    padding: 10px;
                    margin: 10px 0; /* Marge au-dessus et en dessous du bouton */
                    border: none;
                    border-radius: 5px;
                    color: white;
                    
                    cursor: pointer;
                    text-align: center;
                    text-decoration: none;
                }
                .musala-button:hover {
                    background-color:#FFFFFF;
                }
                @media (max-width: 768px) {
                    .musala-content {
                        flex-direction: column;
                    }
                    .musala-column {
                        flex-basis: 100%;
                        margin: 0 0 2vw 0;
                    }
                    .musala-header {
                        font-size: 4vw;
                    }
                    .musala-text {
                        font-size: 2vw;
                    }
                }
            </style>
            """

        # HTML pour les deux colonnes avec le contenu et les boutons
        content = """
            <div class="musala-content">
                <div class="musala-column">
                    <h2 class="musala-header">Explorez avec MISALA NETWORK</h2>
                    <p class="musala-text2">
                        MUSALA consulte plus de 100,000 postes à chaque requête, vous offrant un accès sans précédent aux opportunités d'emploi. Notre IA analyse votre CV et le marché pour trouver les correspondances parfaites, maximisant ainsi vos chances de succès.
                    </p> <br>
                    <a class="musala-button" href="javascript:void(0);" onclick="alert('En savoir plus sur MISALA NETWORK')"></a>
                </div>
                <div class="musala-column">
                    <h2 class="musala-header">Disclaimer</h2>
                    <p class="musala-text2">
                         MUSALA s'adapte à divers métiers, mais excelle spécifiquement dans le Développement de logiciels, l'Ingénierie des données et le Développement web. Nos algorithmes, affinés pour ces secteurs, optimisent la mise en relation entre talents et opportunités. Ainsi, les professionnels de ces domaines bénéficient d'une expérience enrichie, renforcée par notre engagement en recherche et développement. MUSALA se positionne comme la plateforme idéale pour les experts cherchant à avancer dans ces carrières dynamiques et en évolution constante.
                    </p>
                    <a class="musala-button" href="javascript:void(0);" onclick="alert('Lire le disclaimer')"></a>
                </div>
            </div>
            """

        # Afficher le style et le contenu dans Streamlit
        st.markdown(style + content, unsafe_allow_html=True)

        st.markdown(
            """
        <style>
        /* Styliser tous les boutons de Streamlit */
        .stButton>button {
            border: 2px solid #1a73e8; /* Bordure bleue */
            border-radius: 20px; /* Bords arrondis */
            color: white; /* Texte blanc */
            font-size: 16px; /* Taille de police */
            font-weight: bold; /* Gras */
            background-color: #1a73e8; /* Arrière-plan bleu */
            padding: 10px 24px; /* Padding intérieur */
            box-shadow: 0 2px 4px rgba(0,0,0,0.2); /* Ombre douce */
            transition: background-color 0.3s, box-shadow 0.3s; /* Transition douce */
        }
        .stButton>button:hover {
            background-color: #0f62fe; /* Couleur au survol */
            box-shadow: 0 4px 8px rgba(0,0,0,0.3); /* Ombre plus prononcée au survol */
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

    _, col2, col3 = st.columns([0.2, 0.4, 0.4])

    with col2:
        st.button(
            " EN QUETE D'OPPORTUNITE ",
            on_click=set_menu_option_integer,
            args=(1,),
            help="Je trouve mon opportunitée",
            key="button1",  # Assurez-vous que la clé est unique
        )
    with col3:
        st.button(
            "Découvrez qui est derierre MUSALA ",
            on_click=set_menu_option_integer,
            args=(2,),
            help="Découvre qui est derierre MUSALA NETWORK",
            key="button2",  # Assurez-vous que la clé est unique
        )

    with colonne2:
        st.write("")
        # image = Image.open("nest_logo.png")
        # st.image(image)
        # st.markdown(
        #     """
        #     <h4 style='text-align: center;font-weight: bolder;font-size: 2vw;font-weight: 600;color:rgba(56,181,165,255);'> <bold> Hire with Ease </bold>  </h4>

        #     """,
        #     unsafe_allow_html=True,
        # )
