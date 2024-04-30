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
import matplotlib.pyplot as plt
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
import difflib


@st.cache_resource(show_spinner="Chargement...")
def model_parseur_():
    Parseur = NERModel(
        "camembert", "LOSS_NEW2", use_cuda=False, args={"max_seq_length": 512}
    )
    return Parseur


@st.cache_resource(show_spinner="Chargement...")
def model_sim_():
    model_sim = SentenceTransformer("dangvantuan/sentence-camembert-large")
    return model_sim


@st.cache_resource(show_spinner="Chargement...")
def model_loaded_():
    model_loaded = joblib.load("soft_skill_twenty_27_clean_last.sav")
    return model_loaded


@st.cache_resource(show_spinner="Chargement...")
def chargement_hardfit_():
    from fast_bert.prediction import BertClassificationPredictor

    MODEL_PATH = "MACROFIT_model" + "/" + "model_out"
    LABEL_PATH = ""

    return BertClassificationPredictor(
        model_path=MODEL_PATH,
        label_path=LABEL_PATH,  # location for labels.csv file
        multi_label=False,
        model_type="camembert",
        do_lower_case=False,
        device=None,
    )


def afficher_conseil_diplome(note_tache):
    if note_tache < 0:
        return st.markdown(
            f"""
    <div style="cursor: pointer;font-weight:bold;border-color: red; padding: 0px; border-radius: 6px;margin-bottom: 0;  font-size: 0.8vw; color: rgb(44,63,102); text-align: center;">
        Ce pilier n'est pas specifié dans cette opportunité
    </div>
    """,
            unsafe_allow_html=True,
        )
    if note_tache >= 100:
        couleur_degrade = "rgba(56,181,165,255)"  # Vert foncé

        html_content = f"""
        <div style="cursor: pointer; padding: 0px; border-radius: 6px;margin-bottom: 0;  font-size: 0.8vw; color: rgb(44,63,102); text-align: center;">
            <span style="font-size:0.8vw;color:{couleur_degrade};font-weight:bold;"> Vous semblez suffisament qualifié.e</span>
        </div>
        """

        return st.markdown(html_content, unsafe_allow_html=True)
    else:
        couleur_degrade = "#FA8072"  # Rouge
        html_content = f"""
        <div style="cursor: pointer; padding: 0px; border-radius: 6px;margin-bottom: 0;  font-size: 0.7vw; color: rgb(44,63,102); text-align: center;">
            <span style="font-size:0.8vw;color:{couleur_degrade};font-weight:bold;">Vous ne semblez pas assez qualifié.e</span>
        </div>
        """

        return st.markdown(html_content, unsafe_allow_html=True)


def afficher_conseil_langue(note_tache):
    if note_tache < 0:
        return st.markdown(
            f"""
    <div style="cursor: pointer;font-weight:bold;border-color: red; padding: 0px; border-radius: 6px;margin-bottom: 0;  font-size: 0.8vw; color: rgb(44,63,102); text-align: center;">
        Ce pilier n'est pas specifié dans cette opportunité
    </div>
    """,
            unsafe_allow_html=True,
        )
    if note_tache >= 100:
        couleur_degrade = "rgba(56,181,165,255)"  # Vert foncé

        html_content = f"""
        <div style="cursor: pointer; padding: 0px; border-radius: 6px;margin-bottom: 0;  font-size: 0.8vw; color: rgb(44,63,102); text-align: center;">
            <span style="font-size:0.8vw;color:{couleur_degrade};font-weight:bold;"> Vous avez la maitrise linguistique</span>
        </div>
        """

        return st.markdown(html_content, unsafe_allow_html=True)
    else:
        couleur_degrade = "#FA8072"  # Rouge
        html_content = f"""
        <div style="cursor: pointer; padding: 0px; border-radius: 6px;margin-bottom: 0;  font-size: 0.7vw; color: rgb(44,63,102); text-align: center;">
            <span style="font-size:0.8vw;color:{couleur_degrade};font-weight:bold;">Vous n'avez pas la maitrise linguistique</span>
        </div>
        """

        return st.markdown(html_content, unsafe_allow_html=True)


def afficher_conseil_experience(note_tache):
    if note_tache < 0:
        return st.markdown(
            f"""
    <div style="cursor: pointer;font-weight:bold;border-color: red; padding: 0px; border-radius: 6px;margin-bottom: 0;  font-size: 0.8vw; color: rgb(44,63,102); text-align: center;">
        Ce pilier n'est pas specifié dans cette opportunité
    </div>
    """,
            unsafe_allow_html=True,
        )
    if note_tache >= 100:
        couleur_degrade = "rgba(56,181,165,255)"  # Vert foncé

        html_content = f"""
        <div style="cursor: pointer; padding: 0px; border-radius: 6px;margin-bottom: 0;  font-size: 0.8vw; color: rgb(44,63,102); text-align: center;">
            <span style="font-size:0.8vw;color:{couleur_degrade};font-weight:bold;"> Vous semblez etre suffisament experimenté.e</span>
        </div>
        """

        return st.markdown(html_content, unsafe_allow_html=True)
    else:
        couleur_degrade = "#FA8072"  # Rouge
        html_content = f"""
        <div style="cursor: pointer; padding: 0px; border-radius: 6px;margin-bottom: 0;  font-size: 0.7vw; color: rgb(44,63,102); text-align: center;">
            <span style="font-size:0.8vw;color:{couleur_degrade};font-weight:bold;">Vous ne semblez pas assez experimenté.e</span>
        </div>
        """

        return st.markdown(html_content, unsafe_allow_html=True)


def afficher_conseil(note_tache: float):
    if note_tache < 0:
        return st.markdown(
            f"""
    <div style="cursor: pointer;font-weight:bold;border-color: red; padding: 0px; border-radius: 6px;margin-bottom: 0;  font-size: 0.8vw; color: rgb(44,63,102); text-align: center;">
        Ce pilier n'est pas specifié dans cette opportunité
    </div>
    """,
            unsafe_allow_html=True,
        )
    if note_tache >= 85:
        couleur_degrade = "rgba(56,181,165,255)"  # Vert foncé
    elif note_tache >= 70:
        couleur_degrade = "rgba(56,181,165,255)"  # Vert
    elif note_tache >= 55:
        couleur_degrade = "#ff7f00"  # Orange
    elif note_tache >= 40:
        couleur_degrade = "#ff7f00"  # Orange foncé
    else:
        couleur_degrade = "#FA8072"  # Rouge

    pourcentage_correspondance = f"{int(note_tache)}%"

    html_content = f"""
    <div style="cursor: pointer; padding: 0px; border-radius: 6px;margin-bottom: 0;  font-size: 0.8vw; color: rgb(44,63,102); text-align: center;">
        <span style="font-size:1vw;color:{couleur_degrade};font-weight:bold;">{pourcentage_correspondance} </span> de Match
    </div>
    """

    st.markdown(html_content, unsafe_allow_html=True)


def afficher_conseil_stack(note_stack: float):
    if note_stack >= 85:
        conseil = "Votre stack est fortement compatible avec ce poste."
        couleur_degrade = "rgba(56,181,165,255)"
    elif note_stack >= 70:
        conseil = "Votre stack est majoritairement compatible, avec quelques petites exceptions."
        couleur_degrade = "rgba(56,181,165,255)"
    elif note_stack >= 55:
        conseil = "Votre stack montre une compatibilité modérée. Examinez les domaines à améliorer."
        couleur_degrade = "#ff7f00"
    elif note_stack >= 40:
        conseil = "Il y a des incompatibilités notables entre votre stack et ce qui est recherché."
        couleur_degrade = "#ff7f00"
    else:
        conseil = (
            "Il existe une incompatibilité significative entre votre stack et le poste."
        )
        couleur_degrade = "#FA8072"

    html_content = f"""
    <details style="cursor: pointer; background: linear-gradient(to right, {couleur_degrade}, {couleur_degrade}); padding: 10px; border-radius: 6px; font-size: 0.8vw;">
        <summary>l'avis NEST</summary>
        {conseil}
    </details>
    """

    st.markdown(html_content, unsafe_allow_html=True)


@st.cache_resource
def model_comparateur_():
    model_comparateur = Word2Vec.load("hard_skill_matcher.model")
    return model_comparateur


def diviseur_choice(number):
    if number <= 4:
        return number
    elif number > 4 and number <= 10:
        return math.ceil(4 + (number - 4) * 0.4)
    else:
        return math.ceil(10 + (number - 10) * 0.2)


def pdf_to_text(path: str):
    """transformer le pdf en texte"""
    with open(path, mode="rb"):
        uploaded_file = open(path, mode="rb")
        bytes_content: bytes = uploaded_file.read()
        file_name: str = uploaded_file.name
        raw_text: str = extract_text(io.BytesIO(bytes_content)).replace(
            "\x00", "\uFFFD"
        )
        return raw_text


def MEF_alternatif_1(base: pd.DataFrame) -> pd.DataFrame:
    """transformer les sorties en dataframe  pour la lisibilité"""
    df = pd.DataFrame(columns=["WORD", "TAG"])
    for value in base:
        if len(value) > 0:
            temp = pd.DataFrame(
                {
                    "WORD": [i[0] for i in value.items()],
                    "TAG": [i[1] for i in value.items()],
                }
            )
            df = pd.concat([df, temp])

    return df


def transform(text: str) -> list:
    """cette fonction split les phrases en list de mots de manière inteligente que simplement par le split()"""

    text = (
        text.replace(",", " ")
        .replace("\n", " ")
        .replace(":", " ")
        .replace("/", " ")
        .replace("(", "")
        .replace(")", "")
        .replace("~", " ")
        .replace("/", " ")
    )

    # nlp = French()
    # doc = nlp(
    #     text.replace(",", " ")
    #     .replace("\n", " ")
    #     .replace(":", " ")
    #     .replace("/", " ")
    #     .replace("(", "")
    #     .replace(")", "")
    #     .replace("~", " ")
    #     .replace("/", " ")
    # )

    # tokens = [token.text for token in doc]
    tokens = text.split()
    return tokens


def has_useless_text(string: str) -> bool:
    """Check if text is useful or not.

    Args:
        string: str text to check

    Returns:
        number: bool True if text doens't contain information.

    Example:
        has_useless_text("@##") -> False
    """
    numbers = sum(c.isdigit() for c in string)
    letters = sum(c.isalpha() for c in string)

    return numbers + letters == 0


def join_tokens(tokens):
    res = ""
    if tokens:
        res = tokens[0]
        for token in tokens[1:]:
            if not (token.isalpha() and res[-1].isalpha()):
                res += " " + token  # punctuation
            else:
                res += " " + token  # regular word
    return res


def collapse(ner_result):
    ner_result = tuple(ner_result.itertuples(index=False, name=None))
    # List with the result
    collapsed_result = []

    current_entity_tokens = []
    current_entity = None

    # Iterate over the tagged tokens
    for token, tag in ner_result:
        if tag.startswith("B-"):
            # ... if we have a previous entity in the buffer, store it in the result list
            if current_entity is not None:
                collapsed_result.append(
                    [join_tokens(current_entity_tokens), current_entity]
                )

            current_entity = tag[2:]
            # The new entity has so far only one token
            current_entity_tokens = [token]

        # If the entity continues ...
        elif current_entity_tokens != None and tag == "I-" + str(current_entity):
            # Just add the token buffer
            current_entity_tokens.append(token)
        else:
            collapsed_result.append(
                [join_tokens(current_entity_tokens), current_entity]
            )
            collapsed_result.append([token, tag[2:]])

            current_entity_tokens = []
            current_entity = None

            pass

    # The last entity is still in the buffer, so add it to the result
    # ... but only if there were some entity at all
    if current_entity is not None:
        collapsed_result.append([join_tokens(current_entity_tokens), current_entity])
        collapsed_result = sorted(collapsed_result)
        collapsed_result = list(k for k, _ in itertools.groupby(collapsed_result))

    return collapsed_result


def by_size(words: list, size: int) -> list:
    """renvoie la liste d'élement supérieur à 3"""
    return [word for word in words if len(word) > size]


def MEF_final(table: dict) -> dict:
    fichier_retour = {}
    fichier_retour["soft_skills"] = by_size(
        list(
            dict.fromkeys(
                [
                    x.lower()
                    for x in list(
                        dict.fromkeys([el[0] for el in table if el[1] == "SSKILL"])
                    )
                ]
            )
        ),
        3,
    )
    fichier_retour["hard_skills"] = list(
        dict.fromkeys(
            [
                x.lower()
                for x in list(
                    dict.fromkeys([el[0] for el in table if el[1] == "HSKILL"])
                )
            ]
        )
    )

    ####
    fichier_retour["ecole"] = list(
        dict.fromkeys([el[0] for el in table if el[1] == "ECOLE"])
    )
    fichier_retour["entreprises"] = list(
        dict.fromkeys([el[0] for el in table if el[1] == "ORG"])
    )
    fichier_retour["diplome"] = list(
        dict.fromkeys([el[0] for el in table if el[1] == "DIPLÔME"])
    )
    fichier_retour["langue"] = by_size(
        list(dict.fromkeys([el[0] for el in table if el[1] == "LANGUE"])), 3
    )
    fichier_retour["post"] = by_size(
        list(dict.fromkeys([el[0] for el in table if el[1] == "POST"])), 4
    )

    fichier_retour["nom_prenom"] = list(
        dict.fromkeys([el[0] for el in table if el[1] == "NOM"])
    )
    fichier_retour["adresse"] = list(
        dict.fromkeys([el[0] for el in table if el[1] == "ADRESSE"])
    )

    fichier_retour["tache"] = list(
        dict.fromkeys([el[0] for el in table if el[1] == "TACHE"])
    )

    fichier_retour["experience"] = list(
        dict.fromkeys([el[0] for el in table if el[1] == "EXP"])
    )

    fichier_retour["tache"] = by_size(
        list(
            dict.fromkeys(
                [
                    x.lower()
                    for x in list(
                        dict.fromkeys([el[0] for el in table if el[1] == "TACHE"])
                    )
                ]
            )
        ),
        1,
    )
    return fichier_retour


def filter_substring(string: list, substr: str) -> list:
    """netoie les sorties langues"""
    if not substr:
        return []
    return [str for str in string if any(sub in str for sub in substr)]


def extract_telephone(texte):
    ### complète les noms
    liste_numero = []
    # Expression régulière pour capturer les numéros de téléphone
    regex_telephone = re.compile(
        r"\b(?:0\s*[1-9](?:[\s.]?\d\s*){8}|0\s*[1-9]\s*(?:[\s.]?\d\s*){8}|0\s*[1-9](?:[\s.]?\d\s*){3}[\s.]?\d\s*(?:[\s.]?\d\s*){3}[\s.]?\d\s*(?:[\s.]?\d\s*){2})\b"
    )

    # Recherche des numéros de téléphone dans le texte
    resultats = regex_telephone.findall(texte)

    # Affichage des numéros de téléphone trouvés
    for numero in resultats:
        # Supprimer les espaces et les points pour afficher le numéro de téléphone sans séparateurs
        numero_nettoye = re.sub(r"[\s.]", "", numero)
        liste_numero.append(numero_nettoye)
    return liste_numero


def diff_month(d1, d2):
    """calcule la difference de temps entre deux dates"""
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def clean_list(list_: list) -> list:
    """Recois une liste et la mets en minuscule et enleve les accents"""

    return list(dict.fromkeys([(x.lower()) for x in list_]))


def estimated_duration(listed):
    ####### estimated time of duration
    ##### -1 pas de temps
    ####-9 actual
    ###-2 more than 2 date
    #### else estimation

    if not listed:
        return -1

    elif len(listed) == 1:
        return -9

    elif len(listed) == 2:
        if abs(int(listed[0]) - int(listed[1])) == 0:
            return 1
        return abs(int(listed[0]) - int(listed[1]))
    else:
        return -2


def experience_in_field(ma_liste, job_type):
    #######ANNEE DEXP SUR Le field

    """comment cela fonctionne ? on consid§re uniquement les experiences qui ont une date valide donc date de debit et de fin  + l'organisation dans laquelle il travail est reconnu"""

    ma_condition = job_type
    resultat = 0
    for element in ma_liste:
        if element.get("classe") == ma_condition:
            if (
                element.get("estimation_time") >= 0
                and len(element.get("entreprise")) > 0
            ):
                resultat = +element.get("estimation_time")

    return resultat


def close_enough(index_number, resultats, tresh):
    list_return = []

    for element in resultats:
        name, index_nb = element

        if abs(index_number - index_nb) <= tresh:
            list_return.append(name)
    return list_return


def org_plus(texte):
    from transformers import AutoTokenizer, AutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
    model = AutoModelForTokenClassification.from_pretrained(
        "Jean-Baptiste/camembert-ner"
    )

    ##### Process text sample (from wikipedia)

    from transformers import pipeline

    list_org = []

    nlp = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )

    sortie = nlp(texte)

    for el in sortie:
        if el["entity_group"] == "ORG":
            list_org.append((el["word"], el["start"]))
    return list_org


def estimated_duration(listed):
    ####### estimated time of duration
    ##### -1 pas de temps
    ####-9 actual
    ###-2 more than 2 date
    #### else estimation

    if not listed:
        return -1

    elif len(listed) == 1:
        return -9

    elif len(listed) == 2:
        return abs(int(listed[0]) - int(listed[1])) + 1
    else:
        return -2


model_sim = model_sim_()

model_loaded = model_loaded_()


def close_enough(index_number, resultats, tresh):
    list_return = []

    for element in resultats:
        name, index_nb = element

        if abs(index_number - index_nb) <= tresh:
            list_return.append(name)
    return list_return


def closest_classe(word):
    # Liste de mots
    base_job = pd.read_excel("job_listing.xlsx")

    # base_job.to_excel('job_listing.xlsx')
    liste_de_mots = base_job["job_title"].tolist()
    list_classe = base_job["job_types"].tolist()

    # Mot à rechercher
    mot_a_rechercher = word  # Le mot mal orthographié que nous cherchons

    # Seuil de ressemblance minimum
    seuil_minimum = 85

    # Initialisation des variables pour le suivi du terme le plus proche
    meilleur_score = 0
    index_du_meilleur_match = None

    # Parcours de la liste pour trouver le terme le plus proche
    for index, mot in enumerate(liste_de_mots):
        score = fuzz.ratio(str(mot_a_rechercher), str(mot))
        if score > meilleur_score and score >= seuil_minimum:
            meilleur_score = score
            index_du_meilleur_match = index

    # Renvoie l'index du terme le plus proche
    if index_du_meilleur_match is not None:
        return (
            liste_de_mots[index_du_meilleur_match],
            list_classe[index_du_meilleur_match],
        )
        print(
            f"Le terme le plus proche de '{mot_a_rechercher}' est '{liste_de_mots[index_du_meilleur_match]}' à l'index {index_du_meilleur_match}. et la classe {list_classe[index_du_meilleur_match]}"
        )
    else:
        return None, None
        print(
            f"Aucun terme proche de '{mot_a_rechercher}' n'a été trouvé dans la liste."
        )


import re


def find_closest_entity_within_limit(text, index_post, entities, limit=40):
    closest_entity = None
    min_distance = float("inf")
    for entity, index_entity in entities:
        if abs(index_entity - index_post) <= limit:
            distance = abs(index_entity - index_post)
            if distance < min_distance:
                min_distance = distance
                closest_entity = (entity, index_entity)
    return closest_entity


def extract_dates_within_range(text, start_year=1900, end_year=2025):
    dates = re.findall(r"\b(19\d{2}|20[01]\d|202[0-5])\b", text)
    return [
        (date, match.start())
        for date, match in zip(
            dates, re.finditer(r"\b(19\d{2}|20[01]\d|202[0-5])\b", text)
        )
    ]


def calculate_duration(start, end):
    if start and end:
        return abs(int(end) - int(start)) + 1
    return 0  # Retourne 0 si seulement une date est trouvée


def recreate_experience(sortie, texte):
    experiences = []
    total_experience_years = (
        0  # Initialisation du compteur pour la somme totale d'expérience
    )

    # Extraire les postes, entreprises, et dates valides
    list_post = [
        (post, match.start())
        for post in sortie["post"]
        for match in re.finditer(re.escape(post.lower()), texte.lower())
    ]
    list_org = [
        (org, match.start())
        for org in sortie["entreprises"]
        for match in re.finditer(re.escape(org.lower()), texte.lower())
    ]
    list_date = extract_dates_within_range(texte)

    for post, index_post in list_post:
        closest_org = find_closest_entity_within_limit(texte, index_post, list_org)
        closest_dates = sorted(
            [
                (date, index_date)
                for date, index_date in list_date
                if abs(index_date - index_post) <= 40
            ],
            key=lambda x: abs(x[1] - index_post),
        )[:2]

        if closest_org and closest_dates:
            start_date = closest_dates[0][0]
            end_date = closest_dates[1][0] if len(closest_dates) > 1 else None
            duration = calculate_duration(start_date, end_date)
            total_experience_years += duration

            experiences.append(
                {
                    "poste": post,
                    "entreprise": closest_org[0] if closest_org else None,
                    "start_date": start_date,
                    "end_date": end_date,
                    "duration": duration,
                }
            )

    # Suppression des doublons
    unique_experiences = [dict(t) for t in {tuple(d.items()) for d in experiences}]
    return unique_experiences, total_experience_years


from spacy.lang.fr.stop_words import STOP_WORDS


def clean_task(texte: list) -> list:
    texte = [x.lower() for x in texte if x not in STOP_WORDS]

    new_list = []
    for el in texte:
        if len(el) > 2:
            if el.split()[-1] in STOP_WORDS:
                new_list.append(" ".join(el.split()[:-1]))
            else:
                new_list.append(el)

    return new_list


def find_closest_element(word, liste, text, distance):
    text = text.replace("/n", " ")
    print(liste)

    index = text.find(word)
    print(index)

    if index == -1:
        return []

    else:
        liste_index = [text.find(word) for word in liste]
        print(liste_index)

        #######les noms qui correspondent
        loutre = [x for x in liste_index if abs(x - index) < distance]

        almost = [liste_index.index(x) for x in loutre]

        liste_candidat = [liste[i] for i in almost]

        return liste_candidat


def recreate_experience_school(sortie, texte):
    ##### param..
    tresh_org = 30
    tresh_date = 60

    #### info recupération

    ### Post
    list_post = []
    for i in sortie["diplome"]:
        resultats_post = [
            (match.group(0), match.start())
            for match in re.finditer(i.lower(), texte.lower())
        ]
        list_post = list_post + resultats_post

    ### date

    pattern = r"\b\d{4}\b"
    resultats_date = [
        (match.group(0), match.start()) for match in re.finditer(pattern, texte.lower())
    ]

    ###
    list_org = []
    for i in sortie["ecole"]:
        resultats = [
            (match.group(0), match.start())
            for match in re.finditer(i.lower(), texte.lower())
        ]
        list_org = list_org + resultats

    ##### Verification pas de post

    if not list_post:
        """pas de poste disponible"""
        return []

    else:
        list_return = []
        for post in list_post:
            post_name, index_number = post
            list_return.append(
                {
                    "index": index_number,
                    "diplome": post_name,
                    "date_compatible": close_enough(
                        index_number, resultats_date, tresh_date
                    ),
                    "estimation_time": estimated_duration(
                        close_enough(index_number, resultats_date, tresh_date)
                    ),
                    "ecole": close_enough(index_number, list_org, tresh_org),
                }
            )

        def deduplicate_list_of_dicts(input_list, index_key):
            seen = set()
            deduplicated_list = []

            for d in input_list:
                if d[index_key] not in seen:
                    seen.add(d[index_key])
                    deduplicated_list.append(d)

            return deduplicated_list

        index_key = "index"

        return deduplicate_list_of_dicts(list_return, index_key)


import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import re


# from login import login_page

from datetime import time
import datetime
import streamlit as st

from datetime import time
import pandas as pd


def page_acceuil():
    colonne1, colonne2 = st.columns(2)
    with colonne1:
        st.markdown("#")
        st.markdown("#")
        st.markdown(
            """
            <span style='text-align:center;'>
            <h1 style='text-align: center;'> Welcome to the <span style='color:rgba(56,181,165,255);font-weight: bolder;'> &#10024 <bold> Nest </bold> &#10024 </span> </h1> <br> 

            <p> Nous sommes ravis de vous accueillir dans notre nid virtuel dédié au matching et à la gestion de vos talents. </p>
            <p> Notre engagement ? Vous permettre de valoriser au maximum votre <span style='color:rgba(56,181,165,255);font-weight: bolder;'> <em> talent Nest</em>.</span>  </p>
            <p> Constituez votre <span style='color:rgba(56,181,165,255);font-weight: bolder;'> talent Nest  </span> en uploadant vos profils .<br> Ensuite, à chacun de vos besoins, <span style='color:rgba(56,181,165,255);font-weight: bolder;'> NEST </span> s'occupera de trouver la parfaite association dans votre talent <span style='color:rgba(56,181,165,255);font-weight: bolder;'> NEST </span></p>
            </span>
            """,
            unsafe_allow_html=True,
        )
    with colonne2:
        st.markdown("#")
        st.markdown("#")
        st.markdown(
            """
            <span style='text-align:center;'>
            <h1 style='text-align: center;color:rgba(16, 216, 240, 1);'> TOP JOBS </bold> &#10024 </span> </h1> <br> 

            <p> Voici la liste des top 10 emplois les plus recherchés sur tout jobboards </p>
            <p> Notre engagement ? Vous permettre de valoriser au maximum votre <span style='color:rgba(56,181,165,255);font-weight: bolder;'> <em> talent Nest</em>.</span>  </p>
            <p> Constituez votre <span style='color:rgba(56,181,165,255);font-weight: bolder;'> talent Nest  </span> en uploadant vos profils .<br> Ensuite, à chacun de vos besoins, <span style='color:rgba(56,181,165,255);font-weight: bolder;'> NEST </span> s'occupera de trouver la parfaite association dans votre talent <span style='color:rgba(56,181,165,255);font-weight: bolder;'> NEST </span></p>
            </span>
            """,
            unsafe_allow_html=True,
        )

        numbers = [
            "DATA SCIENCE",
            "WEB",
            "UX",
            "WEB",
            "DATA SCIENCE",
            "DATA SCIENCE",
            "UX",
            "WEB",
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(x=numbers, name="count", texttemplate="%{x}", textfont_size=20)
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


def traiter_skills_dict(input_dict):
    # Techniques de gestion de projet à inclure
    techniques_gestion_projet = [
        "agile",
        "kanban",
        "scrum",
        "lean",
        "agiles",
        "crystal",
        "waterfall",
        "cascade",
    ]

    # Extraction des techniques de gestion de projet de la liste des hard skills
    gestion_de_projet_skills = [
        skill
        for skill in input_dict.get("hard_skills", [])
        if skill.lower() in techniques_gestion_projet
    ]

    # Ajout des techniques de gestion de projet à la nouvelle clé 'projet'
    input_dict["projet"] = gestion_de_projet_skills

    # Suppression des techniques de gestion de projet de la liste des hard skills
    input_dict["hard_skills"] = [
        skill
        for skill in input_dict.get("hard_skills", [])
        if skill.lower() not in techniques_gestion_projet
    ]

    return input_dict


def page_analyse(uploaded_file):
    candidat = parsing_joboffer(pdf_to_text(uploaded_file.name))
    colonne1, colonne2 = st.columns((0.3, 0.7), gap="large")
    with colonne1:
        displayPDF(uploaded_file.name)

    with colonne2:
        col1, col2, col3, col4 = st.columns(4, gap="large")
        ##### Colonne d'identité
        with col1:
            html_str = f"""
            <h5 style='text-align: center; color:rgba(56,181,165,255);'> Identité </h5> <b> <p style='text-align: center;font-size:24px' ><strong>{" ".join(candidat["nom_prenom"][:2])}</strong></p></b>
                    """

            st.markdown(html_str, unsafe_allow_html=True)

        with col2:
            html_str = f"""
            <h5 style='text-align: center; color:rgba(56,181,165,255);'>Mail</h5> <b> <p style='text-align: center;font-size:24px' ><strong>{" ".join(candidat["email"][:1])}</strong></p></b>
                    """

            st.markdown(html_str, unsafe_allow_html=True)

        with col3:
            html_str = f"""
            <h5 style='text-align: center; color:rgba(56,181,165,255);'>Télephone </h5> <b> <p style='text-align: center;font-size:24px' ><strong>{" ".join(candidat["numero"][:1])}</strong></p></b>
                    """

            st.markdown(html_str, unsafe_allow_html=True)

        with col4:
            html_str = f"""
            <h5 style='text-align: center; color:rgba(56,181,165,255);'>Adresse </h5> <b> <p style='text-align: center;font-size:24px' ><strong>{" ".join(candidat["adresse"][:1])}</strong></p></b>
                    """

            st.markdown(html_str, unsafe_allow_html=True)

        ##### Colonne diplome experience ###########
        col_dipl, col_experience, col_langue = st.columns(3, gap="large")

        st.markdown("#")
        st.markdown("#")
        st.markdown("#")
        st.markdown("#")

        #######diplome
        with col_dipl:
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            html_str = f"""
            <h5 style='text-align: center; color:rgba(56,181,165,255);'>Diplome </h5> <b> <p style='text-align: center;font-size:24px' ><strong>{candidat["parcours_scolaire"][:1]}</strong></p></b>
                    """

            st.markdown(html_str, unsafe_allow_html=True)
        ######### Colonne parcours Pro
        with col_experience:
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            html_str = f"""
            <h5 style='text-align: center; color:rgba(56,181,165,255);'>Parcours Pro </h5> <b> <p style='text-align: center;font-size:24px' ><strong>{candidat["parcours_pro"][:4]}</strong></p></b>
                    """

            st.markdown(html_str, unsafe_allow_html=True)

        with col_langue:
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            html_str = f"""
            <h5 style='text-align: center; color:rgba(56,181,165,255);'>Langue </h5> <b> <p style='text-align: center;font-size:24px' ><strong>{candidat["langue"]}</strong></p></b>
                    """

            st.markdown(html_str, unsafe_allow_html=True)

        #########
        ##### Colonne diplome experience ###########
        col_stack, col_skillset, col_softskills = st.columns(3, gap="large")

        st.markdown("#")
        st.markdown("#")
        st.markdown("#")
        st.markdown("#")

        #######diplome
        with col_stack:
            st.markdown("#")
            st.markdown("#")
            base_SKILLS = candidat["hard_skills"]
            hard_skills_list_display = []

            hard_skills_display = "AUCUN"
            if base_SKILLS:
                for hard_skill in base_SKILLS:
                    hard_skills_list_display.append((hard_skill, ""))
                html_diplome = f"""
                            <h5 style='text-align: center; color:rgba(56,181,165,255);'> STACKS </h5> <b> <p style='text-align: center;'></p></b>
                        """
                st.markdown(html_diplome, unsafe_allow_html=True)
                annotated_text(hard_skills_list_display)
            else:
                html_diplome = f"""
                    <h5 style='text-align: center; color:rgba(56,181,165,255);'> STACKS </h5> <b> <p style='text-align: center;'>{hard_skills_display} </p></b>
                        """

                st.markdown(html_diplome, unsafe_allow_html=True)
        with col_skillset:
            st.markdown("#")
            st.markdown("#")

            base_SSKILLS = candidat["tache"]
            soft_skills_list_display = []

            if len(base_SSKILLS) > 0:
                for soft_skill in base_SSKILLS:
                    soft_skills_list_display.append((soft_skill, "", "#faf"))
                html_diplome = f"""
                            <h5 style='text-align: center; color:rgba(56,181,165,255);'> SKILLSET </h5> <b> <p style='text-align: center;'></p></b>
                        """
                st.markdown(html_diplome, unsafe_allow_html=True)
                annotated_text(soft_skills_list_display)

        with col_softskills:
            st.markdown("#")
            st.markdown("#")

            base_SSKILLS = candidat["soft_skills"]
            soft_skills_list_display = []

            if len(base_SSKILLS) > 0:
                for soft_skill in base_SSKILLS:
                    soft_skills_list_display.append((soft_skill, "", "#8ef"))
                html_diplome = f"""
                            <h5 style='text-align: center; color:rgba(56,181,165,255);'> SOFT SKILLS </h5> <b> <p style='text-align: center;'></p></b>
                        """
                st.markdown(html_diplome, unsafe_allow_html=True)
                annotated_text(soft_skills_list_display)


def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def pdf_to_text(path: str):
    """transformer le pdf en texte"""
    with open(path, mode="rb"):
        uploaded_file = open(path, mode="rb")
        bytes_content: bytes = uploaded_file.read()
        file_name: str = uploaded_file.name
        raw_text: str = extract_text(io.BytesIO(bytes_content)).replace(
            "\x00", "\uFFFD"
        )
        return raw_text


def diff_month(d1, d2):
    """calcule la difference de temps entre deux dates"""
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def clean_list(list_: list) -> list:
    """Recois une liste et la mets en minuscule et enleve les accents"""

    return list(dict.fromkeys([(x.lower()) for x in list_]))


def name_choice(sortie, text):
    if not sortie["nom_prenom"]:
        if not sortie["mail"]:
            return [" "]
        else:
            [sortie["mail"][0]]
    else:
        index_name = []
        for i in sortie["nom_prenom"]:
            index_name.append(text.find(i))
        try:
            return [
                sortie["nom_prenom"][
                    index_name.index(min(filter(lambda x: x >= 0, index_name)))
                ]
            ]

        except:
            return [sortie["nom_prenom"][0]]


def sliding_window(listed_text_cv, window_size, overlap):
    """
    Applique un fenêtrage glissant sur la liste de segments de texte.
    """
    base = []

    step = window_size - overlap
    for i in range(0, len(listed_text_cv), step):
        window = listed_text_cv[i : i + window_size]
        base.append(window)
    return base


def est_similaire(a, b, seuil=0.9):
    """Vérifie si deux chaînes sont similaires au-dessus d'un seuil donné."""
    return difflib.SequenceMatcher(None, a, b).ratio() > seuil


def filtrer_et_dedoublonner(liste_taches):
    # Filtrer les tâches ayant plus d'un mot
    taches_filtrees = [tache for tache in liste_taches if len(tache.split()) > 1]

    # Dédoublonner les tâches basées sur la similarité
    taches_uniques = []
    for tache in taches_filtrees:
        if not any(
            est_similaire(tache, tache_existante) for tache_existante in taches_uniques
        ):
            taches_uniques.append(tache)

    return taches_uniques


def filtrer_liste_postes(postes):
    # Mots à exclure exactement
    exclusions_exactes = {
        "serveur",
        "équipier",
        "ingénieur",
        "ingenieur",
        "responsable",
        "automobile",
        "apprenti",
        "chargé",
        "senior",
        "sénior",
        "dévellopeur",
    }
    # Liste des sous-chaînes à exclure dans n'importe quel mot
    exclusions_contenant = ["polyvalent", "bureau"]

    # Filtrage
    postes_filtres = [
        poste
        for poste in postes
        if poste not in exclusions_exactes
        and not any(exclusion in poste for exclusion in exclusions_contenant)
    ]

    return postes_filtres


def enlever_points(competences):
    competences_sans_points = [comp.rstrip(".") for comp in competences]
    return competences_sans_points


def filtrer_taches(tasks):
    sorted_tasks = sorted(
        tasks, key=len, reverse=True
    )  # Trier par longueur décroissante
    filtered_tasks = []
    for task in sorted_tasks:
        if not any(
            task in other_task and task != other_task for other_task in filtered_tasks
        ):
            filtered_tasks.append(task)
    return filtered_tasks


def nettoyer_taches(taches):
    # Expression régulière pour identifier les fins de phrases à supprimer
    pattern = r"\s+(des|de|et|les|le|la|du|d’une|d’un|en|de mon)$"

    def nettoyer_tache(tache):
        # Utilise l'expression régulière pour supprimer les fins spécifiées de la tâche
        tache_nettoyee = re.sub(pattern, "", tache)
        return tache_nettoyee

    # Applique la fonction de nettoyage à chaque tâche dans la liste
    taches_nettoyees = [nettoyer_tache(tache) for tache in taches]

    return taches_nettoyees


def sliding_window2(listed_text_cv, window_size, overlap):
    step = window_size - overlap
    return [
        listed_text_cv[i : i + window_size] for i in range(0, len(listed_text_cv), step)
    ]


def parsing_joboffer(
    text_cv: str | st.runtime.uploaded_file_manager.UploadedFile,
) -> dict:
    if not isinstance(text_cv, str):
        bytes_content: bytes = text_cv.read()
        text_cv: str = extract_text(io.BytesIO(bytes_content)).replace("\x00", "\uFFFD")

    listed_text_cv = [
        x for x in list(filter(None, transform(text_cv))) if not has_useless_text(x)
    ]

    window_size = 50
    overlap = 49
    base = sliding_window2(listed_text_cv, window_size, overlap)

    print(base)
    Parseur = model_parseur_()

    # st.write(base)
    # Convertir chaque fenêtre en une liste de mots (pour split_on_space=False)
    # windows_as_word_lists = [base]  # Assurez-vous que `window` est un str ici

    # Appeler le modèle une seule fois avec la liste des listes de mots
    predictions, _ = Parseur.predict(base, split_on_space=False)
    from itertools import chain

    predictions = list(chain.from_iterable(predictions))

    final_predictions = []
    for pred in predictions:
        if pred != "O" or not final_predictions:
            final_predictions.append(pred)
        else:
            final_predictions[-1] = pred if pred != "O" else final_predictions[-1]

    base_data = MEF_alternatif_1(
        list(itertools.chain.from_iterable([final_predictions]))
    )
    # display(base_data)
    base_data["tf"] = base_data["WORD"].apply(lambda x: has_useless_text(x))
    base_data = base_data[["WORD", "TAG"]]  # [base_data["tf"] == False]

    final_version = collapse(base_data)

    sortie = MEF_final(final_version)

    sortie.update(
        raw_text=text_cv,
    )

    sortie.update(
        numero=extract_telephone(text_cv),
    )

    sortie.update(mail=re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text_cv))

    sortie.update(
        langue=[
            element.replace(".", "")
            for element in filter_substring(
                sortie["langue"],
                base_data[base_data["TAG"] == "B-LANGUE"]["WORD"].to_list(),
            )
        ]
    )

    # sortie.update(nom_prenom=name_choice(sortie["nom_prenom"], text_cv))
    sortie.update(nom_prenom=name_choice(sortie, text_cv))

    # sortie.update(
    #     tache=clean_list(
    #         filter_substring(
    #             sortie["tache"],
    #             base_data[base_data["TAG"] == "B-TACHE"]["WORD"].to_list(),
    #         )
    #     )
    # )

    sortie.update(tache=[y for y in sortie["tache"] if len(y.split()) > 1])

    sortie.update(id_pdf_hashed=hashlib.md5(text_cv.encode("utf-8")).hexdigest())

    # sortie.update(
    #     tache=clean_task(clean_task(list(dict.fromkeys(clean_task(sortie["tache"])))))
    # )
    sortie.update(tache=filtrer_et_dedoublonner(sortie["tache"]))

    sortie.update(
        tache=enlever_points(
            filtrer_taches(nettoyer_taches(nettoyer_taches(sortie["tache"])))
        )
    )
    sortie.update(hard_skills=enlever_points(sortie["hard_skills"]))
    sortie.update(
        diplome=filter_substring(
            sortie["diplome"],
            base_data[base_data["TAG"] == "B-DIPLÔME"]["WORD"].to_list(),
        )
    )
    sortie.update(
        nom_prenom=combine_consecutive_words_in_text(text_cv, sortie["nom_prenom"])
    )
    sortie.update(
        diplome=clean_task(clean_task(clean_task(clean_task(sortie["diplome"]))))
    )

    sortie.update(ecole=clean_task(clean_task(clean_task(clean_task(sortie["ecole"])))))

    sortie.update(soft_skills=clean_task(clean_task(clean_task(sortie["soft_skills"]))))
    sortie.update(location=[trouver_region_gagnante(sortie["adresse"])])
    common_eror = [
        "d’une",
        "d'un",
        "un",
        "faites",
        "mémoire",
        "qualité",
        "observation",
        "orthographique",
        "nature",
        "active",
        "prendre",
        "faire",
        "haute",
        "client",
        "efficacement",
        "expression",
        "stress",
        "capacités",
        "équipe",
        "etre",
        "sens",
        "travail",
        "force",
        "esprit",
        "analyse",
        "apprendre",
        "capacité",
        "travailler",
        "realation",
        "oral",
        "ecris",
        "aise",
        "clientèle",
        "problèmes",
        "synthèse",
        "etre",
        "très",
        "avec",
        "dans",
        "rapidement",
        "forte",
        "ecris",
        "tres",
        "voix",
        "capacites a",
        "etre",
        "equipe",
        "savoir",
        "preuve",
        "precision",
        "service",
        "bonne",
        "excellent",
        "analytique",
        "esprit d'",
        "excellentes",
        "capacite a",
        "maniere",
        "qualites d'",
        "sens du",
        "envie de",
        "prise",
        "office",
        "esprit",
        "dotée d'une",
        "doté d'un",
        "dotée d'un",
        "d'une",
        "fibre",
        "analyses",
        "dotee d'une",
        "dote d'un",
        "decison",
        "d'exploitation",
    ]
    sortie.update(
        soft_skills=[x for x in sortie["soft_skills"] if x.lower() not in common_eror]
    )

    sortie.update(
        post=filter_substring(
            sortie["post"],
            base_data[base_data["TAG"] == "B-POST"]["WORD"].to_list(),
        )
    )
    sorted_filtered_posts = sorted(
        filter(lambda post: post in sortie["post"], sortie["post"]),
        key=lambda post: text_cv.find(post),
    )

    sortie.update(post=filtrer_liste_postes(sorted_filtered_posts))

    sortie.update(post=clean_task(clean_task(clean_task(clean_task(sortie["post"])))))

    experiences, time = recreate_experience(sortie, text_cv)
    sortie.update(parcours_pro=experiences),

    predictor = chargement_hardfit_()

    sortie.update(job_type=predictor.predict(text_cv)[0][0])

    sortie.update(parcours_scolaire=recreate_experience_school(sortie, text_cv))

    sortie.update(time_in_field=time)

    sortie = traiter_skills_dict(sortie)
    print(sortie)
    return sortie


"""             Chargement du modèle de  MACRO FIT

 l'objectif de ce  modèle est de definir si les profils sont du meme secteur d'activité   """

"""                 Execution du  MICRO FIT

LE micro compare les details de chaque passage 

"""


def hskill_match(skill1: str, skill2: str) -> float | int:
    # omparateur = Word2Vec.load("word2vec2.model")
    try:
        model_comparateur = model_comparateur_()
        note = model_comparateur.wv.similarity(skill1, skill2)
        return note
    except:
        return -1


def equal(a: str, b: str) -> int:
    """Verification des cas d'éxactitudes d'orthographes car certaines compétences ne sont dans le vocabulaire du modèle Word2vec"""
    if a == b:
        return 1
    return 0


def correspondance_hskill(result_job: dict, result_resume: dict) -> dict:
    if result_job is None:
        result_job = []
    if result_resume is None:
        result_resume = []

    """Fonction qui recoit les deux listes de compétences (rechercher par l'offre et celle du potentiel candidat )  et renvoie deux listes les meilleurs match et  les matchs globales"""
    if len(result_job) == 0:
        print("ici")
        out = []
        line = {
            "skillrecherche": "pas de hardskills sur l'offre ",
            "skilldispo": " ",
            "match": int(1),
            "RGmatch": int(1),
        }
        out.append(line)
        return pd.DataFrame(out).to_dict(orient="list"), -1

    if len(result_resume) == 0:
        result_resume = ["pas de hardskills chez le candidat"]

    result_job = list(
        dict.fromkeys(
            [
                x.lower().replace(".", "") if x is not None else "olpss"
                for x in result_job
            ]
        )
    )

    result_resume = list(
        dict.fromkeys(
            [
                x.lower().replace(".", "") if x is not None else "olpsss"
                for x in result_resume
            ]
        )
    )

    out = []
    for skill_tohave in result_job:
        for skill_tosee in result_resume:
            line = {
                "skillrecherche": skill_tohave,
                "skilldispo": skill_tosee,
                "match": round(
                    max(
                        hskill_match(skill_tohave.lower(), skill_tosee.lower()),
                        equal(skill_tohave.lower(), skill_tosee.lower()),
                    ),
                    3,
                ),
            }
            out.append(line)

    frame = pd.DataFrame(out)
    """ On ne garde que le meilleur match pour chaque compétence """
    best = frame[
        frame.groupby(["skillrecherche"])["match"].transform(max) == frame["match"]
    ]

    # best.loc[best["match"] < 0.49, "match"] = 0
    best["RGmatch"] = 0
    best.loc[best["match"].between(0.50, 0.68, inclusive=True), "RGmatch"] = 0.2
    best.loc[best["match"].between(0.69, 0.99, inclusive=True), "RGmatch"] = 0.6
    best.loc[best["match"].between(1, 1, inclusive=True), "RGmatch"] = 1

    best = best.sort_values("RGmatch").groupby(["skillrecherche", "RGmatch"]).tail(1)

    # Tri et sélection des meilleures compétences par RGmatch
    best_sorted = (
        best.sort_values("RGmatch", ascending=False)
        .groupby(["skillrecherche", "RGmatch"])
        .tail(1)
    )

    # Sélection du top 6 après avoir regroupé et trié par RGmatch
    top_6 = best_sorted.head(6)

    # Calcul du score final sur les top 6 compétences sélectionnées
    score_final = top_6["RGmatch"].sum() / max(
        len(top_6), 1
    )  # Éviter la division par zéro

    return (
        best.sort_values("RGmatch", ascending=False)
        .groupby(["skillrecherche", "RGmatch"])
        .tail(1)
        .to_dict(orient="list"),
        score_final,
    )


################################## TRAITEMENT


def verif_langue(
    job_langue: pd.DataFrame, prospect_langue: pd.DataFrame
) -> tuple[float, dict]:
    """Donne une notation sur le niveau de langue  parametrer
    suivant toute les étapes

    """

    job_langue = traitement_langue(job_langue)
    prospect_langue = traitement_langue(prospect_langue)

    if job_langue.empty:
        return pd.DataFrame({"langue": [], "note": [], "note_prospect": []}), -1

    elif prospect_langue.empty:
        job_langue["note_prospect"] = 0

        #

        new = job_langue

        new["notation"] = [
            min(1, x / y) for x, y in zip(new["note_prospect"], new["note"])
        ]

        return new, (new["notation"].sum() / len(new)) * 100

    else:
        prospect_langue = prospect_langue.rename(
            columns={"langue": "langue", "note": "note_prospect"}, errors="raise"
        )

        new = job_langue.merge(prospect_langue, on="langue", how="left")

        new = new.fillna(0)
        new["notation"] = [
            1 if x >= y else 0 for x, y in zip(new["note_prospect"], new["note"])
        ]
        # new['notation']= 1 if  new['note_prospect']>=new['note'] else 0
        nouvelle_sortie = new.sort_values("note").drop_duplicates(
            ["langue"], keep="last"
        )
        nouvelle_sortie = nouvelle_sortie.sort_values("note", ascending=False)

        return (
            nouvelle_sortie,
            (nouvelle_sortie["notation"].sum() / len(nouvelle_sortie)) * 100,
        )


def traitement_langue(new_list: list) -> pd.DataFrame:
    """L'objectif de cette fonction est de mettre en forme les differentes de descriptifs de langue en echelle CECR -> exemple :"Anglais intermediare"----> "anglais":4"""

    if not new_list:
        return pd.DataFrame(columns=["langue", "note"])
    out = []

    for langue in new_list:
        taille = len(langue.split())
        langue_split = langue.split()

        if taille == 1:
            """cas ou on a la langue sans"""
            line = {"langue": langue_split[0].lower(), "note": 3}
            out.append(line)

        if taille >= 2:
            line = {
                "langue": langue_split[0].lower(),
                "note": equivalent_CERCR(
                    " ".join(langue_split[1:]).lower(),
                ),
            }

            out.append(line)

    return pd.DataFrame(out)


def equivalent_CERCR(qualif: str) -> int:
    """cette fonction attribut une équivalence chiffré à la qualification des niveaux de langue"""
    qualif = unidecode.unidecode(qualif.lower())
    return transcript_point(meilleur_CERCR(qualif))


def meilleur_CERCR(theme: str, ratio_min: int = 36) -> str:
    """renvoie de manière net le la meilleur qualif //// grossier mais fonctionnel"""
    candidats = {}
    thematique = [
        "lu et parle",
        "intermediaire",
        "intermediaire",
        "debutant",
        "moyen",
        "professionnel",
        "confirme",
        "a1",
        "a2",
        "b1",
        "b2",
        "c1",
        "c2",
        "niveau a1",
        "niveau a2",
        "niveau b1",
        "niveau b2",
        "niveau c1",
        "niveau c2",
        "scolaire",
        "maternel",
        "technique",
        "bilingue",
    ]
    for i in thematique:
        score = fuzz.ratio(i, theme)
        # print((i,theme),':',fuzz.ratio(i,theme))
        if score >= ratio_min:
            candidats[i] = score
        # prend le theme avec meilleur score
    if candidats == {}:
        # Si pas de thème trouver
        return ""
    return max(candidats, key=candidats.get)


def transcript_point(CERCR: str) -> int:
    """transforme la description du niveau de langue en note lisible"""

    CERCR = meilleur_CERCR(CERCR)

    if CERCR.lower() in (
        "lu et parle",
        "professionnel",
        "confirme",
        "c2",
        "niveau c2",
        "maternel",
        "courament",
        "bilingue",
    ):
        return 6

    elif CERCR in ("c1", "niveau c1"):
        return 5

    elif CERCR in ("intermediaire", "moyen", "b2", "niveau b2", "technique"):
        return 4

    elif CERCR in ("b1", "niveau b1"):
        return 3

    elif CERCR in ("scolaire", "a2", "niveau a1"):
        return 2
    elif CERCR in ("debutant", "a1", "niveau a1"):
        return 1

    else:
        return 3


def MEP_LANGUE(df_langues):

    # Groupby langue et obtenir le niveau max de note pour chaque langue
    df_max_notes = df_langues.groupby("langue").max().reset_index()

    # Inverser l'échelle CERCR pour obtenir le niveau correspondant en termes descriptifs
    echelle_CECR_inverse = {
        6: "niveau C2",
        5: "niveau C1",
        4: "niveau B2",
        3: "niveau B1",
        2: "niveau A2",
        1: "niveau A1",
    }
    # Appliquer l'échelle CECR inversée pour obtenir le niveau descriptif
    df_max_notes["niveau_CECR"] = df_max_notes["note"].map(echelle_CECR_inverse)

    return [
        f"{row['langue'].capitalize()} {row['niveau_CECR']}"
        for index, row in df_max_notes.iterrows()
    ]


def equivalent_CERCR(qualif: str) -> int:
    """cette fonction attribut une équivalence chiffré à la qualification des niveaux de langue"""
    qualif = unidecode.unidecode(qualif.lower())
    return transcript_point(meilleur_CERCR(qualif))


def meilleur_CERCR(theme: str, ratio_min: int = 36) -> str:
    """renvoie de manière net le la meilleur qualif //// grossier mais fonctionnel"""
    candidats = {}
    thematique = [
        "lu et parle",
        "intermediaire",
        "intermediaire",
        "debutant",
        "moyen",
        "professionnel",
        "confirme",
        "a1",
        "a2",
        "b1",
        "b2",
        "c1",
        "c2",
        "niveau a1",
        "niveau a2",
        "niveau b1",
        "niveau b2",
        "niveau c1",
        "niveau c2",
        "scolaire",
        "maternel",
        "technique",
        "bilingue",
    ]
    for i in thematique:
        score = fuzz.ratio(i, theme)
        # print((i,theme),':',fuzz.ratio(i,theme))
        if score >= ratio_min:
            candidats[i] = score
        # prend le theme avec meilleur score
    if candidats == {}:
        # Si pas de thème trouver
        return ""
    return max(candidats, key=candidats.get)


def transcript_point(CERCR: str) -> int:
    """transforme la description du niveau de langue en note lisible"""

    CERCR = meilleur_CERCR(CERCR)

    if CERCR.lower() in (
        "lu et parle",
        "professionnel",
        "confirme",
        "c2",
        "niveau c2",
        "maternel",
        "courament",
        "bilingue",
    ):
        return 6

    elif CERCR in ("c1", "niveau c1"):
        return 5

    elif CERCR in ("intermediaire", "moyen", "b2", "niveau b2", "technique"):
        return 4

    elif CERCR in ("b1", "niveau b1"):
        return 3

    elif CERCR in ("scolaire", "a2", "niveau a1"):
        return 2
    elif CERCR in ("debutant", "a1", "niveau a1"):
        return 1

    else:
        return 3

    #################################  DIPLOME LEVEL


#################################  DIPLOME LEVEL
def calcul_diplome(langue_job: pd.DataFrame, langue_prospect: pd.DataFrame) -> float:
    """synthétise les diplomes en note sur 100"""

    df_job = mise_en_forme_diplome(langue_job)
    df_job["note"] = pd.to_numeric(df_job["note"], errors="coerce")

    df_prospect = mise_en_forme_diplome(langue_prospect)
    df_prospect["note"] = pd.to_numeric(df_prospect["note"], errors="coerce")

    ##### GARDER SEULEMENT LES DIPLOMES ##########
    df_job = df_job[df_job["diplome_type"] == "diplome"]

    min_job = (
        df_job.nsmallest(1, "note")
        if not df_job[df_job["note"] > 0].empty
        else df_job[df_job["note"] > 0].nsmallest(1, "note")
    )
    min_job["note"] = pd.to_numeric(min_job["note"], errors="coerce")
    df_prospect = df_prospect[df_prospect["diplome_type"] == "diplome"]

    if df_job.empty:
        return -1

    elif df_job["note"].max() == 0:
        return -1
    if df_prospect.empty:
        return 0

    if df_prospect["note"].max() >= min_job["note"].iloc[0]:
        return 1
    else:
        return 0


def mise_en_forme_diplome(liste_diplome: list) -> pd.DataFrame:
    """Mets en forme les diplomes en mombres"""

    liste_diplome_lower = lower_list(liste_diplome)

    """ garde la liste des diplomes"""

    diplome = [
        "licence",
        "master",
        "bachelor",
        "bac",
        "permis",
        "but",
        "mastère",
        "bts",
        "dut",
        "rncp",
        "titre",
        "deug",
        "deust",
        "bachelor",
        "doctorat",
        "cap",
        "bep",
        "baccalaureat",
        "ingenieur",
    ]

    liste_filtrer = filter_use(liste_diplome_lower, diplome)

    """ le reste  non reconnue """

    reste_NR = set(liste_diplome_lower) - set(liste_filtrer)
    NR_list = list(reste_NR)

    pd_sortie = pd.DataFrame(columns=["diplome", "note", "diplome_type"])

    """ """

    liste_diplome = [
        [["cap", "bep"], "diplome", 3],
        [["bac", "baccalaureat"], "diplome", 4],
        [["deug", "dut", "deust", "bts"], "diplome", 5],
        [["bachelor", "licence", "but"], "diplome", 6],
        [["master", "ingenieur"], "diplome", 7],
        [["permis"], "mobilité", 0],
        [["doctorat", "phd"], "diplome", 8],
    ]

    for element in liste_diplome:
        temp = filter_use(liste_filtrer, element[0])

        for competence in temp:
            niveau = 0
            if "bac" in competence:
                niveau = bac_ou_pas(competence)

            new_row = {
                "diplome": competence,
                "note": max(element[2], niveau),
                "diplome_type": element[1],
            }
            pd_sortie = pd.concat(
                [pd_sortie, pd.DataFrame.from_records(new_row, index=[0])],
                ignore_index=True,
            )

    for element in NR_list:
        temp = filter_use(liste_filtrer, element[0])

        for competence in NR_list:
            new_row = {"diplome": competence, "note": 0, "diplome_type": "autre"}
            pd_sortie = pd.concat(
                [pd_sortie, pd.DataFrame.from_records(new_row, index=[0])],
                ignore_index=True,
            )
    # new_row = {"diplome": "nouveau", "note": 0, "diplome_type": "diplome"}
    # pd_sortie = pd.concat(
    #     [pd_sortie, pd.DataFrame.from_records(new_row, index=[0])],
    #     ignore_index=True,
    # )

    return pd_sortie


def lower_list(list_: list) -> list:
    """Recois une liste et la mets en minuscule et enleve les accents"""

    if list_ is not None:
        return [unidecode.unidecode(x.lower()) for x in list_]

    return []


def filter_use(string: str, substr: str) -> list:
    """Ne gardez que les diplomes"""
    string = lower_list(string)
    return [str for str in string if any(sub in str for sub in substr)]


def bac_ou_pas(texte: str) -> int:
    if "bac" in texte:
        num = re.findall(r"\d+", texte)
    if len(num) == 0:
        return 4

    diplome = int(max(num))

    if diplome == 2:
        temp = 5
    elif diplome == 3:
        temp = 6
    elif diplome == 4:
        temp = 6
    elif diplome == 5:
        temp = 7
    else:
        temp = 4
    return temp


def diploma(number_diploma: int) -> str:
    """attribuer à chacun  le diplome donc il dispose pour l'afficher dans le dashboaard"""

    if number_diploma == 3:
        return "Diplome de niveau 3 type CAP ,BEP"
    if number_diploma == 4:
        return "Diplome de niveau 4 type Baccalauréat"
    if number_diploma == 5:
        return "Diplome de niveau 5 type  DEUG,BTS,DUT,DEUST"
    if number_diploma == 6:
        return "Diplome de niveau 6 type  Licence ,Licence Pro , BUT ,Bachelor"
    if number_diploma == 7:
        return "Diplome de niveau 7 type  Master ou titre d'ingénieur"
    if number_diploma == 8:
        return "Diplome de niveau 8 type DOCTORANT"
    else:
        return "Pas de diplome ou non reconnu"


# SOFT SKILLS OU TACHE


def calcul(A, B):
    return float(np.dot(A, B) / (norm(A) * norm(B)))


def correspondance_sskill(result_job: dict, result_resume: dict) -> dict:
    """Fonction qui recoit les deux listes de compétences (rechercher par l'offre et celle du potentiel candidat )  et renvoie deux listes les meilleurs match et  les matchs globales"""
    out = []
    if result_job is None:
        result_job = []
    if result_resume is None:
        result_resume = []

    result_job = list(dict.fromkeys(result_job))
    result_resume = list(dict.fromkeys(result_resume))

    if not result_job:
        line = {
            "skillrecherche": "Pas de soft skill recherché",
            "skilldispo": "Aucun",
            "match": int(1),
        }
        out.append(line)

        best = pd.DataFrame(out)
        return (
            best.sort_values("match", ascending=False)
            .groupby(["skillrecherche", "match"])
            .tail(1)
            .to_dict(orient="list"),
            -1,
        )
    if not result_resume:
        result_resume = ["pas de soft skills"]
        for skill_tohave in result_job:
            line = {
                "skillrecherche": skill_tohave,
                "skilldispo": "pas de soft skills",
                "match": 0,
            }
            out.append(line)
        best = pd.DataFrame(out)
        return (
            best.sort_values("match", ascending=False).to_dict(orient="list"),
            0,
        )

    ########################## PARALELISATION #######################

    """ enleve les differentes duplicates """
    result_job = lower_list(list(dict.fromkeys([x.lower() for x in result_job])))
    result_resume = lower_list(dict.fromkeys([x.lower() for x in result_resume]))
    out = []
    model_sim = model_sim_()
    job_encoded = model_sim.encode(result_job).tolist()
    prospect_encode = model_sim.encode(result_resume).tolist()
    i = 0
    for taf in job_encoded:
        sortie = [calcul(taf, b) for b in prospect_encode]
        max(sortie)
        sortie.index(max(sortie))

        line = {
            "skillrecherche": result_job[i],
            "skilldispo": result_resume[sortie.index(max(sortie))],
            "match": max(sortie),
        }
        i = i + 1
        out.append(line)

    return ajuster_score_selon_correspondance_ajuste(
        out, seuil_correspondance=0.6, pourcentage_accepte=0.6, limite_competences=6
    )


def ajuster_score_selon_correspondance_ajuste(
    out, seuil_correspondance=0.55, pourcentage_accepte=0.7, limite_competences=6
):
    """
    Ajuste le score final en appliquant des règles spécifiques de modification des scores de correspondance,
    puis calcule la moyenne des scores ajustés en tenant compte d'un nombre limité de compétences.

    :param out: Liste des correspondances entre compétences recherchées et compétences disponibles.
    :param seuil_correspondance: Ignoré dans cette version ajustée de la fonction.
    :param pourcentage_accepte: Ignoré dans cette version ajustée de la fonction.
    :param limite_competences: Nombre maximum de compétences à considérer pour l'évaluation.
    :return: Tuple contenant le dictionnaire des meilleures correspondances ajustées et le score moyen ajusté.
    """
    best = pd.DataFrame(out)
    best = best.sort_values("match", ascending=False)

    # Appliquer les nouvelles règles de score
    def ajuster_score(score):
        if score > 0.8:
            return 1
        elif 0.6 < score <= 0.8:
            return 0.8
        elif score < 0.4:
            return 0
        else:
            return 0.5

    best["match_ajuste"] = best["match"].apply(ajuster_score)

    # Limiter aux 'limite_competences' meilleures compétences après ajustement
    if len(best) > limite_competences:
        best = best.nlargest(limite_competences, "match_ajuste")

    # Calculer la moyenne des scores ajustés
    score_final = best["match_ajuste"].mean()

    return best.to_dict(orient="list"), score_final


def encoder_et_obtenir_equivalences_dedupliquees(liste_textes):
    """
    Encode une liste de chaînes de caractères, retourne leurs équivalences,
    et déduplique les résultats.

    :param liste_textes: La liste des chaînes de caractères à encoder.
    :param model_sim: Le modèle utilisé pour encoder les chaînes de caractères.
    :param model_loaded: Le modèle utilisé pour prédire l'équivalence de l'encodage.
    :param fonction_equivalence: La fonction d'équivalence pour obtenir l'équivalence de la prédiction.
    :return: Liste dédupliquée des équivalences pour chaque chaîne de caractères encodée.
    """
    # Assurer que les textes sont en minuscules

    liste_textes = [texte.lower() for texte in liste_textes]

    # Encoder les chaînes de caractères
    encodes = model_sim.encode(liste_textes)
    encodes = [
        encode.reshape(1, -1) for encode in encodes
    ]  # Reshape pour correspondre à l'entrée attendue du modèle

    # Utiliser un dictionnaire pour stocker les équivalences et éviter les doublons
    equivalences = {}

    # Itérer sur chaque élément encodé pour obtenir l'équivalence
    for texte, encode in zip(liste_textes, encodes):
        prediction = model_loaded.predict(encode)[
            0
        ]  # Obtenir la prédiction pour l'encodage
        equivalence_ = equivalence(
            str(prediction)
        )  # Obtenir l'équivalence de la prédiction

        # Ajouter l'équivalence à l'ensemble s'il n'est pas déjà présent
        if equivalence_ not in equivalences:
            equivalences[equivalence_] = texte

    # Retourner la liste des clés du dictionnaire, qui sont les équivalences uniques
    return list(equivalences.keys())


def CR_SSKILL(result_job: dict, result_resume: dict) -> dict:
    """Fonction qui recoit les deux listes de compétences (rechercher par l'offre et celle du potentiel candidat )  et renvoie deux listes les meilleurs match et  les matchs globales"""
    out = []

    if not result_job:
        line = {
            "categorie": "Pas de soft skill recherché",
            "Indices": "Aucun",
            "match": int(1),
        }
        out.append(line)

        best = pd.DataFrame(out)
        return (
            best.sort_values("match", ascending=False)
            .groupby(["categorie", "match"])
            .tail(1)
            .to_dict(orient="list"),
            -1,
        )
    if not result_resume:
        result_resume = ["pas de soft skills"]
        for skill_tohave in result_resume:
            line = {
                "categorie": skill_tohave,
                "Indices": "pas de soft skills",
                "match": 0,
            }
            out.append(line)
        best = pd.DataFrame(out)
        return (
            best.sort_values("match", ascending=False)
            .groupby(["categorie", "match"])
            .tail(1)
            .to_dict(orient="list"),
            0,
        )

    ########################## PARALELISATION #######################

    """ enleve les differentes duplicates """

    result_job = lower_list(list(dict.fromkeys([x.lower() for x in result_job])))
    result_resume = lower_list(dict.fromkeys([x.lower() for x in result_resume]))
    out = []
    out_job = []
    out_prospect = []
    model_sim = model_sim_()
    job_encoded = model_sim.encode(result_job)
    prospect_encode = model_sim.encode(result_resume)
    job_encoded = [x.reshape(1, -1) for x in job_encoded]
    prospect_encode = [x.reshape(1, -1) for x in prospect_encode]

    model_loaded = model_loaded_()
    for ev in job_encoded:
        out_job.append(model_loaded.predict(ev)[0])

    for ev in prospect_encode:
        out_prospect.append(model_loaded.predict(ev)[0])

    retour = []

    out_job_eq_name = [equivalence(str(x)) for x in out_job]
    out_prospect_eq_name = [equivalence(str(x)) for x in out_prospect]
    ######### retirer les doublons

    out_job_eq_name = list(dict.fromkeys(out_job_eq_name))

    print(out_job_eq_name, out_prospect_eq_name, out_job, out_prospect)

    for i in out_job_eq_name:
        if i in out_prospect_eq_name:
            line = {
                "categorie": i,
                "Indices": "/".join(
                    [
                        result_resume[indice]
                        for indice in [
                            x
                            for x in range(len(out_prospect_eq_name))
                            if out_prospect_eq_name[x] == i
                        ]
                    ]
                ),
                "match": 1,
            }
            retour.append(line)

        else:
            line = {
                "categorie": i,
                "Indices": "",
                "match": 0,
            }
            retour.append(line)

    df = pd.DataFrame(retour)
    df = df.sort_values("match", ascending=False)
    df_top_4 = df.head(4)
    note = df_top_4.match.sum() / len(df_top_4)
    return (df.to_dict(orient="list"), note)


def soft_comparaison(sentences1: str, sentences2: str) -> float:
    model_sim = model_sim_()
    A = model_sim.encode(sentences1)
    B = model_sim.encode(sentences2)

    return float(np.dot(A, B) / (norm(A) * norm(B)))


def equivalence(grade):
    if grade == "0":
        return "SENS DE LA CLIENTELE"
    elif grade == "1":
        return "PRISE D'INITIATIVE"
    elif grade == "2":
        return "PRISE DE DECISION"
    elif grade == "3":
        return "ORGANISE"
    elif grade == "4":
        return "A L'ECOUTE"
    elif grade == "5":
        return "SENS DE LA COMMUNICATION"
    elif grade == "6":
        return "FORCE DE PROPOSITION"
    elif grade == "7":
        return "TRAVAILLEUR"
    elif grade == "8":
        return "RIGOUREUX"
    elif grade == "9":
        return "SENS DES RESPONSABILITES"
    elif grade == "10":
        return "ESPRIT D'EQUIPE"
    elif grade == "11":
        return "CAPACITE DE SYNTHESE"
    elif grade == "12":
        return "BON RELATIONNEL"
    elif grade == "13":
        return "SENS DE L'ACTION"
    elif grade == "14":
        return "PERSEVERANT"
    elif grade == "15":
        return "AGREABLE"
    elif grade == "16":
        return "ENGAGE"
    elif grade == "17":
        return "ADAPTABILITE"
    elif grade == "18":
        return "AUTOMIE"
    elif grade == "19":
        return "GOUT DU CHALLENGE"
    elif grade == "20":
        return "SENS DU DETAIL"
    elif grade == "21":
        return "PASSION/MOTIVATION"
    elif grade == "22":
        return "AISANCE REDACTIONNELLE"
    elif grade == "23":
        return "AISANCE ORALE"
    elif grade == "24":
        return "CAPACITE D'ANALYSE"
    elif grade == "25":
        return "ESPRIT CRITIQUE"
    elif grade == "26":
        return "CURIEUX"


import tempfile
import re


def niveau_experience(nombre_annees):
    niveau_experience = "Non spécifié"

    if nombre_annees < 3:
        niveau_experience = "Junior"
    elif nombre_annees < 7:
        niveau_experience = "Intermédiaire"
    elif nombre_annees >= 7:
        niveau_experience = "Senior"

    return niveau_experience


def comparer_niveau_experience(niveau1, niveau2):

    # Convertir les niveaux d'expérience en valeurs numériques pour faciliter la comparaison
    niveau_numerique = {"junior": 1, "intermédiaire": 2, "senior": 3}

    # Obtenir la valeur numérique pour chaque niveau
    val_niveau1 = niveau_numerique.get(str(niveau1).lower(), 0)
    val_niveau2 = niveau_numerique.get(str(niveau2).lower(), 0)

    # Si le niveau1 est supérieur ou égal au niveau2, retourner 100, sinon 0
    if val_niveau1 >= val_niveau2:
        return 100, (niveau1, niveau2)
    else:
        return 0, (niveau1, niveau2)


def matching_offer_with_candidat(result_cv: dict, result_job_offer: dict):

    tache, note_tache = correspondance_sskill(
        result_job_offer["tache"], result_cv["tache"]
    )
    hskill, note_hskill = correspondance_hskill(
        result_job_offer["hard_skills"],
        result_cv["hard_skills"],
    )
    langue_df, note_langue = verif_langue(
        result_job_offer["langue"], result_cv["langue"]
    )
    note_diplome = calcul_diplome(result_job_offer["diplome"], result_cv["diplome"])

    df_sskill, note_sskill = CR_SSKILL(
        result_job_offer["soft_skills"], result_cv["soft_skills"]
    )

    note_exp, detail_exp = comparer_niveau_experience(
        niveau_experience(result_cv["time_in_field"]), result_job_offer["time_in_field"]
    )
    job = mise_en_forme_diplome(result_job_offer["diplome"])
    if result_job_offer["job_type"] == result_cv["job_type"]:
        match_type = 1
    else:
        match_type = 0
    job["note"] = pd.to_numeric(job["note"], errors="coerce")
    job_ = (
        job.nsmallest(1, "note")
        if not job[job["note"] > 0].empty
        else job[job["note"] > 0].nsmallest(1, "note")
    )
    if not job_.empty:
        # Votre DataFrame n'est pas vide; vous pouvez accéder à l'élément.
        required_diploma = job_["note"].iloc[0]
    else:
        # Votre DataFrame est vide; gérez ce cas en conséquence.
        required_diploma = 0

    line = {
        "Talent_name": result_job_offer["nom_prenom"],
        "TALENT_MAIL": result_job_offer["mail"],
        "TALENT_NUMERO": result_job_offer["numero"],
        "TALENT_STACK": result_job_offer["hard_skills"],
        "TALENT_SOFT": result_job_offer["soft_skills"],
        "TALENT_DIPLOME": result_job_offer["diplome"],
        "TALENT_LANGUE": result_job_offer["langue"],
        "TALENT_TACHE": result_job_offer["tache"],
        "TYPE_MATCH": match_type,
        "TALENT_TYPE": result_cv["job_type"],
        "JOB_TYPE": result_job_offer["job_type"],
        # "TALENT_EXP": result_cv["parcours_pro"],
        # "TALENT_TIME_FIELD": result_cv["time_in_field"],
        # "TALENT_EXP": result_cv["parcours_pro"],
        ####### INFORMATION SUR LE TALENT ########
        "DETAIL_LANGUE": langue_df.to_dict(),
        "TALENT_LANGUE_PT": traitement_langue(result_job_offer["langue"]).to_dict(),
        "JOB_LANGUE": traitement_langue(result_job_offer["langue"]).to_dict(),
        "MATCH_SCORE": calculer_match_score(
            note_hskill, note_tache, note_sskill, note_langue, note_diplome, note_exp
        ),
        "DETAIL_EXP": detail_exp,
        "STACKS": hskill,
        "LANGUE": langue_df.to_dict(),
        "MAX_DIPLOMA_CANDIDAT": required_diploma,
        "REQUIERED_DIPLOMA": mise_en_forme_diplome(result_cv["diplome"])["note"].max(),
        "TACHE": tache,
        "LINK": result_job_offer["id_pdf_hashed"],
        "DETAIL_NOTE_SSKILL": df_sskill,
        "NOTE_SSKILL": note_sskill * 100,
        "NOTE_STACK": note_hskill * 100,
        "NOTE_TACHE": note_tache * 100,
        "NOTE LANGUE": note_langue,
        "NOTE_DIPLOME": note_diplome * 100,
        "NOTE_EXP": note_exp,
    }
    return line


def calculer_match_score(
    note_hskill, note_tache, note_sskill, note_langue, note_diplome, note_exp
):
    # print("VIVE LES PIEDS")
    # Remplacer NaN par -1 pour chaque note
    notes = [
        float(note) if not math.isnan(note) else float(-1)
        for note in [
            note_hskill,
            note_tache,
            note_sskill,
            note_langue,
            note_diplome,
            note_exp,
        ]
    ]

    poids = {
        "hskill": 0.25,
        "tache": 0.25,
        "sskill": 0.2,
        "langue": 0.1,
        "diplome": 0.1,
        "experience": 0.1,
    }  # Exemple de poids pour chaque critère

    # Ajuster le poids si la note est négative (l'offre ne demande pas cet élément)
    poids_hskill = poids["hskill"] if notes[0] >= 0 else 0
    poids_tache = poids["tache"] if notes[1] >= 0 else 0
    poids_sskill = poids["sskill"] if notes[2] >= 0 else 0
    poids_langue = poids["langue"] if notes[3] >= 0 else 0
    poids_diplome = poids["diplome"] if notes[4] >= 0 else 0
    poids_exp = poids["experience"] if notes[5] >= 0 else 0

    # S'assurer que la somme des poids ajustés est égale à 1
    somme_poids = (
        poids_hskill
        + poids_tache
        + poids_sskill
        + poids_langue
        + poids_diplome
        + poids_exp
    )
    if somme_poids == 0:
        return 0  # Éviter la division par zéro si tous les éléments sont non demandés

    # Calcul du score avec les poids ajustés
    match_score = (
        ((notes[0] * 100) * poids_hskill)
        + ((notes[1] * 100) * poids_tache)
        + ((notes[2] * 100) * poids_sskill)
        + ((notes[3]) * poids_langue)
        + ((notes[4] * 100) * poids_diplome)
        + ((notes[5]) * poids_exp)
    ) / somme_poids

    return match_score


def afficher_avis(score):
    if score >= 90:
        message = "Votre profil est parfait pour cette opportunité ! Nous vous recommandons vivement de postuler."
        couleur_degrade = "rgba(56,181,165,255)"
    elif 70 <= score < 90:
        message = "Votre profil présente une bonne synergie avec cette opportunité. Vous êtes encouragé à postuler."
        couleur_degrade = "rgba(56,181,165,255)"
    elif 65 <= score < 70:
        message = "Votre profil est assez intéressant pour cette opportunité, mais il manque quelques compétences clés. À envisager."
        couleur_degrade = "#ff7f00"
    elif 50 <= score < 65:
        message = "Votre profil pourrait convenir, mais il nécessite un approfondissement sur certaines compétences requises."
        couleur_degrade = "#ff7f00"
    elif 40 <= score < 50:
        message = "Votre profil présente quelques écarts avec les exigences de l'opportunité. Une mise à niveau serait nécessaire."
        couleur_degrade = "#FA8072"
    else:
        message = "Votre profil n'est pas recommandé pour cette opportunité. Il serait préférable de cibler des postes plus adaptés à vos compétences."
        couleur_degrade = "#FA8072"

    html_content = f"""
    <div style='background: linear-gradient(to right, {couleur_degrade}, {couleur_degrade}); padding: 10px; border-radius: 5px; border: 1px solid #fff;'>
        <p style='color: white; font-size: 0.8vw; text-align: center;'>{message}</p>
    </div>
    """

    st.markdown(html_content, unsafe_allow_html=True)


def afficher_avis_new(
    score, score_task, score_langue, score_diplome, score_stack, score_exp, score_soft
):

    if score >= 90:
        message = "Opportunitée adaptée."
        couleur_degrade = "rgba(56,181,165,255)"
    elif 70 <= score < 90:
        message = "L'opportunitée semble etre à votre avantage."
        couleur_degrade = "rgba(56,181,165,255)"
    elif 65 <= score < 70:
        message = "Opportunitée à envisager."
        couleur_degrade = "#ff7f00"
    elif 50 <= score < 65:
        message = "Opportunitée à envisager avec des points qui peuvent etre bloquant"
        couleur_degrade = "#ff7f00"
    elif 40 <= score < 50:
        message = "De gros écart avec l'attendue de l'opportunitée."
        couleur_degrade = "#FA8072"
    else:
        message = "Opportunitée inadaptée."
        couleur_degrade = "#FA8072"
    # Définir les messages et les couleurs selon les scores
    message_general = message

    # Générer des listes de points forts et faibles
    categories_scores = {
        "Vos tâches réalisées": score_task,
        "Votre niveau de langue": score_langue,
        "Votre parcours scolaire": score_diplome,
        "Vos outils utilisés": score_stack,
        "Votre niveau d'éxperience": score_exp,
        "Vos compétences humaines ": score_soft,
    }
    points_forts = [
        cat for cat, score in categories_scores.items() if score >= 50 and score >= 0
    ]
    points_faibles = [
        cat for cat, score in categories_scores.items() if score < 50 and score >= 0
    ]

    # Construire le HTML pour les points forts et faibles
    points_forts_html = "".join(
        [f"<li style='color:green;'>{point}</li>" for point in points_forts]
    )
    points_faibles_html = "".join(
        [f"<li style='color:red;'>{point}</li>" for point in points_faibles]
    )

    # HTML Template
    html_template = f"""
    <style>
        .info-box {{
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #ccc;
            text-align:center;
            border: 3px solid {couleur_degrade};
        }}
        .points-forts {{
            background-color: #ddffdd;
        }}
        .points-faibles {{
            background-color: #ffdddd;
        }}
    </style>
    <div class="info-box">
        <p style='font-size: 0.9vw; text-align: center;font-weight: 600;color:{couleur_degrade};' >{message_general}</p>
        <details class="points-forts">
            <summary style='font-weight: 600;' >Vos atouts</summary>
            <ul>{points_forts_html}</ul>
        </details>
        <details class="points-faibles">
            <summary style='font-weight: 600;' >Vos points d'attentions</summary>
            <ul>{points_faibles_html}</ul>
        </details>
    </div>
    """

    # Utiliser st.markdown pour afficher le contenu HTML/CSS
    st.markdown(html_template, unsafe_allow_html=True)


def parsing_joboffer_local(text_cv: str) -> dict:
    # text_cv=text_cv.encode("ascii", "ignore").decode()
    """concatenation de toutes les fonctions et creation de la sortie"""
    listed_text_cv = [
        x for x in list(filter(None, transform(text_cv))) if not has_useless_text(x)
    ]

    predictor = chargement_hardfit_()

    base = [listed_text_cv[i : i + 32] for i in range(0, len(listed_text_cv), 32)]

    Parseur = model_parseur_()

    predict_text, raw = Parseur.predict(base, split_on_space=False)

    base_data = MEF_alternatif_1(list(itertools.chain.from_iterable(predict_text)))
    # display(base_data)
    base_data["tf"] = base_data["WORD"].apply(lambda x: has_useless_text(x))
    base_data = base_data[["WORD", "TAG"]]  # [base_data["tf"] == False]

    final_version = collapse(base_data)

    sortie = MEF_final(final_version)

    sortie.update(
        raw_text=text_cv,
    )

    sortie.update(
        numero=extract_telephone(text_cv),
    )

    sortie.update(
        nom_prenom=combine_consecutive_words_in_text(text_cv, sortie["nom_prenom"])
    )

    sortie.update(mail=re.findall(r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+", text_cv))

    sortie.update(
        langue=filter_substring(
            sortie["langue"],
            base_data[base_data["TAG"] == "B-LANGUE"]["WORD"].to_list(),
        )
    )

    sortie.update(
        tache=clean_list(
            filter_substring(
                sortie["tache"],
                base_data[base_data["TAG"] == "B-TACHE"]["WORD"].to_list(),
            )
        )
    )

    sortie.update(id_pdf_hashed=hashlib.md5(text_cv.encode("utf-8")).hexdigest())

    sortie.update(
        tache=clean_task(clean_task(list(dict.fromkeys(clean_task(sortie["tache"])))))
    )

    sortie.update(
        diplome=filter_substring(
            sortie["diplome"],
            base_data[base_data["TAG"] == "B-DIPLÔME"]["WORD"].to_list(),
        )
    )

    sortie.update(
        diplome=clean_task(clean_task(clean_task(clean_task(sortie["diplome"]))))
    )

    sortie.update(ecole=clean_task(clean_task(clean_task(clean_task(sortie["ecole"])))))

    sortie.update(soft_skills=clean_task(clean_task(clean_task(sortie["soft_skills"]))))

    sortie.update(
        post=filter_substring(
            sortie["post"],
            base_data[base_data["TAG"] == "B-POST"]["WORD"].to_list(),
        )
    )

    sortie.update(post=clean_task(clean_task(clean_task(clean_task(sortie["post"])))))
    sortie.update(parcours_pro=recreate_experience(sortie, text_cv)),

    sortie.update(job_types=predictor.predict(text_cv)[0][0])

    sortie = traiter_skills_dict(sortie)

    common_eror = [
        "qualité",
        "observation",
        "orthographique",
        "nature",
        "active",
        "prendre",
        "faire",
        "haute",
        "efficacement",
        "expression",
        "stress",
        "capacités",
        "équipe",
        "etre",
        "sens",
        "travail",
        "force",
        "esprit",
        "analyse",
        "apprendre",
        "capacité",
        "travailler",
        "realation",
        "oral",
        "ecris",
        "aise",
        "clientèle",
        "problèmes",
        "synthèse",
        "etre",
        "très",
        "avec",
        "dans",
        "rapidement",
        "forte",
        "ecris",
        "tres",
        "voix",
        "capacites a",
        "etre",
        "equipe",
        "savoir",
        "preuve",
        "precision",
        "service",
        "bonne",
        "excellent",
        "analytique",
        "esprit d'",
        "excellentes",
        "capacite a",
        "maniere",
        "qualites d'",
        "sens du",
        "envie de",
        "prise",
        "office",
        "esprit",
        "dotee d'une",
        "extremement",
    ]
    sortie.update(
        soft_skills=[x for x in sortie["soft_skills"] if x.lower() not in common_eror]
    )

    sortie.update(
        time_in_field=time_exp_in_field(sortie["parcours_pro"], sortie["job_types"])
    )

    # sortie.update(parcours_scolaire=recreate_school_exp(sortie, text_cv))

    return sortie


def time_exp_in_field(base_tech, field):
    # Initialiser la somme des valeurs de estimation_time pour dev-log
    total_estimation_time = 0

    # Parcourir la liste et agréger les valeurs de estimation_time pour dev-log
    for item in base_tech:
        classe = item["classe"]
        estimation_time = item["estimation_time"]

        # Ajouter la valeur de estimation_time uniquement si la classe est dev-log et estimation_time > 0
        if classe == field and estimation_time > 0:
            total_estimation_time += estimation_time
    return total_estimation_time


def combine_consecutive_words_in_text(text, word_list):
    # Check if word_list is None and return an empty list or handle appropriately
    if word_list is None:
        return []

    # Remove newline characters from the text
    cleaned_text = text.replace("\n", " ")

    combined_list = []
    i = 0
    words_to_combine = set(
        word_list
    )  # Now safe to use since we've handled the case where word_list could be None

    while i < len(word_list):
        current_word = word_list[i]

        # Check if the current word and the next word are consecutive in the cleaned text
        if (
            i + 1 < len(word_list)
            and f"{current_word} {word_list[i + 1]}" in cleaned_text
        ):
            combined_list.append(f"{current_word} {word_list[i + 1]}")
            i += 2
        else:
            combined_list.append(current_word)
            i += 1

    return combined_list


def concat_names_if_followed(word_list, text):
    result_list = text.split()
    positions = [result_list.index(word) for word in word_list]

    concatenated_words = []

    for i in range(len(positions) - 1):
        current_pos = positions[i]
        next_pos = positions[i + 1]

        if current_pos + 1 == next_pos:
            concatenated_word = f"{result_list[current_pos]} {result_list[next_pos]}"
            concatenated_words.append(concatenated_word)
            result_list[current_pos] = concatenated_word
            result_list[next_pos] = ""

    # Supprimer les mots concaténés de la liste de départ
    word_list = [word for word in word_list if word not in " ".join(result_list)]

    return word_list + concatenated_words


def trouver_region_gagnante(vecteur_a_trouver):
    # Charger le fichier cities.json une seule fois
    with open("cities.json", "r") as file:
        cities_data = json.load(file)

    # Seuil de ressemblance
    seuil_ressemblance = 90

    # Dictionnaire pour stocker le nombre de voix par région
    voix_par_region = {}

    # Parcourir chaque élément dans le vecteur à trouver
    for element in vecteur_a_trouver:
        # Parcourir les données des villes dans le fichier JSON
        for city in cities_data["cities"]:
            score_total = fuzz.partial_ratio(city["label"].lower(), element.lower())

            # Si le score est supérieur au seuil, considérez la région
            if score_total >= seuil_ressemblance:
                region_correspondante = city["region_name"]

                # Ajouter une voix à la région correspondante
                voix_par_region[region_correspondante] = (
                    voix_par_region.get(region_correspondante, 0) + 1
                )

    # Trouver la région avec le plus de voix
    region_gagnante = (
        max(voix_par_region, key=voix_par_region.get) if voix_par_region else " "
    )

    return region_gagnante


from google.cloud import storage
from google.oauth2 import service_account
from io import BytesIO


import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import re
import pickle
from annotated_text import annotated_text
import hashlib
import yaml
from yaml.loader import SafeLoader
import streamlit as st
import streamlit.components.v1 as components
import hydralit_components as hc
import psycopg2


def match_visualisation(sortie):
    # st.write(st.session_state)
    sortie = sorted(sortie, key=lambda x: x["MATCH_SCORE"], reverse=True)

    # DEBUT DE  DE L'APP
    def Graph_Langue(dictionaire):
        """sortie du graphe des compétences prospects"""
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                # y=list(dictionaire["langue"].values()),
                # x=list(dictionaire["note"].values()),
                x=list(dictionaire["langue"].values()),
                y=list(dictionaire["note"].values()),
                name="Niveau souhaité",
                # orientation="h",
                marker=dict(
                    color="rgb(114,118,122)",
                    line=dict(color="rgb(114,118,122)", width=3),
                ),
            )
        )
        fig.add_trace(
            go.Bar(
                x=list(dictionaire["langue"].values()),
                y=list(dictionaire["note_prospect"].values()),
                name="Niveau du Talent",
                # orientation="h",
                marker=dict(
                    color="rgb(138,164,197)",
                    line=dict(color="rgb(138,164,197)", width=3),  # here
                ),
            )
        )

        fig.update_polars(radialaxis=dict(range=[0, 1]))

        return fig.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=[1, 2, 3, 4, 5, 6],
                ticktext=[
                    "Niveau A1",
                    "Niveau A2",
                    "Niveau B1",
                    "Niveau B2",
                    "Niveau C1",
                    "Niveau C2",
                ],
            ),
            yaxis_range=[0, 6],
        )

    def affichage_scatter(candidate_skill: dict):
        fig = go.Figure(
            data=go.Scatterpolar(
                r=candidate_skill["RGmatch"],
                theta=candidate_skill["skillrecherche"],
                text=candidate_skill["skilldispo"],
                fill="toself",
                fillcolor="rgb(138,164,197)",
                texttemplate="plotly_dark",
            )
        )

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                ),
            ),
            showlegend=False,
        )
        fig.update_polars(radialaxis=dict(range=[0, 1]))

        return fig

    def Graph_Diplome(list1, list2):
        """sortie du graphe des compétences prospects"""
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                # y=list(dictionaire["langue"].values()),
                # x=list(dictionaire["note"].values()),
                x=list1[0],
                y=list1[1],
                name=list1[0][0],
                # orientation="h",
                marker=dict(
                    color="rgb(138,164,197)",
                    line=dict(color="rgb(138,164,197)", width=3),
                ),
            )
        )
        fig.add_trace(
            go.Bar(
                x=list2[0],
                y=list2[1],
                name=list2[0][0],
                # orientation="h",
                marker=dict(
                    color="rgb(114,118,122)",
                    line=dict(color="rgb(114,118,122)", width=3),
                ),
            )
        )

        return fig.update_layout(
            yaxis=dict(
                tickmode="array",
                tickvals=[0, 3, 4, 5, 6, 7, 8],
                ticktext=[
                    "Aucun diplôme ou non reconu",
                    "Equiv. CAP /BEP",
                    "Equiv. BAC",
                    "Equiv. BTS/DUT/DEUG BAC +2",
                    "Equiv. Licence BAC +3",
                    "Equiv. Master/ing BAC +5",
                    "Equiv. Doctorat Bac+8",
                ],
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_range=[0, 8],
            margin=dict(l=0, r=0, t=1, b=0),
        )

    def Graph_Comparaison_Niveau_avec_Liste(niveaux):
        """Génère un histogramme comparant le niveau du candidat et de l'offre d'emploi.

        Les niveaux sont exprimés en texte dans la liste d'entrée et convertis en valeurs numériques pour le graphique.

        Args:
            niveaux (list): Liste contenant deux chaînes de caractères, représentant le niveau du candidat et celui de l'offre.

        Returns:
            fig (go.Figure): Figure Plotly représentant l'histogramme.
        """

        niveau_numerique = {"Junior": 1, "Intermédiaire": 4, "Senior": 8}
        niveau_candidat = niveaux[0]
        niveau_offre = niveaux[1]

        # Conversion des niveaux textuels en valeurs numériques
        val_candidat = niveau_numerique.get(niveau_candidat, 0)
        val_offre = niveau_numerique.get(niveau_offre, 0)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=["Vous"],
                y=[val_candidat],
                name="Vous",
                marker=dict(color="rgb(138,164,197)"),
            )
        )
        fig.add_trace(
            go.Bar(
                x=["""Expérience requise"""],
                y=[val_offre],
                name="Expérience requise",
                marker=dict(color="rgb(114,118,122)"),
            )
        )

        fig.update_layout(
            yaxis=dict(
                title="Niveau",
                tickvals=[1, 4, 8],
                ticktext=["Junior", "Intermédiaire", "Senior"],
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        return fig

    def affiche_competences(candidat):
        st.divider()
        compt = 0
        for a, b, c in zip(
            candidat["DETAIL_NOTE_SSKILL"]["categorie"],
            candidat["DETAIL_NOTE_SSKILL"]["Indices"],
            candidat["DETAIL_NOTE_SSKILL"]["match"],
        ):
            if c >= 0 and compt < 5:  # Limite à 5 éléments
                if c == 0:
                    b = "Aucune correspondance"
                    # Utilisez le texte annoté avec la couleur rouge pour un match de 0%
                    annotated_text(
                        (a, "Rechercher", "#999"),
                        (b.capitalize(), "Disponible", "#faa"),
                        (str(int(c * 100)) + " %", "**Tx**", "#faa"),
                    )
                else:
                    # Utilisez le texte annoté avec la couleur verte pour les autres matches
                    annotated_text(
                        (a, "Rechercher", "#999"),
                        (b.capitalize(), "Disponible", "#afa"),
                        (str(int(c * 100)) + " %", "**Tx**", "#afa"),
                    )
                st.divider()
                compt += 1

    def affiche_taches(candidat):
        st.markdown("#")

        # candidat.sort_values("match", ascending=False)

        st.divider()
        compt = 0
        for a, b, c in zip(
            candidat["TACHE"]["skillrecherche"],
            candidat["TACHE"]["skilldispo"],
            candidat["TACHE"]["match"],
        ):
            if c >= 0.4 and compt < 5:  # Limite à 5 éléments et vérifie si c > 0
                # Couleur pour "Match" basée sur la condition de c
                match_color = "#afa" if c > 0.5 else "#faa"

                annotated_text(
                    (a.capitalize(), "Rechercher", "#aaa"),  # Gris pour "Rechercher"
                    (
                        b.capitalize(),
                        "Disponible",
                        "#fff",
                    ),  # Blanc pour "Disponible", changez selon le besoin
                    (
                        str(int(c * 100)) + " %",
                        "**Tx**",
                        match_color,
                    ),  # Couleur conditionnelle pour "Match"
                )
                st.divider()
                compt += 1

    def taux_remplissage(
        note_tache, note_langue, note_diplome, note_stack, note_exp, note_sskill
    ):
        scores = [
            note_tache,
            note_langue,
            note_diplome,
            note_stack,
            note_exp,
            note_sskill,
        ]

        # Compter le nombre de scores qui ne sont pas renseignés (note négative)
        non_renseignes = sum(1 for score in scores if score < 0)

        # Vérifier si 4 éléments ou moins ne sont pas renseignés
        if (
            non_renseignes >= 2
        ):  # Si 4 ou moins sont renseignés, alors 2 ou plus ne le sont pas
            message = "❌ Attention manque d'information sur l'offre."
            couleur = "orange"
        else:
            message = "✅ Offre suffisamment renseignée."
            couleur = "green"

        # Afficher le message dans Streamlit
        st.markdown(
            f"<div style='color: {couleur};font-size:0.6vw;text-align:center;'>{message}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("#")

    def score_color(note):
        if note < 40:
            return "rgb(244,67,54)"
        elif 40 <= note < 60:
            return "rgb(255,87,34)"
        elif 60 <= note < 78:
            return "rgb(76,175,80)"  # Vert pour les notes moyennes
        elif 78 <= note < 90:
            return "rgb(139,195,74)"
        else:
            return "rgb(76,175,80)"

    colonne_resultat, colonne_technique, colonne_middle, colonne_right = st.columns(
        (0.2, 0.27, 0.26, 0.26), gap="large"
    )

    with colonne_resultat:
        # st.write(sortie)
        ###### CHOIX DU CANDIDAT ########
        # option = st.selectbox("Liste de Talent", (range(0, len(sortie))))

        # candidat = sortie[option]
        if st.button("Retour au choix des opportunitées"):
            st.session_state["matching_stage"] = 1
            st.session_state["indices_selectionnes"] = []
            st.session_state.result_matching = {}
            offres_emploi = st.session_state["offres_emploi"]
            selected = "Job Finder"
            del st.session_state.parsed_offres
            del st.session_state.region_selectionnee

            # Utilisez st.rerun() pour forcer le script à s'exécuter à nouveau depuis le déb
            st.rerun()
        # st.write(sortie)
        noms_talents = [talent["Talent_name"] for talent in sortie]

        ###### CHOIX DU CANDIDAT ########
        option = st.selectbox("**Liste de Poste**", noms_talents)

        # Trouver le candidat sélectionné
        candidat_index = noms_talents.index(option)
        candidat = sortie[candidat_index]

        # st.write(candidat)
        ####### RESULTAT #################
        st.markdown("#")

        taux_remplissage(
            candidat["NOTE_TACHE"],
            candidat["NOTE LANGUE"],
            candidat["NOTE_DIPLOME"],
            candidat["NOTE_STACK"],
            candidat["NOTE_EXP"],
            candidat["NOTE_SSKILL"],
        )

        html_diplome = f"""
            <h6 title='Score de compatibilité' 
                style='text-align: center; color:rgb(44,63,102);font-size: 1.8vw;'>
                NESTFIT
            </h6> <br> 
        """
        st.markdown(html_diplome, unsafe_allow_html=True)

        def generate_gauge(note):
            if math.isnan(note):
                # Handle NaN value. Example: Set note to 0 or another default value, or raise an exception
                note = 0

            gradient_color = "rgba(56,181,165,255)"  # Couleur par défaut (vert)

            if note < 50:
                gradient_color = "#FA8072"  # Rouge pour les notes inférieures à 50
                text_color = "#FA8072"
            elif 50 <= note < 70:
                gradient_color = "#ffcb60"  # Orange pour les notes entre 50 et 70
                text_color = "#ffcb60"
            else:
                text_color = "rgba(56,181,165,255)"

            html_str = f"""
                    <style>
                        .gauge-container {{
                            width: 150px;
                            height: 150px;
                            position: relative;
                            margin: 0 auto;
                        }}
                        .gauge {{
                            width: 100%;
                            height: 100%;
                            background: conic-gradient({gradient_color} {int(note)}%, #fff {int(note)}% 100%);
                            border-radius: 50%;
                            position: absolute;
                            border: 2px solid #000;
                        }}
                        .gauge::before {{
                            content: '';
                            width: 80%;
                            height: 80%;
                            background-color: #fff;
                            border-radius: 50%;
                            position: absolute;
                            top: 10%;
                            left: 10%;
                        }}
                        .gauge-text {{
                            position: absolute;
                            top: 50%;
                            left: 50%;
                            transform: translate(-50%, -50%);
                            font-size: 2.1vw;
                            font-weight: bolder;
                            color: {text_color};
                            z-index: 1;
                        }}
                    </style>
                    <div class="gauge-container">
                        <div class="gauge"></div>
                        <div class="average-indicator"></div>
                        <p class="gauge-text ">{str(int(note))}</p>
                    </div>
                """
            return html_str

        st.markdown(generate_gauge(candidat["MATCH_SCORE"]), unsafe_allow_html=True)
        st.markdown("#")

        score = candidat["MATCH_SCORE"]

        # afficher_avis(score)

        afficher_avis_new(
            score,
            candidat["NOTE_TACHE"],
            candidat["NOTE LANGUE"],
            candidat["NOTE_DIPLOME"],
            candidat["NOTE_STACK"],
            candidat["NOTE_EXP"],
            candidat["NOTE_SSKILL"],
        )

        st.markdown("#")
        if candidat["Talent_name"] is None:
            name = "le candidat"
        else:
            name = candidat["Talent_name"]

        with st.expander("Apprenez d'avantage sur **l'opportunité**"):
            html_str = f"""
                    <h6 style='text-align: center; color:rgb(44,63,102);font-size: 1.1vw;'>Position </h6> <b> <p style='text-align: center;font-size: 1vw;' ><strong>{name}</strong></p></b>
                            """

            st.markdown(html_str, unsafe_allow_html=True)

            if not candidat["TALENT_NUMERO"]:
                mail = "Pas de mail reconnu"
            else:
                mail = candidat["TALENT_NUMERO"].upper()
            html_mail = f"""
                        <h6 style='text-align: center; color:rgb(44,63,102);font-size: 1.1vw;'>ENTREPISE:</h6><b> <p  style='text-align: center;font-size: 1vw;'><strong>{mail}</strong></p></b>
                                """
            st.markdown(html_mail, unsafe_allow_html=True)

            if not candidat["TALENT_MAIL"]:
                mail = "Pas d'information sur le revenu"
            else:
                mail = candidat["TALENT_MAIL"]
            html_mail = f"""
                        <h6 style='text-align: center; color:rgb(44,63,102);font-size: 1.1vw;'>Revenue</h6><b> <p  style='text-align: center;font-size: 0.9vw;'><strong>{mail}</strong></p></b>
                                """
            st.markdown(html_mail, unsafe_allow_html=True)

            #################  SOFT SKILLS

            base_SKILLS = encoder_et_obtenir_equivalences_dedupliquees(
                candidat["TALENT_SOFT"]
            )
            hard_skills_list_display = []

            if base_SKILLS is None:
                base_SKILLS = []

            if len(base_SKILLS) > 0:
                for hard_skill in base_SKILLS:
                    hard_skills_list_display.append(
                        (hard_skill.capitalize(), "", "#8af")
                    )
                html_diplome = f"""
                                <h6 style='text-align: center; color:rgb(44,63,102);font-size: 1.1vw;'>SOFT SKILLS </h6> <b> <p style='text-align: center;'></p></b>
                            """
                st.markdown(html_diplome, unsafe_allow_html=True)
                annotated_text(hard_skills_list_display)

            else:
                hard_skills_display = "Pas de descriptifs de personalité "
                html_diplome = f"""
                        <h6 style='text-align: center; color:rgb(44,63,102);font-size: 1.1vw;'> SOFT SKILLS </h6> <b> <p style='text-align: center;font-size: 0.8vw;''>{hard_skills_display} </p></b>
                            """

                st.markdown(html_diplome, unsafe_allow_html=True)

            ######################  DIPLOME

            base_SKILLS = candidat["TALENT_DIPLOME"]

            hard_skills_list_display = []
            if base_SKILLS is None:
                base_SKILLS = []
            else:
                based = mise_en_forme_diplome(candidat["TALENT_DIPLOME"])
                based["note"] = pd.to_numeric(based["note"], errors="coerce")
                min_score_row_with_zero_fallback = (
                    based.nsmallest(1, "note")
                    if based[based["note"] > 0].empty
                    else based[based["note"] > 0].nsmallest(1, "note")
                )
                if len(min_score_row_with_zero_fallback["note"]) > 0:
                    # It's safe to access the first element
                    valeur = min_score_row_with_zero_fallback["note"].iloc[0]
                else:
                    # Handle the case where there is no data
                    valeur = 0

                base_SKILLS = [diploma(valeur)]

            if len(base_SKILLS) > 0:
                for hard_skill in base_SKILLS:
                    hard_skills_list_display.append(
                        (hard_skill.capitalize(), "", "#8ef")
                    )
                html_diplome = f"""
                                <h5 style='text-align: center; color:rgb(44,63,102);font-size: 1.1vw;'> FORMATIONS </h5> <b> <p style='text-align: center;'></p></b>
                            """

                st.markdown(html_diplome, unsafe_allow_html=True)
                annotated_text(hard_skills_list_display)

            else:
                hard_skills_display = "Non spécifiée dans le descriptif"
                html_diplome = f"""
                       <h5 style='text-align: center; color:rgb(44,63,102);font-size: 1.1vw;'> FORMATIONS</h5> <b> <p style='text-align: center;'>{hard_skills_display} </p></b>
                            """

                st.markdown(html_diplome, unsafe_allow_html=True)

            ##################### NIVEAU DE LANGUE

            base_langue = candidat["TALENT_LANGUE"]
            langue_list_display = []

            if base_langue is None:
                base_langue = []

            if len(base_langue) > 0:
                base_langue = MEP_LANGUE(traitement_langue(base_langue))
                for langue in base_langue:
                    langue_list_display.append((langue.capitalize(), "", "#fea"))
                html_diplome = f"""
                                <h5 style='text-align: center; color:rgb(44,63,102);font-size: 1.1vw;'>LANGUE</h5> <b> <p style='text-align: center;'></p></b>
                            """
                st.markdown(html_diplome, unsafe_allow_html=True)
                annotated_text(langue_list_display)

            else:
                langue_display = "Non spécifiée dans le descriptif"
                html_diplome = f"""
                        <h5 style='text-align: center; color:rgb(44,63,102);font-size: 1.1vw;'>LANGUE</h5> <b> <p style='text-align: center;'>{langue_display} </p></b>
                            """

                st.markdown(html_diplome, unsafe_allow_html=True)

            #################  STACK TECHNIQUE

            base_SKILLS = candidat["TALENT_STACK"]
            hard_skills_list_display = []

            if base_SKILLS is None:
                base_SKILLS = []

            if len(base_SKILLS) > 0:
                for hard_skill in base_SKILLS:
                    hard_skills_list_display.append((hard_skill.upper(), "", "#faa"))
                html_diplome = f"""
                                <h5 style='text-align: center; color:rgb(44,63,102);font-size: 1.1vw;'> STACK TECHNIQUE </h5> <b> <p style='text-align: center;'></p></b>
                            """
                st.markdown(html_diplome, unsafe_allow_html=True)
                annotated_text(hard_skills_list_display)

            else:
                hard_skills_display = "Pas de competences techniques "
                html_diplome = f"""
                        <h5 style='text-align: center; color:rgb(44,63,102);font-size: 1.1vw;'> STACK TECHNIQUE </h5> <b> <p style='text-align: center;'>{hard_skills_display} </p></b>
                            """

                st.markdown(html_diplome, unsafe_allow_html=True)

            ################# LISTE DE TACHE

            base_SKILLS = candidat["TALENT_TACHE"]
            hard_skills_list_display = []

            if base_SKILLS is None:
                base_SKILLS = []

            if len(base_SKILLS) > 0:
                for hard_skill in base_SKILLS:
                    hard_skills_list_display.append((hard_skill.capitalize(), ""))
                html_diplome = f"""
                                <h5 style='text-align: center; color:rgb(44,63,102);font-size: 1.1vw;'> COMPETENCES </h5> <b> <p style='text-align: center;'></p></b>
                            """
                st.markdown(html_diplome, unsafe_allow_html=True)
                annotated_text(hard_skills_list_display)

            else:
                hard_skills_display = "Pas de competences techniques "
                html_diplome = f"""
                        <h5 style='text-align: center; color:rgb(44,63,102);font-size: 1.1vw;'> COMPETENCES </h5> <b> <p style='text-align: center;'>{hard_skills_display} </p></b>
                            """

                st.markdown(html_diplome, unsafe_allow_html=True)

            ########  PARTI DE CONTACT
            c1, c2, c3 = st.columns((0.2, 0.6, 0.2), gap="large")

            with c2:
                st.markdown("#")

                st.markdown(
                    f"""
                    <a href="{candidat['LINK']}" target="_blank">
                        <button style="
                            background-color: #4CAF50; /* Green */
                            border: none;
                            color: white;
                            padding: 15px 32px;
                            text-align: center;
                            text-decoration: none;
                            display: inline-block;
                            font-size: 16px;
                            margin: 4px 2px;
                            cursor: pointer;
                            border-radius: 12px;
                        ">
                        Je postule
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )

    with colonne_technique:
        st.markdown("#")
        st.markdown("#")
        ##########  Matching Hard skills
        st.markdown(
            "<h2 style='text-align: center; color:rgb(44,63,102);font-size: 1.8vw;'> STACK </h2>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns([0.1, 0.8, 0.1], gap="large")
        with c2:
            afficher_conseil(candidat["NOTE_STACK"])

        st.plotly_chart(
            affichage_scatter(candidat["STACKS"]),
            use_container_width=True,
        )

        st.markdown(
            "<h2 style='text-align: center; color:rgb(44,63,102);font-size: 1.8vw;'>MISSIONS</h2>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns([0.1, 0.8, 0.1], gap="large")
        with c2:
            afficher_conseil(candidat["NOTE_TACHE"])

        affiche_taches(candidat)

    with colonne_middle:
        st.markdown("#")
        st.markdown("#")
        st.markdown(
            "<h2 style='text-align: center; color:rgb(44,63,102);font-size: 1.8vw;'>FORMATION</h2>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([0.1, 0.8, 0.1], gap="large")
        with c2:
            afficher_conseil_diplome(candidat["NOTE_DIPLOME"])
            st.markdown("#")

        prospect_dipl = [
            ["Niveau requis"],
            [candidat["MAX_DIPLOMA_CANDIDAT"]],
        ]
        job_dipl = [["Vous"], [candidat["REQUIERED_DIPLOMA"]]]

        st.plotly_chart(
            Graph_Diplome(job_dipl, prospect_dipl),
            config={"displayModeBar": False},
            use_container_width=True,
        )

        st.markdown(
            "<h2 style='text-align: center; color:rgb(44,63,102);font-size: 1.8vw;'>SOFT SKILLS</h2>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([0.1, 0.8, 0.1], gap="large")
        with c2:
            afficher_conseil(candidat["NOTE_SSKILL"])

        affiche_competences(candidat)

    with colonne_right:
        st.markdown("#")
        st.markdown("#")
        st.markdown(
            "<h2 style='text-align: center; color:rgb(44,63,102);font-size: 1.8vw;'> LANGUE </h2>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([0.1, 0.8, 0.1], gap="large")
        with c2:
            afficher_conseil_langue(candidat["NOTE LANGUE"])

        st.plotly_chart(
            Graph_Langue(candidat["DETAIL_LANGUE"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown(
            "<h2 style='text-align: center; color:rgb(44,63,102);font-size: 1.8vw;'>EXPERIENCE<br></h2>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns([0.1, 0.8, 0.1], gap="large")
        with c2:
            afficher_conseil_experience(candidat["NOTE_EXP"])
        st.plotly_chart(
            Graph_Comparaison_Niveau_avec_Liste(candidat["DETAIL_EXP"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )
