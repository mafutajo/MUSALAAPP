import streamlit as st
import pandas as pd
from PIL import Image
import hashlib
import io
import plotly.graph_objects as go
from annotated_text import annotated_text
import streamlit as st
import pandas as pd
import time
import json
import math
from pdfminer.high_level import extract_text
import re
import streamlit as st
import pandas as pd
from PIL import Image
import hashlib
import io
from itertools import count
import time
import matplotlib.pyplot as plt
from pdfminer.high_level import extract_text
from annotated_text import annotated_text
from collections import Counter
from datetime import datetime, timedelta
from offres_emploi.utils import dt_to_str_iso
from datetime import datetime, timedelta
from offres_emploi import Api


# from visualisation_match import match_visualisation


def ajouter_region_a_offres_depuis_fichier(offres, chemin_cities_json):
    """
    Ajoute une clé 'region' à chaque offre dans la liste des offres en fonction de son code postal ou du libellé de lieu de travail,
    en utilisant les données des villes stockées dans un fichier JSON. Si aucun code postal ou libellé n'est disponible,
    la région est définie comme 'Non spécifiée'.

    :param offres: Liste des offres d'emploi (dictionnaires).
    :param chemin_cities_json: Chemin vers le fichier JSON contenant les données des villes.
    :return: La liste des offres avec la clé 'region' ajoutée.
    """
    with open(chemin_cities_json, "r", encoding="utf-8") as file:
        cities_data = json.load(file)

    for offre in offres:
        # Initialiser la région comme 'Non spécifiée' par défaut
        region_name = "Non spécifiée"

        # Extraire le code postal si disponible, sinon utiliser le libellé du lieu de travail
        code_postal_offre = offre["lieuTravail"].get("codePostal")
        libelle_lieu_travail = offre["lieuTravail"].get("libelle")

        if code_postal_offre:
            for city in cities_data["cities"]:
                if city["zip_code"] == code_postal_offre:
                    region_name = city["region_name"]
                    break
        elif libelle_lieu_travail:
            # Recherche basée sur le libellé du lieu de travail si le code postal n'est pas disponible
            for city in cities_data["cities"]:
                if city["label"].lower() == libelle_lieu_travail.lower():
                    region_name = city["region_name"]
                    break

        offre["region"] = region_name

    return offres


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


from lst_fction import (
    parsing_joboffer,
    matching_offer_with_candidat,
    diploma,
    mise_en_forme_diplome,
    match_visualisation,
    encoder_et_obtenir_equivalences_dedupliquees,
    MEP_LANGUE,
    traitement_langue,
    afficher_conseil,
)


# Autres fonctions spécifiques à votre application...
def generate_gauge(note, seuil_incomplet=100, piliers_incomplets=None):
    gradient_color = "rgba(56,181,165,255)"  # Couleur par défaut (vert)

    if note < 50:
        gradient_color = "#FA8072"  # Rouge pour les notes inférieures à 50
        text_color = "#FA8072"
    elif 50 <= note < 70:
        gradient_color = "#ff7f00"  # Orange pour les notes entre 50 et 70
        text_color = "#ff7f00"
    else:
        text_color = "rgba(56,181,165,255)"

    html_str = f"""
        <style>
            .gauge-container {{
                width: 120px;
                height: 120px;
                position: relative;
                margin: 0 auto;
                border-color:rgb(75,98,133);
            }}
            .gauge {{
                width: 100%;
                height: 100%;
                background: conic-gradient({gradient_color} {int(note)}%, #fff {int(note)}% 100%);
                border-radius: 50%;
                position: absolute;
                border-color:rgb(75,98,133);
                border: 4px solid #000;
            }}
            .gauge::before {{
                content: '';
                width: 80%;
                height: 80%;
                background-color: #fff;
                border-radius: 50%;
                position: absolute;
                border-color:rgb(75,98,133);
                top: 10%;
                left: 10%;
                border: 3px solid #000;
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
            <p class="gauge-text ">{str(int(note))}</p>
        </div>
    """
    return html_str


def clarifier_nom(nom_micro):
    noms_clairs = {
        "dev_log": "Développement de logiciels",
        "data": "Ingénierie des données",
        "dev_web": "Développement web",
        "devop": "OpsDev (Opérations de Développement)",
        "ing_reseau": "Ingénierie réseau",
        "ing_systeme": "Ingénierie système",
        "dev_mobile": "Développement mobile",
        "cloud": "Cloud computing",
        "cyber_sec": "Sécurité informatique",
        "chefprojet": "Chef de projet",
    }

    return noms_clairs.get(nom_micro, "Nom non trouvé")


def inverser_clarification(nom_clair):
    noms_clairs_inverse = {v: k for k, v in noms_clairs.items()}
    return noms_clairs_inverse.get(nom_clair, " ")


noms_clairs = {
    "dev_log": "Développement de logiciels",
    "data": "Ingénierie des données",
    "dev_web": "Développement web",
    "devop": "OpsDev (Opérations de Développement)",
    "ing_reseau": "Ingénierie réseau",
    "ing_systeme": "Ingénierie système",
    "dev_mobile": "Développement mobile",
    "cloud": "Cloud computing",
    "cyber_sec": "Sécurité informatique",
    "chefprojet": "Chef de projet",
}


def renommer_cles(dictionnaire):
    nouveau_dictionnaire = {
        "tache": dictionnaire.get("task", []),
        "diplome": dictionnaire.get("diploma", ""),
        "job_type": dictionnaire.get("job_type", ""),
        "langue": [dictionnaire.get("language", "")],
        "hard_skills": dictionnaire.get("hard_skill", []),
        "soft_skills": dictionnaire.get("soft_skill", []),
        "tache_median": dictionnaire.get("task_median", 0),
        "hard_skills_median": dictionnaire.get("hard_skill_median", 0),
        "soft_skills_median": dictionnaire.get("soft_skill_median", 0),
    }
    return nouveau_dictionnaire


def renommer_cles2(dictionnaire):
    nouveau_dictionnaire = {
        "tache": dictionnaire.get("task", []),
        "diplome": dictionnaire.get("diplome", []),
        "job_type": dictionnaire.get("job_type", ""),
        "langue": dictionnaire.get("language", []),
        "hard_skills": dictionnaire.get("hard_skill", []),
        "soft_skills": dictionnaire.get("soft_skill", []),
        "tache_median": dictionnaire.get("task_median", 0),
        "hard_skills_median": dictionnaire.get("hard_skill_median", 0),
        "soft_skills_median": dictionnaire.get("soft_skill_median", 0),
        "nom_prenom": dictionnaire.get("nom_prenom", ""),
        "mail": dictionnaire.get("email", ""),
        "numero": dictionnaire.get("numero", ""),
    }
    return nouveau_dictionnaire


@st.cache_data(ttl=3600, show_spinner=False)
def cache_resume(raw_text: str) -> dict:
    return parsing_joboffer(raw_text)


@st.cache_data  # Décorateur pour mettre en cache les résultats de la fonction
def load_json_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return data


def normalize_skills(skill):
    """
    Normalise les compétences pour unifier les termes similaires.

    Args:
    skill (str): La compétence à normaliser.

    Returns:
    str: La compétence normalisée.
    """
    skill = skill.lower()
    if skill == "power" or "bi" == skill or skill == "powerbi" or skill == "power.bi":
        return "power bi"

    return skill


import statistics  # Import pour calculer la moyenne et la médiane


def calculate_top_hard_skills(filtered_data, top_n=10):
    """
    Calcule et renvoie le top N des hard_skills les plus fréquents avec leur pourcentage par rapport
    au nombre total d'offres d'emploi analysées ayant des hard_skills non vides. Renvoie également la médiane
    et la moyenne du nombre de hard skills par offre.

    Args:
    filtered_data (list of dict): Données filtrées, chaque dictionnaire représentant une offre d'emploi.
    top_n (int): Nombre de top compétences à retourner.

    Returns:
    dict: Dictionnaire contenant la liste des tuples (skill, count, percentage), la médiane et la moyenne
    du nombre de hard skills par offre.
    """
    hard_skill_counts = {}
    num_skills_per_offer = []  # Liste pour stocker le nombre de hard skills par offre

    for item in filtered_data:
        if "hard_skills" in item and item["hard_skills"]:
            num_skills_per_offer.append(
                len(item["hard_skills"])
            )  # Ajouter le nombre de skills pour cette offre
            normalized_skills = set(
                normalize_skills(skill) for skill in item["hard_skills"]
            )
            for skill in normalized_skills:
                if skill in hard_skill_counts and skill != "devops":
                    hard_skill_counts[skill] += 1
                else:
                    hard_skill_counts[skill] = 1

    offers_with_skills_count = len(
        num_skills_per_offer
    )  # Nombre d'offres ayant au moins un hard_skill

    # S'il n'y a aucune offre avec hard_skills, éviter la division par zéro
    if offers_with_skills_count == 0:
        return {"top_skills": [], "median_skills": 0, "mean_skills": 0}

    # Trier les compétences par occurrence et renvoyer les top N
    sorted_skills = sorted(hard_skill_counts.items(), key=lambda x: x[1], reverse=True)[
        :top_n
    ]
    top_skills_with_percentage = [
        (skill, count, (count / offers_with_skills_count) * 100)
        for skill, count in sorted_skills
    ]

    # Calculer la médiane et la moyenne du nombre de hard skills par offre
    median_skills = statistics.median(num_skills_per_offer)
    mean_skills = statistics.mean(num_skills_per_offer)

    return top_skills_with_percentage, median_skills, mean_skills


def calculate_top_soft_skills(filtered_data, top_n=10):
    """
    Calcule et renvoie le top N des soft_skills les plus fréquents avec leur pourcentage par rapport
    au nombre total d'offres d'emploi analysées ayant des soft_skills non vides. Renvoie également la médiane
    et la moyenne du nombre de soft skills par offre.

    Args:
    filtered_data (list of dict): Données filtrées, chaque dictionnaire représentant une offre d'emploi.
    top_n (int): Nombre de top compétences à retourner.

    Returns:
    dict: Dictionnaire contenant la liste des tuples (skill, count, percentage), la médiane et la moyenne
    du nombre de soft skills par offre.
    """
    soft_skill_counts = {}
    num_skills_per_offer = []  # Liste pour stocker le nombre de soft skills par offre

    for item in filtered_data:
        if "soft_skills" in item and item["soft_skills"]:
            corrected_skills = [
                correct_skill_name(skill) for skill in item["soft_skills"]
            ]
            num_skills_per_offer.append(
                len(corrected_skills)
            )  # Ajouter le nombre de skills pour cette offre
            for skill in corrected_skills:
                if skill in soft_skill_counts:
                    soft_skill_counts[skill] += 1
                else:
                    soft_skill_counts[skill] = 1

    offers_with_skills_count = len(
        num_skills_per_offer
    )  # Nombre d'offres ayant au moins un soft_skill

    if offers_with_skills_count == 0:
        return {"top_skills": [], "median_skills": 0, "mean_skills": 0}

    sorted_skills = sorted(soft_skill_counts.items(), key=lambda x: x[1], reverse=True)[
        :top_n
    ]
    top_skills_with_percentage = [
        (skill, count, (count / offers_with_skills_count) * 100)
        for skill, count in sorted_skills
    ]

    median_skills = statistics.median(num_skills_per_offer)
    mean_skills = statistics.mean(num_skills_per_offer)

    return top_skills_with_percentage, median_skills, mean_skills


def correct_skill_name(skill):
    corrections = {"AUTOMIE": "AUTONOMIE"}
    return corrections.get(
        skill, skill
    )  # Retourne la correction si disponible, sinon retourne le skill original


import plotly.express as px


def display_top_skills_with_plotly(top_hard_skills, selected_name):
    if top_hard_skills:
        # Préparation des données pour Plotly
        skills, counts, percentages = zip(
            *top_hard_skills
        )  # Dépaquetage en trois listes
        data = {"Skills": skills, "Percentage": percentages, "Counts": counts}
        df = pd.DataFrame(data).sort_values(by="Percentage", ascending=True)

        # Créer un graphique à barres avec Plotly Express
        fig = px.bar(
            df,
            y="Skills",
            x="Percentage",
            text="Percentage",
            orientation="h",
            color="Skills",
            color_discrete_sequence=px.colors.sequential.Plasma_r,
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(
            uniformtext_minsize=8,
            uniformtext_mode="hide",
            xaxis_title="Pourcentage de demande (%)",
            yaxis_title="Compétences",
            showlegend=False,
            margin=dict(
                t=10, l=10, r=10, b=10
            ),  # Réglage des marges (top, left, right, bottom)
        )
        fig.update_yaxes(autorange="reversed")  # Inverser l'axe y

        # Afficher le graphique dans Streamlit
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Aucune donnée disponible pour afficher le graphique.")


def display_sskills_with_plotly(top_soft_skills, selected_name):
    if top_soft_skills:
        # Préparation des données pour Plotly
        skills, counts, percentages = zip(*top_soft_skills)
        soft_data = {"Skills": skills, "Percentage": percentages}
        df_soft = pd.DataFrame(soft_data)

        # Calcul du maximum ajusté pour l'échelle radiale
        max_percentage = max(percentages) + 10
        max_percentage = min(
            max_percentage, 100
        )  # Assure que le maximum ne dépasse pas 100%

        # Création du graphique Radar pour les soft skills
        fig_soft = px.line_polar(
            df_soft,
            r="Percentage",
            theta="Skills",
            line_close=True,
            color_discrete_sequence=px.colors.sequential.Plasma,
        )
        fig_soft.update_traces(fill="toself")
        fig_soft.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=False,
                    range=[0, max_percentage],  # Utilisation de max_percentage calculé
                )
            ),
            showlegend=False,
        )

        # Affichage du graphique avec Streamlit
        st.plotly_chart(fig_soft, use_container_width=True)
    else:
        st.write("Aucune donnée disponible pour les soft skills.")


def analyze_and_display_diploma_distribution_pie(jobs):
    # Compter la fréquence de chaque diplôme
    diploma_count = Counter()
    excluded_diplomas = ["Pas de diplome ou non reconnu"]
    total_excluded = 0  # Pour compter les diplômes exclus

    for job in jobs:
        for diploma in job["diplome"]:
            if diploma in excluded_diplomas:
                total_excluded += 1
            else:
                diploma_count[diploma] += 1

    # Calculer la part en pourcentage de chaque type de diplôme non exclu
    total = sum(diploma_count.values())
    percentage = {
        diploma: (count / total) * 100 for diploma, count in diploma_count.items()
    }

    # Préparer les données pour le graphique
    diplomas = list(percentage.keys())
    percentages = list(percentage.values())
    data = {"Diplomas": diplomas, "Percentage": percentages}
    df = pd.DataFrame(data)

    # Créer un graphique en camembert avec Plotly Express
    fig = px.pie(
        df,
        names="Diplomas",
        values="Percentage",
        hole=0.4,  # Vous pouvez ajuster ceci pour créer un effet de donut
        color_discrete_sequence=px.colors.sequential.Plasma_r,
    )

    # Mise à jour de l'apparence du graphique
    fig.update_traces(textposition="inside", textinfo="label+percent")
    fig.update_layout(
        showlegend=False,
        title_x=0.5,  # Centrer le titre
        margin=dict(l=20, r=20, t=40, b=20),  # Ajustement des marges
    )

    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Calculer la part des diplômes exclus
    total_general = total + total_excluded
    if total_general > 0:
        excluded_percentage = (total_excluded / total_general) * 100


def analyze_and_display_diploma_distribution_histogram(jobs):
    # Compter la fréquence de chaque diplôme
    diploma_count = Counter()
    excluded_diplomas = [
        "Pas de diplome ou non reconnu",
        "Diplome de niveau 4 type Baccalauréat",
    ]
    total_excluded = 0

    for job in jobs:
        for diploma in job["diplome"]:
            if diploma in excluded_diplomas:
                total_excluded += 1
            else:
                diploma_count[diploma] += 1

    # Calculer la part en pourcentage de chaque type de diplôme non exclu
    total = sum(diploma_count.values())
    percentage = {
        diploma: (count / total) * 100 for diploma, count in diploma_count.items()
    }

    # Préparer les données pour le graphique
    diplomas = list(percentage.keys())
    percentages = list(percentage.values())
    data = {"Diplomas": diplomas, "Percentage": percentages}
    df = pd.DataFrame(data).sort_values("Percentage", ascending=False)
    df.Percentage = df.Percentage.astype(int)
    # Créer un histogramme avec Plotly Express
    fig = px.bar(
        df,
        y="Diplomas",
        x="Percentage",
        text="Percentage",  # Ajout du texte pour afficher les pourcentages
        labels={"Diplomas": " ", "Percentage": "Pourcentage (%)"},
        color="Diplomas",
        orientation="h",
        color_discrete_sequence=px.colors.sequential.Teal,
    )

    # Déterminer le positionnement du texte en fonction des valeurs
    df["text_position"] = df["Percentage"].apply(
        lambda x: "inside" if x > 10 else "outside"
    )

    # Mise à jour des traces pour afficher le texte à l'extérieur ou à l'intérieur
    for d, position in zip(fig.data, df["text_position"]):
        d.update(textposition=position)

    # Formatage du texte pour inclure le signe de pourcentage
    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside")

    # Mise à jour de l'apparence du graphique
    fig.update_layout(
        xaxis_title="Pourcentage (%)",
        showlegend=False,
        xaxis_showgrid=False,
        xaxis={"visible": False, "showticklabels": False},
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_yaxes(categoryorder="total ascending")

    # Calculer la part des diplômes exclus
    total_general = total + total_excluded
    if total_general > 0:
        excluded_percentage = (total_excluded / total_general) * 100

    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Déterminer le diplôme dominant et créer une phrase descriptive
    dominant_diploma = df.iloc[0]["Diplomas"]
    dominant_percentage = df.iloc[0]["Percentage"]
    if dominant_percentage > 50:
        result = f"La majorité des offres ({dominant_percentage:.0f}%) demandent un  <bold>{dominant_diploma}</bold> au minimum."
    else:
        result = f"<bold>{dominant_diploma}</bold> ,en niveau minimum, est le plus fréquemment demandé , présent dans {dominant_percentage:.0f}% des offres."

    # Calculer la part des diplômes exclus
    total_general = total + total_excluded
    if total_general > 0:
        excluded_percentage = (total_excluded / total_general) * 100

    return result


def analyze_and_display_time_distribution_histogram(jobs):
    # Compter la fréquence de chaque année d'expérience, en excluant "Non spécifié"
    time_count = Counter()

    for job in jobs:
        time_in_field = job["time_in_field"]
        # Skip the count if time_in_field is "Non spécifié"
        if time_in_field != "Non spécifié":
            time_count[time_in_field] += 1

    # Calculer la part en pourcentage de chaque catégorie d'années d'expérience
    total = sum(time_count.values())
    percentage = {time: (count / total) * 100 for time, count in time_count.items()}

    # Préparer les données pour le graphique
    times = list(percentage.keys())
    percentages = list(percentage.values())
    data = {"Time in Field": times, "Percentage": percentages}
    df = pd.DataFrame(data)
    df = df.sort_values(by="Percentage", ascending=False)

    # Créer un diagramme circulaire avec Plotly Express
    fig = px.pie(
        df,
        names="Time in Field",
        values="Percentage",
        color="Time in Field",  # Cette ligne est optionnelle, pour colorer différemment chaque tranche
        color_discrete_sequence=px.colors.sequential.Plasma,
    )

    # Formatage du texte pour inclure le signe de pourcentage
    fig.update_traces(textposition="inside", textinfo="percent+label")

    # Mise à jour de l'apparence du graphique
    fig.update_layout(
        legend_title="Années d'Expérience",
        margin=dict(l=10, r=10, t=10, b=10),
    )

    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)
    # Analyser la dominance des niveaux d'expérience
    # Analyser la dominance des niveaux d'expérience
    dominant_exp = df.iloc[0]["Time in Field"]
    dominant_percentage = df.iloc[0]["Percentage"]

    # Interpréter les implications de la dominance
    if "Senior" in dominant_exp and dominant_percentage > 43:
        advice = (
            "Ceci n'est pas idéal pour ceux cherchant à entrer dans le secteur, "
            "car la majorité des postes demandent une expérience approfondie."
        )
    elif "Junior" in dominant_exp and dominant_percentage > 43:
        advice = (
            "Cela représente une bonne opportunité pour ceux cherchant à entrer dans le secteur, "
            "car la majorité des postes sont ouverts aux débutants."
        )
    else:
        advice = "Les offres sont assez diversifiées en termes de niveaux d'expérience requis."

    return f"Le niveau d'expérience le plus fréquemment demandé est '{dominant_exp}', représentant {dominant_percentage:.0f}% des offres. {advice}"


def retribution_HS(top_hard_skills, selected_name, median, moyenne):

    col1, col2, col3 = st.columns([0.2, 0.5, 0.3], gap="large")

    with col1:
        title_html = """
            <div style="background-color: #302838; padding: 10px; border-radius: 10px;">
                <h2 style="color: white; text-align: center; font-weight: bold;">HARD SKILLS</h2>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown("#")
        with st.expander("Qu'est ce qu'un **hard skill**?"):
            title_html_presentation = """
                <p style="color:  #34495E; text-align: center;font-size:0.75vw;">Il s'agit de l'ensemble des compétences techniques spécifiques à un domaine, acquises par la formation et l'expérience professionnelle. Ces compétences sont quantifiables et mesurables.</p>
                <p style="font-style: italic; font-weight: bold;text-align: center;font-size:0.75vw;color: color: #34495E;">Exemple: Maîtrise du logiciel <span style='color:green;'><strong>Excel</strong></span>, qui permet l'analyse de données et la création de tableaux complexes.</p>
                """
            st.markdown(title_html_presentation, unsafe_allow_html=True)

    with col2:
        if top_hard_skills:
            display_top_skills_with_plotly(top_hard_skills, selected_name)
        else:
            st.warning(
                "Aucune compétence comportementale n'est demandée pour ce poste."
            )
            # Affichage d'une barre de progression représentant la note globale des soft skills
            # La note est supposée être sur 100, ajustez si nécessaire

    with col3:
        title_html = """
            <div style=" padding: 10px; border-radius: 10px;">
                <h4 style="color:#34495E; text-align: center; font-weight: bold;font-size:1.4vw;">Avis Nest</h4>
            </div>
            """

        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="padding: 10px;">
                <p style="color:#34495E; text-align: center; font-weight: lighter;font-size:1.0vw;">Pour le métier de <strong>{selected_name}</strong>, la maîtrise* d'au moins <strong>{int(min(median,moyenne))}</strong> outils sont demandés.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(
            " *la maitrise ne signifie pas etre **expert** mais au moins connaitre ces technologies ou équivalent."
        )
        # if overall_note < 0:
        #     st.markdown(
        #         "Aucune compétence comportementale n'est demandée pour ce poste."
        #     )


def retribution_SS(top_hard_skills, selected_name, median_ss, moyenne_ss):

    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

    with col1:
        title_html = """
            <div style="background-color: #302838; padding: 10px; border-radius: 10px;">
                <h2 style="color: white; text-align: center; font-weight: bold;">SOFT SKILLS</h2>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown("#")
        with st.expander("Qu'est ce qu'un **soft skill**?"):
            title_html_presentation = """
                <p style="color:  #34495E; text-align: center;font-size:0.75vw;">Les Soft Skills sont des compétences interpersonnelles qui influencent la manière dont une personne interagit avec les autres. Elles comprennent la communication, l'empathie, et la capacité à travailler en équipe.</p>
                <p style="font-style: italic; font-weight: bold;text-align: center;font-size:0.75vw;color:  #34495E;">Exemple: La <span style='color:green;'><strong>communication efficace</strong></span> est essentielle pour la coordination d'équipe et le leadership.</p>
                """
            st.markdown(title_html_presentation, unsafe_allow_html=True)

    with col2:
        if top_hard_skills:
            display_sskills_with_plotly(top_hard_skills, selected_name)
        else:
            st.warning(
                "Aucune compétence comportementale n'est demandée pour ce poste."
            )
            # Affichage d'une barre de progression représentant la note globale des soft skills
            # La note est supposée être sur 100, ajustez si nécessaire

    with col3:
        title_html = """
            <div style=" padding: 10px; border-radius: 10px;">
                <h4 style="color:  #34495E; text-align: center; font-weight: bold;font-size:1.4vw;">Avis Nest</h4>
            </div>
            """

        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown("#")
        st.markdown(
            f"""
            <div style="padding: 10px;">
                <p style="color:#34495E; text-align: center; font-weight: lighter;font-size:1.0vw;"><strong>{int(min(median_ss,moyenne_ss))} qualités humaines</strong> semblent nécessaires parmis ceux specifiés dans les offres.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # if overall_note < 0:
        #     st.markdown(
        #         "Aucune compétence comportementale n'est demandée pour ce poste."
        #     )


def retribution_DIPL(filtered_data):

    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

    with col1:
        title_html = """
            <div style="background-color: #302838; padding: 10px; border-radius: 10px;">
                <h2 style="color: white; text-align: center; font-weight: bold;">DIPLOME</h2>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown("#")
        with st.expander("Qu'est ce qui est decris par **diplome**?"):
            description_diplome = """
            <p style="color:  #34495E; text-align: center; font-size:0.75vw;">
                Le diplôme indique le niveau de formation académique attendu pour le poste. Cela inclut les qualifications spécifiques qui sont souvent nécessaires pour pratiquer dans certains domaines professionnels.
            </p>
            """
            st.markdown(description_diplome, unsafe_allow_html=True)

    with col2:
        if filtered_data:
            sentence = analyze_and_display_diploma_distribution_histogram(filtered_data)
        else:
            st.warning(
                "Aucune compétence comportementale n'est demandée pour ce poste."
            )
            # Affichage d'une barre de progression représentant la note globale des soft skills
            # La note est supposée être sur 100, ajustez si nécessaire

    with col3:
        title_html = """
            <div style=" padding: 10px; border-radius: 10px;">
                <h4 style="color:  #34495E; text-align: center; font-weight: bold;font-size:1.4vw;">Avis Nest</h4>
            </div>
            """

        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown("#")
        st.markdown(
            f"""
            <div style="padding: 10px;">
                <p style="color:#34495E; text-align: center; font-weight: lighter;font-size:1.0vw;">{sentence}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # if overall_note < 0:
        #     st.markdown(
        #         "Aucune compétence comportementale n'est demandée pour ce poste."
        #     )


def retribution_LANGUE(filtered_data):

    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

    with col1:
        title_html = """
            <div style="background-color: #302838; padding: 10px; border-radius: 10px;">
                <h2 style="color: white; text-align: center; font-weight: bold;">LANGUE</h2>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)

        st.markdown("#")
        with st.expander("Qu'est ce qui est decris par niveau de **langue**?"):

            # Ajouter une description pour les compétences linguistiques
            description_langue = """
                <p style="color:  #34495E; text-align: center; font-size:0.75vw;">
                    La maîtrise de langues supplémentaires peut être cruciale pour les postes nécessitant la communication internationale ou le travail dans des environnements multiculturels.
                </p>
                """
            st.markdown(description_langue, unsafe_allow_html=True)

    with col2:
        if filtered_data:
            language_data = extract_and_count_languages(filtered_data)
            df = prepare_data_for_plotting(language_data)
            plot_language_levels(df)
        else:
            st.warning(
                "Aucune compétence comportementale n'est demandée pour ce poste."
            )
            # Affichage d'une barre de progression représentant la note globale des soft skills
            # La note est supposée être sur 100, ajustez si nécessaire

    with col3:
        title_html = """
            <div style=" padding: 10px; border-radius: 10px;">
                <h4 style="color:  #34495E; text-align: center; font-weight: bold;font-size:1.4vw;">Avis Nest</h4>
            </div>
            """

        st.markdown(title_html, unsafe_allow_html=True)
        # if overall_note < 0:
        #     st.markdown(
        #         "Aucune compétence comportementale n'est demandée pour ce poste."
        #     )


def retribution_TACHE(filtered_data):

    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

    with col1:
        title_html = """
            <div style="background-color: #302838; padding: 10px; border-radius: 10px;">
                <h2 style="color: white; text-align: center; font-weight: bold;">TACHES</h2>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown("#")
        with st.expander("Details sur les **taches**?"):

            # Ajouter une description pour les tâches typiques associées au poste
            description_taches = """
                <p style="color:  #34495E;text-align: center; font-size:0.75vw;">
                    Les tâches comprennent les activités quotidiennes et les responsabilités spécifiques au poste. Elles varient grandement en fonction du domaine et de la spécialité de l'emploi.
                </p>
                """
            st.markdown(description_taches, unsafe_allow_html=True)

    with col2:
        if filtered_data:
            tasks = load_and_combine_tasks(filtered_data)
            filtered_tasks = filter_tasks_with_min_tokens(tasks)
            display_tasks(filtered_tasks)
        else:
            st.warning(
                "Aucune compétence comportementale n'est demandée pour ce poste."
            )
            # Affichage d'une barre de progression représentant la note globale des soft skills
            # La note est supposée être sur 100, ajustez si nécessaire

    with col3:
        title_html = """
            <div style=" padding: 10px; border-radius: 10px;">
                <h4 style="color:  #34495E; text-align: center; font-weight: bold;font-size:1.4vw;">Avis Nest</h4>
            </div>
            """

        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown("#")
        st.markdown(
            f"""
            <div style="padding: 10px;">
                <p style="color:#34495E; text-align: center; font-weight: lighter;font-size:1.0vw;">Gardez en tête que les taches dependent fortement du contexte dans lequel s'imbrique ces offres.<br> Explorez les pour decouvrir leur grandes diversitées.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # if overall_note < 0:
        #     st.markdown(
        #         "Aucune compétence comportementale n'est demandée pour ce poste."
        #     )


def retribution_EXP(filtered_data):

    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

    with col1:
        title_html = """
            <div style="background-color: #302838; padding: 10px; border-radius: 10px;">
                <h2 style="color: white; text-align: center; font-weight: bold;">EXPÉRIENCE</h2>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)

        st.markdown("#")
        with st.expander("Qu'est ce qui est decris par niveau **d'éxperience**?"):

            # Ajouter une description pour les compétences linguistiques
            description_langue = """
                <p style="color:  #34495E; text-align: center; font-size:0.75vw;">
                   Il s'agit ici du niveau d'experience specifié sur les offres
                   <li style="color:  #34495E; text-align: center; font-size:0.75vw;"><span style="font-weight:bolder;">Senior:</span> Plus de 7 ans d'éxperiences</li>
                   <li style="color:  #34495E; text-align: center; font-size:0.75vw;"><span style="font-weight:bolder;">Intermediare:</span> Entre 3 et 7 ans d'éxperiences</li>
                   <li style="color:  #34495E; text-align: center; font-size:0.75vw;"><span style="font-weight:bolder;">Junior:</span> Moins de 3 ans d'éxperiences</li>
                </p>
                """
            st.markdown(description_langue, unsafe_allow_html=True)

    with col2:
        if filtered_data:
            sentence = analyze_and_display_time_distribution_histogram(filtered_data)
        else:
            st.warning(
                "Aucune compétence comportementale n'est demandée pour ce poste."
            )
            # Affichage d'une barre de progression représentant la note globale des soft skills
            # La note est supposée être sur 100, ajustez si nécessaire

    with col3:
        title_html = """
            <div style=" padding: 10px; border-radius: 10px;">
                <h4 style="color:  #34495E; text-align: center; font-weight: bold;font-size:1.4vw;">Avis Nest</h4>
            </div>
            """

        st.markdown(title_html, unsafe_allow_html=True)
        st.markdown("#")
        st.markdown(
            f"""
            <div style="padding: 10px;">
                <p style="color:#34495E; text-align: center; font-weight: lighter;font-size:1.0vw;">{sentence}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # if overall_note < 0:
        #     st.markdown(
        #         "Aucune compétence comportementale n'est demandée pour ce poste."
        #     )


def entete(nom_metier, description_metier):
    # HTML and CSS
    html_content = f"""
    <style>
        .header-container {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            width: 80%;  /* Largeur fixe relative à la largeur de la page ou de l'élément parent */
            min-height: 100px;  /* Hauteur minimum pour s'assurer que le conteneur ne soit pas trop petit */
            background: #302838;  /* Couleur de fond */
            color: white;  /* Couleur du texte */
            margin: 20px auto;  /* Centrage et espace autour du conteneur */
            padding: 20px;  /* Espacement interne */
            border-radius: 10px;  /* Bords arrondis */
            font-family: 'Arial', sans-serif;  /* Police de caractères */
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);  /* Ombre portée pour un effet de profondeur */
        }}

        .job-title {{
            font-size: 1.5vw;  /* Taille du texte pour le titre */
            font-weight: bold;  /* Gras pour le titre */
            margin-bottom: 10px;  /* Espace entre le titre et la description */
        }}

        .job-description {{
            font-size: 0.9vw;  /* Taille du texte pour la description */
            text-align: center;  /* Justification du texte pour une meilleure lisibilité */
            line-height: 1.5;  /* Espacement des lignes */
        }}
    </style>

    <div class="header-container">
        <div class="job-title">{nom_metier}</div>
        <div class="job-description">{description_metier}</div>
    </div>
    """

    # Utilisation de cette ligne pour afficher le HTML dans Streamlit
    st.markdown(html_content, unsafe_allow_html=True)


def obtenir_description_metier(nom_metier):
    descriptions = {
        "Product owner": "Le Product Owner est responsable de définir les besoins des utilisateurs et de s'assurer que l'équipe de développement livre le meilleur produit possible.",
        "Data scientist": "Le Data Scientist utilise des techniques statistiques avancées et des algorithmes d'intelligence artificielle pour créer des modèles complexes qui analysent de grandes quantités de données. Ces modèles aident à la prise de décision stratégique et à prédire des tendances futures.",
        "Data engineer": "Le Data Engineer conçoit et construit des infrastructures de données qui supportent l'analyse et l'extraction de données à grande échelle, en veillant à leur accessibilité, leur fiabilité et leur rapidité.",
        "Data analyst": "Le Data Analyst a un rôle central dans la prise de décision en analysant les données et en générant des insights pertinents pour l'entreprise. Il aide à comprendre les tendances du marché et le comportement des consommateurs.",
        "Développeur fullstack": "Le Développeur fullstack est compétent à la fois en back-end et en front-end. Il conçoit et développe des applications web complètes, s'occupant à la fois de la logique côté serveur et de l'interaction côté client.",
        "Développeur front end": "Le Développeur front end se spécialise dans la création de l'interface utilisateur et de l'expérience utilisateur des applications web, utilisant des technologies comme HTML, CSS, et JavaScript.",
        "Développeur logiciel": "Le Développeur logiciel crée des programmes informatiques qui aident les utilisateurs à effectuer des tâches spécifiques sur l'ordinateur ou d'autres dispositifs.",
        "Devops": "Le spécialiste Devops joue un rôle clé dans le développement continu et l'administration des systèmes, en travaillant à l'intersection du développement de logiciels et des opérations informatiques pour améliorer la collaboration et l'efficacité.",
        "Administrateur système": "L'Administrateur système est responsable de la gestion, de la configuration et du maintien en condition opérationnelle des systèmes informatiques d'une entreprise, assurant la sécurité et la stabilité de l'infrastructure.",
        "Ux designer": "L'UX Designer conçoit et améliore l'expérience utilisateur des produits digitaux. Il analyse les comportements des utilisateurs et crée des interfaces qui sont à la fois intuitives et esthétiques.",
        "Data ingénieur": "Le Data Ingénieur développe, construit, teste et maintient des architectures telles que des bases de données et des systèmes de traitement des données à grande échelle.",
        "Specialist seo ⭐": "Le Spécialiste SEO optimise les sites web pour améliorer leur visibilité sur les moteurs de recherche et augmenter le trafic organique. Il analyse et audite les sites, réalise des recherches de mots-clés, et optimise les balises et le contenu. Il développe des stratégies de contenu en collaboration avec les rédacteurs et met en œuvre des campagnes de link building pour renforcer l'autorité du site.",
    }

    return descriptions.get(nom_metier, "Description non disponible pour ce métier.")


def update_search_names(data):
    # Parcourir chaque dictionnaire dans la liste
    for item in data:
        # Vérifier si 'search_name' est dans le dictionnaire
        if "search_name" in item:
            # Remplacer 'Full stack' par 'Développeur Fullstack'
            if "full stack" in item["search_name"].lower():
                item["search_name"] = "Développeur Fullstack"
            elif "fullstack" in item["search_name"].lower():
                item["search_name"] = "Développeur Fullstack"
            elif "data ingénieur" in item["search_name"].lower():
                item["search_name"] = "Développeur Fullstack"
            elif "full-stack" in item["search_name"].lower():
                item["search_name"] = "Développeur Fullstack"
            elif "data analyste" in item["search_name"].lower():
                item["search_name"] = "Data analyst"
            # Remplacer 'front-end' par 'Développeur Front End'
            elif item["search_name"] == "front-end":
                item["search_name"] = "Développeur Front End"
            elif item["search_name"] == "front end":
                item["search_name"] = "Développeur Front End"
            elif item["search_name"] == "Développeur":
                item["search_name"] = "Développeur logiciel"
            elif item["search_name"] == "UX":
                item["search_name"] = "UX Designer"
            elif item["search_name"] == "seo":
                item["search_name"] = "Specialist SEO ⭐"
            elif item["search_name"] == "q&a":
                item["search_name"] = "Développeur logiciel"
    # Retourner la liste modifiée
    return data


import random


def affichage_presentation(search_names):

    split = st.columns([2, 0.5, 2], gap="large")

    with split[0]:

        st.markdown(
            """
            <style>
                .main-container {
                    padding: 4rem 1rem;
                }
                .pretitle {
                    font-size: 3vw;
                    color:#302838;
                    font-weight: lighter;
                    margin-bottom: 0;
                    text-align: center;
                }
                .pretitle_ {
                    font-size: 2vw;
                    color:#302838;
                    font-weight: bolder;
                    margin-bottom: 0px;
                    text-align: center;
                }
                
                .title {
                    font-size: 4vw;
                    font-weight: bold;
                    margin-bottom: 0.5em;
                    color:#302838;
                    text-align: center;
                } 

                .slogan {
                    font-size: 1.5vw;
                    color:#302838;
                    color: darkblue;
                    text-align: left;
                    
                }
                .description {
                    font-size: 1vw;
                    
                    color:#302838;
                    text-align: left;
                    margin-top: 20px;
                }
            </style>
        """,
            unsafe_allow_html=True,
        )

        # Header section
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="pretitle">The</h2>', unsafe_allow_html=True)
        st.markdown('<h1 class="title">Market</h1>', unsafe_allow_html=True)
        st.markdown(
            "<p class='slogan'>Restez competitif grace à l'analyse du <b>marché</b></p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p class="description">
            Gardez un avantage compétitif sur le marché en analysant les technologies en vogue et les <strong>hard skills</strong> nécessaires. Adoptez l'attitude adéquate avec les <strong>soft skills</strong> pertinents, assurez-vous d'avoir le diplôme requis, et maîtrisez les <strong>niveaux de langue</strong> nécessaires pour vous démarquer. <br>Notre IA analyse chaque jour des milliers d'offres d'emploi, vous permettant de rester à la pointe de l'évolution du marché et de vous positionner stratégiquement pour les opportunités à venir.
            </p>
            """,
            unsafe_allow_html=True,
        )

    # Ajout de CSS personnalisé pour améliorer l'apparence

    with split[2]:

        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown('<div class="main-container">', unsafe_allow_html=True)

        st.markdown(
            '<h3 class="pretitle_">Explorez les carrières passionnantes dans le domaine de la technologie !</h3>',
            unsafe_allow_html=True,
        )
        st.markdown("#")
        st.markdown("#")
        cols = st.columns(3)
        for index, job in enumerate(search_names):
            with cols[index % 3]:
                if st.button(
                    "**" + job + "**", key=job, on_click=set_job_and_rerun, args=(job,)
                ):
                    pass  # The bs  # The button click handling is in the set_job_and_rerun function


import base64


def get_base64_of_bin_file(file_path):
    """Encode le fichier binaire en base64."""
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def set_job_and_rerun(job):
    st.session_state.selected_job = job
    st.session_state.phase = 2
    st.session_state.page = 0


def affichage_analyse(chemin_analyse):

    if "phase" not in st.session_state:
        st.session_state.phase = 1
        st.session_state.selected_job = ""

    # Charger les données
    base_analyse = load_json_data(chemin_analyse)

    base_analyse = update_search_names(base_analyse)

    # Extraire les options uniques pour le filtre 'search_name'
    search_names = list(
        {
            item["search_name"].capitalize()
            for item in base_analyse
            if "search_name" in item
        }
    )

    if st.session_state.phase == 1:
        affichage_presentation(search_names)

    elif st.session_state.phase == 2:

        # Créer un widget de sélection pour le filtre

        selected_name = st.session_state.selected_job
        # Filtrer les données basées sur le choix
        filtered_data = [
            item
            for item in base_analyse
            if item.get("search_name").lower() == selected_name.lower()
        ]

        # Afficher les données filtrées

        st.markdown("#")
        st.markdown("#")

        col1, col2 = st.columns([0.2, 0.8], gap="large")
        with col2:
            entete(selected_name, obtenir_description_metier(selected_name))

        with col1:
            st.button(
                "Decouvrir d'autre metier",
                on_click=lambda: setattr(st.session_state, "phase", 1),
            )
            st.markdown("#")
            st.success(
                f"**Plus de {math.ceil((len(filtered_data) * 2) / 100) * 100 * 3} offres d'emploi analysés**"
            )
            # Convertir les chaînes de dates en objets datetime et trouver la date maximale
            dates = []
            for item in filtered_data:
                try:
                    date_str = item.get("date", [])[0]
                    # Ensure the date string is not empty and correctly formatted
                    if datetime.strptime(date_str, "%Y-%m-%d"):
                        dates.append(datetime.strptime(date_str, "%Y-%m-%d"))
                except ValueError:
                    print(f"Skipping invalid date: {date_str}")

            if dates:
                max_date = max(dates)
                days_since_last_update = (
                    datetime.today().date() - max_date.date()
                ).days

            # Préparation de l'affichage Streamlit
            if days_since_last_update == 0:
                st.warning(f"⏰⏰ **Dernière mise à jour aujourd'hui**")

            elif days_since_last_update == 1:
                st.warning(f"⏰⏰ **Dernière mise à jour hier**")

            else:
                st.warning(
                    f"⏰⏰ **Dernière mise à jour il y a {days_since_last_update} jours**"
                )

        st.markdown("#")
        st.markdown("#")
        st.markdown("#")
        top_hard_skills, median, moyenne = calculate_top_hard_skills(filtered_data)
        top_soft_skills, median_ss, moyenne_ss = calculate_top_soft_skills(
            filtered_data
        )

        retribution_HS(top_hard_skills, selected_name, median, moyenne)
        st.markdown("#")
        st.markdown("#")
        retribution_SS(top_soft_skills, selected_name, median_ss, moyenne_ss)
        st.markdown("#")
        st.markdown("#")
        retribution_DIPL(filtered_data)
        st.markdown("#")
        st.markdown("#")
        retribution_LANGUE(filtered_data)
        st.markdown("#")
        st.markdown("#")
        retribution_EXP(filtered_data)
        st.markdown("#")
        st.markdown("#")
        retribution_TACHE(filtered_data)

    #     html_diplome = f"""
    #                         <h3 style='text-align: center; color:rgb(44,63,102);font-size: 1.7vw;'> LANGUE </h3>
    #                     """
    #     st.markdown(html_diplome, unsafe_allow_html=True)
    #     # st.write(filtered_data)
    #     language_data = extract_and_count_languages(filtered_data)
    #     df = prepare_data_for_plotting(language_data)
    #     plot_language_levels(df)

    # with split3:
    #     html_diplome = f"""
    #                         <h3 style='text-align: center; color:rgb(44,63,102);font-size: 1.7vw;'> TACHES </h3>
    #                 """
    #     st.markdown(html_diplome, unsafe_allow_html=True)

    #     tasks = load_and_combine_tasks(filtered_data)
    #     filtered_tasks = filter_tasks_with_min_tokens(tasks)
    #     display_tasks(filtered_tasks)


@st.cache_data(ttl=3600, show_spinner=False)
def cache_matching(result_cv: dict, result_job_offer: dict):
    return matching_offer_with_candidat(result_cv, result_job_offer)


from collections import defaultdict, Counter


def load_and_combine_tasks(json_list):
    """Combine les tâches de plusieurs objets JSON en une seule liste."""
    all_tasks = []
    for item in json_list:
        if "tache" in item:
            all_tasks.extend(item["tache"])
    return all_tasks


def filter_tasks_with_min_tokens(tasks, min_tokens=3):
    """
    Filtre les tâches qui contiennent au moins 'min_tokens' tokens, les déduplique,
    et les trie par ordre décroissant du nombre de tokens.

    Args:
    tasks (list of str): Liste des tâches à filtrer.
    min_tokens (int): Nombre minimum de tokens requis dans une tâche.

    Returns:
    list of str: Liste des tâches filtrées, sans doublons et triées.
    """
    # Filtrer et dédupliquer les tâches tout en préservant l'ordre
    seen = set()
    filtered_tasks = []
    for task in tasks:
        if task not in seen and len(task.split()) >= min_tokens:
            seen.add(task)
            filtered_tasks.append(task)

    # Trier les tâches par nombre décroissant de tokens
    filtered_tasks.sort(key=lambda task: len(task.split()), reverse=True)

    return filtered_tasks


def display_tasks(tasks):
    """Affiche les tâches filtrées avec pagination et style HTML/CSS."""
    tasks_per_page = 15  # Nombre total de tâches par page
    tasks_per_column = 5  # Nombre maximum de tâches par colonne

    # Initialisation de la pagination si elle n'existe pas déjà dans l'état de session
    if "page" not in st.session_state:
        st.session_state.page = 0

    # Calcul du nombre total de pages
    total_pages = len(tasks) // tasks_per_page + (
        1 if len(tasks) % tasks_per_page > 0 else 0
    )

    # Calcul de l'index de début et de fin des tâches à afficher
    start = st.session_state.page * tasks_per_page
    end = start + tasks_per_page
    displayed_tasks = tasks[start:end]

    # Mise en page CSS pour améliorer l'affichage des tâches
    st.markdown(
        """
    <style>
    .task-item {
        border-radius: 10px;
        border: 2px solid #443266;
        margin: 5px;
        font-size:0.6vw;
        font-weight:bold;
        padding: 10px;
        font-family: Arial, sans-serif;
        background-color:  #F1F0FF;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Création des colonnes pour l'affichage des tâches
    cols = st.columns(3)
    col_index = 0

    # Remplissage des colonnes avec les tâches
    for i, task in enumerate(displayed_tasks):
        with cols[col_index]:
            st.markdown(
                f'<div class="task-item">{task.capitalize()}</div>',
                unsafe_allow_html=True,
            )
        if (i + 1) % tasks_per_column == 0:
            col_index = (
                col_index + 1
            ) % 3  # Passer à la colonne suivante après 5 tâches

    # Boutons de navigation pour la pagination
    with st.container():
        if st.session_state.page > 0:
            if st.button("Précédent", key="prev"):
                st.session_state.page -= 1

        if st.session_state.page < total_pages - 1:
            if st.button("Suivant", key="next"):
                st.session_state.page += 1


def extract_and_count_languages(jobs):
    language_data = defaultdict(Counter)

    for job in jobs:
        if "langue" in job:
            for lang_string in job["langue"]:
                parts = lang_string.split(maxsplit=1)
                if len(parts) > 1:
                    langue, niveau = parts[0], parts[1]
                    language_data[langue][niveau] += 1

    return language_data


def prepare_data_for_plotting(language_data):
    # Calculer le total global pour toutes les langues et niveaux
    global_total = sum(sum(niveaux.values()) for niveaux in language_data.values())

    rows = []
    # Filtrer seulement les langues spécifiées
    specified_languages = [
        "Anglais",
        "Français",
        "Espagnol",
        "Italien",
        "Russe",
        "Arabe",
    ]
    for langue, niveaux in language_data.items():
        if langue in specified_languages:
            langue_total = sum(niveaux.values())
            for niveau, count in niveaux.items():
                # Calculer le pourcentage par rapport au total global
                percentage = (count / global_total) * 100
                rows.append(
                    {"Langue": langue, "Niveau": niveau, "Pourcentage": percentage}
                )

    return pd.DataFrame(rows)


def plot_language_levels(df):
    # Tracer les parts de chaque langue avec niveaux
    fig = px.bar(
        df,
        x="Langue",
        y="Pourcentage",
        color="Niveau",
        labels={"Pourcentage": "Pourcentage (%)"},
        barmode="group",
        category_orders={
            "Niveau": ["Débutant", "Intermédiaire", "Avancé", "Expert"]
        },  # Assurez-vous que les niveaux sont corrects
    )
    fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside")
    fig.update_layout(
        xaxis_title="Langue",
        yaxis_title="Pourcentage (%)",
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False
        ),  # Suppression de l'axe y et de la grille
        xaxis=dict(showgrid=False),  # Suppression de la grille de l'axe x
    )
    st.plotly_chart(fig, use_container_width=True)


def add_logo_tiny(logo_path, width, height):
    """Read and return a resized logo"""

    logo = Image.open(logo_path)
    new_image = logo.resize((287, 175))
    return new_image


def hash_uploaded_file(
    uploaded_file: st.runtime.uploaded_file_manager.UploadedFile,
) -> dict:
    bytes_content: bytes = uploaded_file.read()
    file_name: str = uploaded_file.name
    raw_text: str = extract_text(io.BytesIO(bytes_content)).replace("\x00", "\uFFFD")
    id_pdf_hashed = hashlib.md5(bytes_content).hexdigest()

    return {id_pdf_hashed: {"raw_text": raw_text, "file_name": file_name}}


def nettoyer_intitule(intitule):
    # Supprimer les mentions relatives au genre
    for mention in ["(H/F)", "(h/f)", "H/F", "F/H", "(F/H)", "(f/h)", "f/h"]:
        intitule = intitule.replace(mention, "").strip()
    return intitule


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


def determiner_niveau_experience(offre_emploi):
    offre_emploi["niveau_experience"] = "Non spécifié"
    offre_emploi["annees_experience"] = 0

    exp_exige = offre_emploi.get("experienceExige")
    exp_libelle = offre_emploi.get("experienceLibelle", "")

    # Extraire le nombre depuis exp_libelle
    nombre_exp = re.search(r"\d+", exp_libelle)
    if nombre_exp:
        nombre_exp = nombre_exp.group()
        if "mois" in exp_libelle:
            # Convertir les mois en années
            nombre_annees = round(int(nombre_exp) / 12, 1)
        else:
            nombre_annees = int(nombre_exp)
    else:
        # Si aucun nombre n'est trouvé, renvoyer directement l'offre avec les valeurs par défaut
        return offre_emploi["niveau_experience"]

    if exp_exige == "D" or (exp_exige == "E" and nombre_annees < 3):
        offre_emploi["niveau_experience"] = "Junior"
    elif exp_exige == "E" and nombre_annees < 7:
        offre_emploi["niveau_experience"] = "Intermédiaire"
    elif exp_exige == "E" and nombre_annees >= 7:
        offre_emploi["niveau_experience"] = "Senior"

    offre_emploi["annees_experience"] = nombre_annees

    return offre_emploi["niveau_experience"]


@st.cache_data(ttl=3600, show_spinner=False)
def on_match_button_click(offres):
    # st.write(st.session_state)
    # Récupération du CV parsé depuis le session state
    parsed_text = st.session_state.get("parsed_resume", {})
    selected_offres_ids = st.session_state.get("indices_selectionnes", [])

    # Filtrer les offres en utilisant les identifiants uniques
    selected_offres = [
        offre for offre in offres if offre["id_unique"] in selected_offres_ids
    ]

    parsed_offres = []  # Liste pour stocker les offres après parsing

    with st.spinner(
        "**:green[Nous analysons les besoins specifiques à ces opportunitées...]**"
    ):
        # Initialisation de la liste pour les offres parsées
        parsed_offres = []

        # Parcourir les identifiants uniques des offres sélectionnées
        for id_unique in st.session_state["indices_selectionnes"]:
            # Récupérer l'offre correspondante à l'identifiant unique
            offre = next((o for o in offres if o["id_unique"] == id_unique), None)

            if offre is not None:
                # Parsing de la description de l'offre
                parsed_offre = cache_resume(offre["description"])

                # Mise à jour des champs de l'offre parsée
                nom_compose = nettoyer_intitule(offre["intitule"])
                parsed_offre["nom_prenom"] = nom_compose
                parsed_offre["numero"] = offre["entreprise"].get(
                    "nom", "Entreprise Non Spécifiée"
                )
                parsed_offre["mail"] = offre.get("salaire", {}).get(
                    "libelle", "Salaire non spécifié"
                )

                # Préparation de l'objet parsed_offre pour stocker l'URL prioritaire
                # Tentative d'extraire 'urlPostulation' depuis 'contact'
                contact = offre.get(
                    "contact", {}
                )  # Retourne un dictionnaire vide si 'contact' n'existe pas
                url_postulation = contact.get(
                    "urlPostulation", ""
                )  # Retourne une chaîne vide si 'urlPostulation' n'existe pas

                # Utiliser 'urlOrigine' si 'urlPostulation' est vide
                url_finale = (
                    url_postulation
                    if url_postulation
                    else offre["origineOffre"].get("urlOrigine", "")
                )

                parsed_offre["id_pdf_hashed"] = url_finale

                # Ajout des compétences à la liste 'tache'
                competences_libelles = [
                    comp["libelle"] for comp in offre.get("competences", [])
                ]
                parsed_offre.setdefault("tache", []).extend(competences_libelles)
                # Ajout des compétences à la liste soft skills
                competences_pro = [
                    comp["libelle"]
                    for comp in offre.get("qualitesProfessionnelles", [])
                ]
                parsed_offre.setdefault("soft_skills", []).extend(competences_pro)

                # Ajout des compétences à la liste Langue
                competences_langue = [
                    comp["libelle"] for comp in offre.get("langues", [])
                ]

                parsed_offre.setdefault("langue", []).extend(competences_langue)

                parsed_offre["langue"] = [
                    langue
                    for langue in parsed_offre["langue"]
                    if not any(
                        francais in langue.lower()
                        for francais in [
                            "français",
                            "française",
                            "francais",
                            "francaise",
                        ]
                    )
                ]

                # Ajout du niveau de formation à 'diplome' si disponible
                niveau_formations = [
                    form.get("niveauLibelle", "")
                    for form in offre.get("formations", [])
                    if "niveauLibelle" in form
                ]
                parsed_offre.setdefault("diplome", []).extend(niveau_formations)

                parsed_offre["ecole"] = offre.get("typeContrat", "Non spécifié")

                parsed_offre["time_in_field"] = determiner_niveau_experience(offre)
                # Ajouter l'offre parsée à la liste des offres parsées
                parsed_offres.append(parsed_offre)

        st.session_state["parsed_offres"] = parsed_offres

    with st.spinner("**:blue[Nous trouvons vos atouts pour ces opportunités]**"):
        # Réalisation du matching pour chaque offre parsée avec le CV
        result_matching = [
            cache_matching(parsed_text, parsed_offre) for parsed_offre in parsed_offres
        ]

        # st.write(result_matching)
        # Tri de result_matching par "MATCH_SCORE" en ordre décroissant

        result_matching = sorted(
            result_matching, key=lambda x: x["MATCH_SCORE"], reverse=True
        )

        # Enregistrement des résultats du matching dans le session state
        st.session_state["result_matching"] = result_matching

        st.session_state["matching_stage"] = 2
        st.rerun()  # Indiquez que le matching est terminé


# @st.cache_data(ttl=3600, show_spinner=False)
# def run_parse_resume(uploaded_files, username):
#     return parse_list_resume(uploaded_files, username)


def set_matching_state(i: int, type_: str):
    st.session_state["matching_stage"] = i
    st.session_state["matching_type"] = type_


def afficher_messages():
    messages = [
        "L'oiseau fait son nid **upload**",
        "Chargement en cours...",
        "Vérification des données...",
        "Préparation de l'analyse...",
        "Presque terminé...",
        "Terminé !",
    ]

    for message in messages:
        st.spinner(message)
        time.sleep(15)  # Pause de 15 secondes entre chaque message
        st.spinner(empty=True)  # Efface le spinner


def display_initial_message():
    if st.session_state.get("matching_stage", 0) == 0:
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
                font-weight: lighter;
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
                background-color: #f0f8ff; /* Couleur pastel */
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

        coler = st.columns(2, gap="large")
        with coler[0]:
            st.image("./static/candidat.jpg", use_column_width=True)
        # Contenu de l'analyse et du matching

        # Afficher le contenu HTML dans Streamlit

        with coler[1]:
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown(
                '<h1 class="title">I.A au service de votre <bold> carrière</bold> </h1>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <p style='font-size:1.3vw;'>
                    Votre CV est <b>analysé</b> et <b>testé</b> sur les principaux <b>job boards</b> du moment afin de vous proposer le <b>job</b> qui correspond le mieux à vos compétences et aspirations.
                </p>
                """,
                unsafe_allow_html=True,
            )

            coli1, coli2, coli3 = st.columns([1, 2, 1])
            with coli2:
                uploaded_file = st.file_uploader(" ", type="pdf")

            with coli2:

                bouton_validation = st.button("Lancez la recherche")
                if uploaded_file is not None and bouton_validation is not None:
                    with coli2:
                        st.success(
                            "**La section Job Finder est disponible en demonstration** rendez vous dans la section About Nest "
                        )
                        st.empty()
                        # # Stockez temporairement le fichier pour éviter la perte lors du rechargement
                        st.session_state["temp_uploaded_file"] = uploaded_file
                        st.session_state["fichier_telecharger"] = 1
                        # # Changement de l'état pour passer au parsing
                        st.session_state["matching_stage"] = 1

                        reinitialiser_pagination_et_selections()


@st.cache_resource
def rechercher_offres_emploi(mots_cles, mot_cle_repli="dev_web"):
    client = Api(
        client_id="PAR_bottwiiter_6253422b7e25285a1895dc28fcf76733f93ca17cad6a9e0aca8c9de42a8364c0",
        client_secret="b8f09d357980a24a9b4869081a8d28471a868fae7aa148364ecbf7d9c539a78a",
    )
    # mots_cles = mots_cles[:2]
    # st.write(mots_cles)

    start_dt = datetime.now() - timedelta(days=7)
    end_dt = datetime.now()
    params_base = {
        "minCreationDate": dt_to_str_iso(start_dt),
        "maxCreationDate": dt_to_str_iso(end_dt),
    }

    resultats_totaux = []
    mots_cles_effectues = 0

    tous_mots_cles = mots_cles + [mot_cle_repli]

    for mot_cle in mots_cles + [mot_cle_repli]:
        if mots_cles_effectues >= 3:  # Si 3 mots clés ont déjà réussi, on arrête.
            break

        else:
            try:
                params = params_base.copy()
                params["motsCles"] = mot_cle
                resultats_recherche = client.search(params=params)
                if resultats_recherche.get("resultats"):
                    resultats_totaux.extend(resultats_recherche["resultats"])
                mots_cles_effectues += 1
            except Exception as e:
                print(f"Erreur lors de la recherche pour le mot-clé '{mot_cle}': {e}")

    # Dédoublonner les offres avec gestion des clés manquantes
    offres_uniques = {}
    for offre in resultats_totaux:
        # Utiliser des valeurs par défaut si les clés sont manquantes
        libelle = offre.get("libelle", "Libellé non spécifié")
        nom_entreprise = offre.get("entreprise", {}).get(
            "nom", "Entreprise non spécifiée"
        )
        lieu_travail = offre.get("lieuTravail", {}).get(
            "libelle", "Lieu de travail non spécifié"
        )

        cle_unique = (libelle, nom_entreprise, lieu_travail)

        if cle_unique not in offres_uniques:
            offres_uniques[cle_unique] = offre

    # Attribuer un ID unique à chaque offre
    offres_avec_id = []
    unique_id_generator = count(start=1)  # Génère des ID uniques à partir de 1
    for offre in offres_uniques.values():
        offre_id = next(unique_id_generator)
        offre["id_unique"] = offre_id
        offres_avec_id.append(offre)

    return offres_avec_id


@st.cache_resource
def filtrer_offres_par_region(offres, region_selectionnee):
    if region_selectionnee == "Toutes les régions":
        return offres
    else:
        return [offre for offre in offres if offre["region"] == region_selectionnee]


@st.cache_resource
def changer_page(direction):
    current_page = st.session_state.get("page_actuelle", 1)
    if direction == "prev" and current_page > 1:
        st.session_state.page_actuelle -= 1
    elif direction == "next" and current_page < st.session_state.get("nombre_pages", 1):
        st.session_state.page_actuelle += 1
    st.rerun()


def reinitialiser_pagination_et_selections():
    st.session_state["page_actuelle"] = 1
    st.session_state["indices_selectionnes"] = []


def filtrer_offres_par_region_contrat_niveau(
    offres, region_selectionnee, type_contrat_selectionne, niveau_experience_selectionne
):
    # Filtrer d'abord par région si une région spécifique est sélectionnée
    if region_selectionnee != "Toutes les régions":
        offres_filtrees = [
            offre for offre in offres if offre["region"] == region_selectionnee
        ]
    else:
        offres_filtrees = offres[:]

    # Ensuite, filtrer par type de contrat si un type spécifique est sélectionné
    if type_contrat_selectionne != "Tous les contrats":
        offres_filtrees = [
            offre
            for offre in offres_filtrees
            if offre["type_contrat"] == type_contrat_selectionne
        ]

    # Enfin, filtrer par niveau d'expérience si un niveau spécifique est sélectionné
    if niveau_experience_selectionne != "Tous les niveaux d'expérience":
        offres_filtrees = [
            offre
            for offre in offres_filtrees
            if offre.get("niveau_experience", "Non spécifié")
            == niveau_experience_selectionne
        ]

    return offres_filtrees


def afficher_offres_emploi(
    offres, page_actuelle, offres_par_page, nombre_pages, region_prospect
):
    # Initialisation de la liste des indices sélectionnés s'il n'existe pas dans st.session_state
    if region_prospect == "":
        region_prospect = "Toutes les régions"

    if "indices_selectionnes" not in st.session_state:
        st.session_state.indices_selectionnes = []

    if "region_selectionnee" not in st.session_state:
        st.session_state.region_selectionnee = region_prospect

    if "page_actuelle" not in st.session_state:
        st.session_state.page_actuelle = 1
    if "filtred" not in st.session_state:
        st.session_state.filtred = []
    if "nombre_pages" not in st.session_state:
        st.session_state.nombre_pages = 1

    regions_uniques = ["Toutes les régions"] + sorted(
        set(offre["region"] for offre in offres)
    )

    contrat_uniques = ["Tous les contrats"] + sorted(
        set(offre["type_contrat"] for offre in offres)
    )

    niveaux_experience_uniques = ["Tous les niveaux d'expérience"] + sorted(
        set(offre["niveau_experience"] for offre in offres)
    )

    # Calculer si le maximum de sélections est atteint
    max_selection_atteint = len(st.session_state["indices_selectionnes"]) > 5
    col1, col2 = st.columns([5, 1])
    with col2:
        bouton_disabled = not st.session_state["indices_selectionnes"]
        match_but = st.button("Matchez !", disabled=bouton_disabled)
    if match_but:
        # st.write(offres)

        # st.rerun()
        on_match_button_click(offres)

        # Réinitialisez l'étape pour revenir au téléchargement
        # Votre logique de traitement ici...

        # Utilisez st.experimental_rerun() pour forcer le script à s'exécuter à nouveau depuis le début
        st.rerun()

    # Calculer si le maximum de sélections est atteint
    max_selection_atteint = len(st.session_state["indices_selectionnes"]) > 5

    col1, col2 = st.columns([5, 1], gap="small")
    with col1:
        if st.session_state.region_selectionnee in regions_uniques:
            default_index = regions_uniques.index(st.session_state.region_selectionnee)
        else:
            default_index = (
                0  # Default to the first item if the selected region is not in the list
            )
        with st.expander("**FILTRES**"):
            region_selectionnee = st.selectbox(
                "Mobilité souhaitée:",
                regions_uniques,
                index=default_index,
                key="region_select",
                on_change=lambda: setattr(
                    st.session_state, "page_actuelle", 1
                ),  # Réinitialiser la pagination à la sélection d'une nouvelle région
            )

            # Déterminer le niveau d'expérience du candidat
            niveau_candidat = generate_stars_for_seniority_(
                st.session_state["parsed_resume"]["time_in_field"]
            )

            # Trouver l'index du niveau dans la liste des niveaux d'expérience, sinon utiliser "Tous les niveaux"
            if niveau_candidat in niveaux_experience_uniques:
                index_defaut = niveaux_experience_uniques.index(niveau_candidat)
            else:
                index_defaut = (
                    0  # "Tous les niveaux" si aucun niveau spécifique ou non trouvé
                )

            # Sélecteur Streamlit pour choisir le niveau d'expérience
            niveau_selectionne = st.selectbox(
                "Niveau d'expérience souhaité :",
                options=niveaux_experience_uniques,
                index=index_defaut,  # Utiliser l'index trouvé ou "Tous les niveaux" par défaut
                key="niveau_experience_select",
            )

            type_contrat_selectionne = st.selectbox(
                "Sélectionnez le type de contrat :",
                options=contrat_uniques,
                index=0,  # Définissez l'index de la valeur par défaut, 0 pour "Tous les contrats"
                key="contrat_select",
            )

        offres_filtrees = filtrer_offres_par_region_contrat_niveau(
            offres,
            region_selectionnee,
            type_contrat_selectionne,
            niveau_selectionne,
        )
        st.session_state.nombre_pages = max(
            1,
            len(offres_filtrees) // offres_par_page
            + (len(offres_filtrees) % offres_par_page > 0),
        )
        nb_offres = len(offres)
        nb_offres_region = len(
            [offre for offre in offres if offre["region"] == region_prospect]
        )  # Calcul du nombre d'offres dans offres_page

        html_content = f"""
            <style>
                .info-container {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: #fde9e5; /* Un bleu clair pour un aspect frais et accueillant */
                    border-left: 5px solid #1B4F72; /* Une bordure bleu foncé pour l'élégance */
                    padding: 20px;
                    margin-bottom: 20px;
                    border-radius: 8px; /* Des coins arrondis pour un look moderne */
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Une ombre subtile pour du relief */
                }}
                .info-title {{
                    font-size: 20px; /* Adaptation de la taille pour une meilleure lisibilité */
                    color: #1B4F72; /* Reprend le bleu foncé pour une cohérence visuelle */
                    font-weight: 600;
                    margin-bottom: 15px; /* Un peu plus d'espace sous le titre */
                    text-align: center;
                }}
                .info-content {{
                    font-size: 16px; /* Légère augmentation pour l'accessibilité */
                    color: #212529; /* Un gris foncé pour le texte, facile à lire */
                    text-align: center;
                    line-height: 1.5; /* Amélioration de l'espacement des lignes pour la lisibilité */
                }}
                .action-button {{
                    display: block;
                    width: max-content;
                    margin: 20px auto; /* Centrage du bouton */
                    padding: 10px 20px;
                    background-color: #fde9e5; /* Bleu foncé pour le bouton */
                    color: #ffffff; /* Texte blanc pour le contraste */
                    border-radius: 5px; /* Bords arrondis pour le bouton */
                    text-align: center;
                    text-decoration: none; /* Suppression du soulignement du texte */
                    font-weight: bold;
                }}
            </style>
            <div class="info-container">
                <div class="info-title">Opportunités à Saisir</div>
                <div class="info-content">Parmi les <strong>{nb_offres} opportunitées</strong> disponibles, <strong>{nb_offres_region} se trouvent dans votre région</strong>. Sélectionnez celles qui éveillent votre intérêt et entamez le processus de matching pour découvrir un chemin professionnel enrichissant.</div>
            </div>
            """
        st.markdown(html_content, unsafe_allow_html=True)

        debut = (st.session_state.page_actuelle - 1) * offres_par_page
        fin = debut + offres_par_page
        offres_page = offres_filtrees[debut:fin]
        if not offres_page:
            st.markdown("#")
            st.warning(
                """
                Essayez d'**ajuster vos critères** de sélection pour découvrir encore plus d'opportunités passionnantes.
                car il semblerait que nous n'ayons pas d'offres correspondant exactement à **vos critères actuels**.
            """
            )
    for offre in offres_page:
        id_unique_offre = offre["id_unique"]  # Utiliser l'ID unique de l'offre

        col1, col2 = st.columns([5, 1], gap="small")

        with col2:
            # Vérifier si l'offre est déjà sélectionnée en utilisant son ID unique
            est_deja_selectionne = (
                id_unique_offre in st.session_state["indices_selectionnes"]
            )
            est_desactive = max_selection_atteint and not est_deja_selectionne

            # Utiliser l'ID unique comme clé pour la checkbox
            est_selectionne = st.checkbox(
                "Selectionnez",
                value=est_deja_selectionne,
                key=f"select_{id_unique_offre}",  # Clé unique basée sur l'ID de l'offre
                disabled=est_desactive,
            )
        # st.write(st.session_state)
        with col1:
            job_name = (
                offre["intitule"]
                .replace("(H/F)", "")
                .replace("(h/f)", "")
                .replace("h/f", "")
                .replace("H/F", "")
                .replace("(F/H)", "")
                .replace("(f/h)", "")
                .replace("f/h", "")
                .replace("F/H", "")
                .strip()
            )
            with st.expander(
                f"**{job_name}** chez {offre['entreprise'].get('nom', 'Non spécifié')}"
            ):
                st.write(f"**Description:** {offre['description']}")
                st.write(f"**Lieu de travail:** {offre['lieuTravail']['libelle']}")
                st.write(f"**Type de contrat:** {offre['typeContratLibelle']}")
                salaire = offre["salaire"].get("libelle", "Non spécifié")
                st.write(f"**Salaire:** {salaire}")

        # Mise à jour des indices sélectionnés en utilisant l'ID unique
        if est_selectionne and not est_deja_selectionne:
            st.session_state["indices_selectionnes"].append(id_unique_offre)
        elif not est_selectionne and est_deja_selectionne:
            st.session_state["indices_selectionnes"].remove(id_unique_offre)

    col1, col2 = st.columns(2)

    with col1:
        if st.session_state.get("page_actuelle", 1) > 1:
            st.button("Page précédente", on_click=lambda: changer_page("prev"))

    with col2:
        if st.session_state.get("page_actuelle", 1) < st.session_state.get(
            "nombre_pages", 1
        ):
            st.button("Page suivante", on_click=lambda: changer_page("next"))

        # Maintenant, utilisez st.session_state.page_actuelle pour afficher le contenu de la page actuelle

    st.write(
        f"Vous êtes sur la page {st.session_state.page_actuelle} sur {st.session_state.nombre_pages}"
    )
    # st.write(offres)
    # st.write(st.session_state["indices_selectionnes"])
    # st.write(st.session_state)
    st.warning(
        f"""Vous avez selectionés {len(st.session_state["indices_selectionnes"])} offres potentielles"""
    )


def generate_stars_for_seniority(years_of_experience):
    if years_of_experience < 3:
        stars_filled = 0
        seniority_text = "Junior"
    elif years_of_experience < 7:
        stars_filled = 2
        seniority_text = "Intermédiaire"
    else:
        stars_filled = 5
        seniority_text = "Senior"

    stars_html = '<div style="color: black; margin-bottom: 1vw;font-weight: bold;">{}</div>'.format(
        seniority_text
    )

    return stars_html


def afficher_infos_candidat(data):
    st.write("#")
    st.write("#")

    html_content = f"""
    <style>
        .candidat-container {{
            font-family: Arial, sans-serif;
            background-color: #E8F0FE; /* Fond gris clair */
            border-radius: 20px;
            box-shadow: 0px 8px 8px rgba(0,0,0,0.1); /* Ombre plus douce */
            padding: 20px;
            max-width: 600px;
            margin: auto;
        }}
        .candidat-title {{
            font-size: 24px;
            font-weight: bolder;
            color: #1B4F72; /* Bleu Musala */
            margin-bottom: 20px;
        }}
        .candidat-section {{
            margin-bottom: 15px;
        }}
        .candidat-section-title {{
            color: #1B4F72; /* Bleu Musala */
            font-size: 18px;
            margin-bottom: 10px;
            font-weight: bold;
            text-align: center;
        }}
        .candidat-info {{
            font-size: 16px;
            line-height: 1.5;
            text-align: center;
            font-weight: bold;
            color:black;
            
        }}
    </style>
    <div class="candidat-container">
        <div class="candidat-section">
            <div class="candidat-section-title">Profil Candidat</div>
            <div class="candidat-info">{clarifier_nom(data['job_type'])}</div>
        </div>
        <div class="candidat-section">
            <div class="candidat-section-title">Seniorité</div>
            <div class="candidat-info">{generate_stars_for_seniority(data['time_in_field'])}</div>
        </div>
        <div class="candidat-section">
            <div class="candidat-section-title">Localisation</div>
            <div class="candidat-info">{data['location'][0].capitalize()}</div>
        </div>
        <div class="candidat-section">
            <div class="candidat-section-title">Formation Académique</div>
            <div class="candidat-info">{diploma(
                mise_en_forme_diplome(data["diplome"])["note"].max()
            )}</div>
        </div>
        <div class="candidat-section">
            <div class="candidat-section-title">Compétences Linguistiques</div>
            <div class="candidat-info">{' - '.join([x for x in MEP_LANGUE(traitement_langue(data['langue']))])}</div>
        </div>
        <div class="candidat-section">
            <div class="candidat-section-title">Aptitudes Comportementales</div>
            <div class="candidat-info">{' - '.join([x.capitalize() for x in  encoder_et_obtenir_equivalences_dedupliquees(data['soft_skills'])])}</div>
        </div>
        <div class="candidat-section">
            <div class="candidat-section-title">Expertise Technique</div>
            <div class="candidat-info">{' - '.join([x.capitalize() for x in data['hard_skills']])}</div>
        </div>
        <div class="candidat-section">
            <div class="candidat-section-title">Gestion de projet</div>
            <div class="candidat-info">{' - '.join([x.capitalize() for x in data['projet']])}</div>
        </div>
        <div class="candidat-section">
            <div class="candidat-section-title">Responsabilités Clés</div>
            <div class="candidat-info">{' - '.join([x.capitalize() for x in data['tache']])}</div>
        </div>
    </div>
    """

    return html_content


def generate_stars_for_seniority_(years_of_experience):
    if years_of_experience < 3:
        return "Junior"
    elif years_of_experience < 7:
        return "Intermédiaire"
    elif years_of_experience >= 7:
        return "Senior"
    else:
        return None


def ajouter_type_contrat(offres_emploi):
    for offre in offres_emploi:
        # Ajout de 'type_contrat' avec la valeur de 'typeContrat' ou 'Non spécifié' si absent
        offre["type_contrat"] = offre.get("typeContrat", "Non spécifié")
    return offres_emploi


def ajouter_niveau_experience(offres_emploi):
    for offre in offres_emploi:
        offre["niveau_experience"] = "Non spécifié"
        offre["annees_experience"] = 0

        exp_exige = offre.get("experienceExige")
        exp_libelle = offre.get("experienceLibelle", "")

        # Extraire le nombre depuis exp_libelle
        nombre_exp = re.search(r"\d+", exp_libelle)
        if nombre_exp:
            nombre_exp = nombre_exp.group()
            if "mois" in exp_libelle:
                # Convertir les mois en années
                nombre_annees = round(int(nombre_exp) / 12, 1)
            else:
                nombre_annees = int(nombre_exp)
        else:
            continue  # Passer à l'offre suivante si aucun nombre n'est trouvé

        if exp_exige == "D" or (exp_exige == "E" and nombre_annees < 3):
            offre["niveau_experience"] = "Junior"
        elif exp_exige == "E" and nombre_annees < 7:
            offre["niveau_experience"] = "Intermédiaire"
        elif exp_exige == "E" and nombre_annees >= 7:
            offre["niveau_experience"] = "Senior"

        offre["annees_experience"] = nombre_annees

    return offres_emploi


def display_parsed_cv():
    if st.session_state["fichier_telecharger"] == 1:
        # uploaded_file = st.session_state["temp_uploaded_file"]
        # progress_text = "Operation in progress. Please wait."

        st.empty()
        with st.spinner(
            "**Prenons le temps de nous connaitre... cela nous prendra moins d'une minute** "
        ):
            if st.button("Retour au choix du CV"):
                # Réinitialisez l'étape pour revenir au téléchargement
                st.session_state["matching_stage"] = 0
                if "temp_uploaded_file" in st.session_state:
                    del st.session_state["temp_uploaded_file"]
                st.session_state["parsed_resume"] = None
                st.session_state["fichier_telecharger"] = 0
                st.session_state["offres_emploi"] = []
                del st.session_state.region_selectionnee
                # Utilisez st.rerun() pour forcer le script à s'exécuter à nouveau depuis le début
                st.rerun()

            if st.session_state["parsed_resume"] is None:
                uploaded_file = st.session_state["temp_uploaded_file"]

                parsed_text = cache_resume(uploaded_file)
                st.session_state["parsed_resume"] = parsed_text
                del st.session_state["temp_uploaded_file"]
            else:
                parsed_text = st.session_state["parsed_resume"]
                # Effacez l'écran ici si nécessaire. Streamlit n'a pas une fonction native pour "effacer l'écran",
            # mais vous pouvez contrôler ce qui est affiché via l'état.
            st.markdown(
                "#"
            )  # Utilisé ici juste pour ajouter un espace, ajustez selon besoin
            # st.write(st.session_state)
            # st.write(parsed_text)
            col, base = st.columns([0.35, 0.65], gap="large")
            with col:
                st.markdown(
                    afficher_infos_candidat(parsed_text), unsafe_allow_html=True
                )
            with base:
                offres_par_page = 10
                if st.session_state["offres_emploi"] == []:
                    offres_emploi = rechercher_offres_emploi(
                        parsed_text["post"], clarifier_nom(parsed_text["job_type"])
                    )
                    offres_emploi = ajouter_region_a_offres_depuis_fichier(
                        offres_emploi,
                        "cities.json",
                    )

                    offres_emploi = ajouter_type_contrat(offres_emploi)
                    offres_emploi = ajouter_niveau_experience(offres_emploi)

                    st.session_state["offres_emploi"] == offres_emploi
                else:
                    offres_emploi = st.session_state["offres_emploi"]
                    offres_emploi = ajouter_region_a_offres_depuis_fichier(
                        offres_emploi,
                        "cities.json",
                    )

                    offres_emploi = ajouter_type_contrat(offres_emploi)
                    offres_emploi = ajouter_niveau_experience(offres_emploi)

                nombre_total_offres = len(
                    offres_emploi
                )  # Remplacer 'offres_emploi' par votre liste d'offres
                nombre_pages = (nombre_total_offres - 1) // offres_par_page + 1

                # Initialiser 'page_actuelle' dans st.session_state si ce n'est pas déjà fait
                if "page_actuelle" not in st.session_state:
                    st.session_state.page_actuelle = 1

                # Boutons de navigation

                # Affichage des offres pour la page actuelle
                afficher_offres_emploi(
                    offres_emploi,
                    st.session_state.page_actuelle,
                    offres_par_page,
                    nombre_pages,
                    parsed_text["location"][0],
                )

                # Afficher le numéro de la page actuelle
                # st.write(f"Page {st.session_state.page_actuelle} sur {nombre_pages}")
    else:
        # Gérez le cas où 'temp_uploaded_file' n'est pas défini (par exemple, en affichant un message ou en redirigeant l'utilisateur)
        st.error(
            "Aucun fichier n'a été téléchargé. Veuillez retourner à l'étape de téléchargement."
        )


def display_matching_results():
    # st.write(st.session_state["result_matching"])
    pass


def job_offer_parser():
    if "result_matching" not in st.session_state:
        st.session_state.result_matching = []
    if "fichier_telecharger" not in st.session_state:
        st.session_state["fichier_telecharger"] = 0
    if "matching_stage" not in st.session_state:
        st.session_state["matching_stage"] = 0
    if "job_offer" in st.session_state:
        del st.session_state["job_offer"]
    if "get_all_cv_from_matching" in st.session_state:
        del st.session_state["get_all_cv_from_matching"]
    if "parsed_resume" not in st.session_state:
        st.session_state["parsed_resume"] = None
    if "offres_emploi" not in st.session_state:
        st.session_state["offres_emploi"] = []

    if "parsed_offres" not in st.session_state:
        st.session_state.parsed_offres = []

    if st.session_state.get("matching_stage", 0) == 0:
        display_initial_message()

    elif st.session_state.get("matching_stage", 0) == 1:
        # st.write(st.session_state)
        # st.write(st.session_state["matching_stage"])

        display_parsed_cv()
    elif (
        st.session_state.get("matching_stage", 0) == 2
    ):  # Assurez-vous que cette étape est bien définie pour le matching
        # st.write(st.session_state)

        # st.write(st.session_state["indices_selectionnes"])

        match_visualisation(st.session_state["result_matching"])


def initial_test():

    col1, col2, col3 = st.columns([0.1, 0.8, 0.1])
    with col2:
        st.markdown("#")
        st.markdown(
            """
            <style>
                /* Effacer tout padding ou margin par défaut de Streamlit */
                .reportview-container .main .block-container {
                    padding-top: 0;
                    padding-right: 1rem;
                    padding-left: 1rem;
                    padding-bottom: 1rem;
                }
                /* Supprimer le footer de Streamlit */
                footer {
                    visibility: hidden;
                }
                .side-column {
                background-color: #2980B9; /* Dark blue color */
                height: 100vh; /* Full height of the viewport */
                    }
                /* Supprimer l'icone de menu de Streamlit */
                header {visibility: hidden;}
                
                /* Style pour le titre */
                .title {
                    color: #3498db; /* Bleu lumineux */
                    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; /* Police élégante */
                    text-align: center;
                    font-size: 3.4vw; /* Taille de la police grande */
                    font-weight: 600; /* Gras */
                    margin-top: 50px; /* Espacement du haut */
                    margin-bottom: 20px; /* Espacement du bas */
                }

                /* Bouton de téléchargement stylisé */
                .upload-button {
                    background-color: #3498db;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    font-size: 16px;
                    cursor: pointer;
                    display: inline-block;
                    text-align: center;
                }
                .description {
               font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
               font-size: 1.3vw;
               color: #34495E; /* Darker text color for better readability */
               text-align: center;
               margin: 20px;
               background-color: #DAE8FC; /* Couleur subtile pour les sections d'explication */
                       padding: 20px;
                       border-radius: 10px;
                       margin-bottom: 20px
                }

            </style>

            <div>
                <h1 class="title">Job<span style=" color:rgb(113,224,203);"> Fit</span></h1>
                <p class=\"description\"> Vous souhaitez postuler à une <b>opportunité</b> une première étape serait de passé les <b>filtres automatique des entreprises</b><br> Nest vous permet d'évaluer vos chances et vous donne les clés pour<b> augmenter vos chances</b></p>
                <p style="font-size: 1.2vw; text-align: center; color: #2C3E50; font-weight: bold; margin-top: 20px;">
                    Votre CV et l'offre d'emploi  <span style="color: rgb(113,224,203); cursor: pointer;" onclick="document.getElementById('file-upload').click();">, voyons si vous êtes un FIT</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    # Utilisation de colonnes pour une meilleure organisation visuelle
    with col2:
        coli1, coli2, coli3, coli4 = st.columns([1, 2, 2, 1])
        st.markdown("#")
        st.markdown("#")

        with coli2:
            job_offer_text = None
            job_offer_pdf = None
            with st.expander("Votre CURRICULUM", expanded=True):
                st.markdown(
                    "<div class='uploader'>Chargez votre <b>curriculum</b> au format PDF :</div>",
                    unsafe_allow_html=True,
                )
                resume_upload = st.file_uploader("", type=["pdf"], key="resume_upload")

        with coli3:
            mode_entree = st.selectbox(
                "Comment souhaitez-vous fournir l'offre d'emploi ?",
                [
                    "Sélectionner",
                    "Entrer le texte descriptif",
                    "Charger une offre en PDF",
                ],
            )
            if mode_entree == "Entrer le texte descriptif":
                st.markdown(
                    "<div class='textarea'>Copiez le <b>descriptif</b> de l'offre :</div>",
                    unsafe_allow_html=True,
                )
                job_offer_text = st.text_area("", height=300, key="job_offer_text")
            elif mode_entree == "Charger une offre en PDF":
                job_offer_pdf = st.file_uploader("", type=["pdf"], key="job_offer_pdf")

    # Activation du bouton de soumission selon les conditions spécifiées
    condition_soumission = resume_upload and (
        job_offer_text and len(job_offer_text) >= 100 or job_offer_pdf is not None
    )
    with col2:
        if condition_soumission:
            A, B, C, D, E, AA, BB, CC, DD = st.columns(9)
            with E:
                if st.button("Lancer l'analyse "):
                    # st.session_state["testing_stage"] = 1
                    # st.session_state["cv_holder"] = resume_upload
                    # # Stocker le texte de l'offre d'emploi ou le chemin du fichier PDF selon le cas
                    # if job_offer_text:
                    #     st.session_state["job_holder"] = job_offer_text
                    # else:
                    #     st.session_state["job_holder"] = job_offer_pdf
                    # st.rerun()
                    with col2:
                        st.markdown(
                            """
                                <div style="background-color: #75da7e; color: black; padding: 10px; border-radius: 5px; font-family: Arial, sans-serif;font-size:1.2vw;text-align:center;">
                                    <strong>Attention :</strong> Cette section est disponible uniquement en démonstration. Pour plus d'informations, veuillez contacter notre équipe via la section <a href="#about-nest" style="color: #3498db;">'About Nest'</a>.
                                </div>
                                """,
                            unsafe_allow_html=True,
                        )

        else:
            st.markdown(
                "<h3 style='text-align: center;'>Les deux éléments sont <span style='color:rgb(113,224,203);'> necessaires </span> lancer l'analyse </h3>",
                unsafe_allow_html=True,
            )


def test_cv_page():
    if "resume_data" not in st.session_state:
        st.session_state["resume_data"] = None
    if "job_offer_data" not in st.session_state:
        st.session_state["job_offer_data"] = None

    if "loading" not in st.session_state:
        st.session_state["loading"] = False

    if "testing_stage" not in st.session_state:
        st.session_state["testing_stage"] = 0

    if "cv_holder" not in st.session_state:
        st.session_state["cv_holder"] = None
    if "job_holder" not in st.session_state:
        st.session_state["job_holder"] = None

    if "matching_stat" not in st.session_state:
        st.session_state["matching_stat"] = None

    if st.session_state["testing_stage"] == 0:
        initial_test()
    if st.session_state["testing_stage"] == 1:
        if st.session_state["matching_stat"] == None:

            with st.spinner("**Analyse de votre profil**"):
                parsed_resume = parsing_joboffer(st.session_state["cv_holder"])
            with st.spinner("**Analyse de l'offre**"):
                parsed_job_offer = parsing_joboffer(st.session_state["job_holder"])
            col1, col2, COL = st.columns(3)
            with col2:
                with st.spinner("**Simulation d'entretien**"):
                    matched_stuff = matching_offer_with_candidat(
                        parsed_resume, parsed_job_offer
                    )

                # Stocker les résultats dans st.session_state
                st.session_state["resume_data"] = parsed_resume
                st.session_state["job_offer_data"] = parsed_job_offer
                st.session_state["matching_stat"] = matched_stuff
                st.session_state["testing_stage"] = 2
                st.rerun()

    if st.session_state["testing_stage"] == 2:
        visionnage_test()

        # Exemple d'utilisation de match_visualisation si applicable
        # match_visualisation(...)


def calculer_note_globale():
    parsed_data = st.session_state["resume_data"]
    elements_presents = sum(
        [
            bool(", ".join(parsed_data.get("numero", []))),
            bool(", ".join(parsed_data.get("mail", []))),
            bool(", ".join(parsed_data.get("nom_prenom", []))),
            bool(", ".join(parsed_data.get("location", []))),
        ]
    )
    pourcentage_identification = int((elements_presents / 4) * 100)

    nombre_soft_skills = len(parsed_data.get("soft_skills", []))
    note_soft_skills = min(nombre_soft_skills / 4, 1) * 100

    diplomes = parsed_data.get("diplome", [])
    langues = parsed_data.get("langue", [])
    score_diplome = (50 if diplomes else 0) + (50 if langues else 0)  # Ajustement ici

    nombre_taches = len(parsed_data.get("tache", []))
    score_taches = min(nombre_taches / 8, 1) * 6

    nombre_hard_skills = len(parsed_data.get("hard_skills", []))
    score_hard_skills = min(nombre_hard_skills / 4, 1) * 6
    score_total_taches_et_skills = (score_taches + score_hard_skills) / 12 * 100

    # Poids attribués à chaque score pour le calcul de la note globale
    poids = [0.3, 0.1, 0.3, 0.3]

    # Calcul de la note globale en utilisant les poids définis
    note_globale = (
        pourcentage_identification * poids[0]
        + note_soft_skills * poids[1]
        + score_total_taches_et_skills * poids[2]
        + score_diplome * poids[3]
    )

    return note_globale


def calculer_note_globale_job():
    parsed_data = st.session_state["job_offer_data"]
    elements_presents = sum([bool(", ".join(parsed_data.get("location", [])))])
    pourcentage_identification = int((elements_presents / 1) * 100)

    nombre_soft_skills = len(parsed_data.get("soft_skills", []))
    note_soft_skills = min(nombre_soft_skills / 4, 1) * 100

    diplomes = parsed_data.get("diplome", [])
    langues = parsed_data.get("langue", [])
    score_diplome = (50 if diplomes else 0) + (50 if langues else 0)  # Ajustement ici

    nombre_taches = len(parsed_data.get("tache", []))
    score_taches = min(nombre_taches / 8, 1) * 6

    nombre_hard_skills = len(parsed_data.get("hard_skills", []))
    score_hard_skills = min(nombre_hard_skills / 4, 1) * 6
    score_total_taches_et_skills = (score_taches + score_hard_skills) / 12 * 100

    # Poids attribués à chaque score pour le calcul de la note globale
    poids = [0.1, 0.3, 0.3, 0.3]

    # Calcul de la note globale en utilisant les poids définis
    note_globale = (
        pourcentage_identification * poids[0]
        + note_soft_skills * poids[1]
        + score_total_taches_et_skills * poids[2]
        + score_diplome * poids[3]
    )

    return note_globale


def afficher_qualites_cv():
    st.markdown("#")
    if st.button("Retour à la sélection"):
        # Mettre à jour "testing_stage" dans st.session_state
        st.session_state["testing_stage"] = 0
        st.session_state["matching_stat"] = None
        st.rerun()
    st.markdown("#")
    st.markdown("#")
    st.markdown("#")
    parsed_data = st.session_state["resume_data"]
    # Extraction des informations pertinentes
    numero = ", ".join(parsed_data.get("numero", []))
    mail = ", ".join(parsed_data.get("mail", []))
    nom_prenom = ", ".join(parsed_data.get("nom_prenom", []))
    adresse = ", ".join(parsed_data.get("location", []))
    title_html = """
            <div style=" padding: 10px; border-radius: 10px;">
                <h3 style="color:  #34495E; text-align: center; font-weight: bold;">Votre profil</h3>
            </div>
            """

    st.markdown(title_html, unsafe_allow_html=True)
    afficher_grade_cv(calculer_note_globale(), 0)
    with st.expander("**Décrouvez en detail l'evaluation de votre CV**"):
        # Calculer le pourcentage d'identification
        elements_presents = sum(
            [bool(numero), bool(mail), bool(nom_prenom), bool(adresse)]
        )
        pourcentage_identification = int((elements_presents / 4) * 100)

        # Définition des couleurs
        couleur_identifiable = "#006400"  # Vert foncé
        couleur_non_identifiable = "#8B0000"  # Rouge foncé

        html_content = f"""
        <div style="text-align: center;color:#1B4F72;">
            <h2 style="color: rgb(113, 224, 203);">IDENTITÉ</h2>
            <div style="background-color: #eee; border-radius: 10px; width: 80%; height: 24px; margin: 10px auto; position: relative;">
                <div style="background-color: rgb(113, 224, 203); width: {pourcentage_identification}%; height: 100%; border-radius: 10px; position: absolute;">
                    <span style="color: #fff; font-weight: bold; position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">{pourcentage_identification}%</span>
                </div>
            </div> 
            <p><strong>Votre nom </strong> <span style="font-weight:bold; color: {couleur_identifiable if nom_prenom else couleur_non_identifiable};">{'IDENTIFIABLE ' if nom_prenom else 'NON IDENTIFIABLE ❌'}</span></p>
            <p><strong>Votre region </strong> <span style="font-weight:bold; color: {couleur_identifiable if adresse else couleur_non_identifiable};">{'IDENTIFIABLE ' if adresse else 'NON IDENTIFIABLE ❌'}</span></p>
            <p><strong>Contact par téléphone </strong> <span style="font-weight:bold; color: {couleur_identifiable if numero else couleur_non_identifiable};">{'IDENTIFIABLE ' if numero else 'NON IDENTIFIABLE ❌'}</span></p>
            <p><strong>Contact par mail </strong> <span style="font-weight:bold; color: {couleur_identifiable if mail else couleur_non_identifiable};">{'IDENTIFIABLE ' if mail else 'NON IDENTIFIABLE ❌'}</span></p>
        
        </div>
        """
        st.markdown("#")

        st.markdown(html_content, unsafe_allow_html=True)
        ##### SOFT SKILLS
        soft_skills = parsed_data.get("soft_skills", [])

        # Calcul de la note basée sur le nombre de soft skills
        nombre_soft_skills = len(soft_skills)
        note_soft_skills = min(nombre_soft_skills / 4, 1) * 100
        couleur_score = (
            "#ff0000" if note_soft_skills < 50 else "#fff"
        )  # Rouge pour les scores < 50%, blanc sinon
        commentaire = (
            "<p style='color: #006400; font-size: 14px;font-weight:bolder;'>Vous donnez suffisaments d'élements au recruteur pour vous identifié </p>"
            if nombre_soft_skills >= 4
            else "<p style='color: red; font-size: 14px;font-weight:bolder;'>Décrivez au moins 4 aspects de votre personnalité.</p>"
        )

        # Construction du contenu HTML pour la section Qui vous êtes
        html_content = f"""
        <div style="text-align: center; margin-top: 20px;color:#1B4F72;">
            <h2 style="color: rgb(113, 224, 203);">SOFT SKILLS</h2>
            <div style="background-color: #eee; border-radius: 10px; width: 80%; height: 24px; margin: 10px auto; position: relative;">
                <div style="background-color: rgb(113, 224, 203); width: {note_soft_skills}%; height: 100%; border-radius: 10px; position: absolute;">
                    <span style="color: {couleur_score}; font-weight: bold; position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">{int(note_soft_skills)}%</span>
                </div>
            </div>
            {commentaire}
        </div>
        """

        # Affichage du contenu HTML dans Streamlit
        st.markdown(html_content, unsafe_allow_html=True)
        ######## ACADEMIQUES
        diplomes = parsed_data.get("diplome", [])
        langues = parsed_data.get("langue", [])

        # Calculer le score d'identifiabilité (50% pour chaque section présente)
        score_diplome = (50 if diplomes else 0) + (50 if langues else 0)
        couleur_barre = (
            "#006400" if score_diplome > 50 else "#8B0000"
        )  # Vert foncé si score > 50, rouge foncé sinon

        langue_message = (
            "IDENTIFIABLE"
            if langues
            else "Aucune langue indiquée. Mentionnez au moins votre niveau d'anglais."
        )
        couleur_langue = (
            "#006400" if langues else "#8B0000"
        )  # Vert foncé si présente, rouge foncé sinon

        # Construction du contenu HTML pour la section Académique et Langues avec barre de progression
        html_content = f"""
        <div style="text-align: center; margin-top: 20px;color:#1B4F72;">
            <h2 style="color: rgb(113, 224, 203);">ACADEMIE</h2>
            <div style="background-color: #eee; border-radius: 10px; width: 80%; height: 24px; margin: 10px auto; position: relative;">
                <div style="background-color: rgb(113, 224, 203); width: {score_diplome}%; height: 100%; border-radius: 10px; position: absolute;">
                    <span style="color: #fff; font-weight: bold; position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">{score_diplome}%</span>
                </div>
            </div>
            <p><strong>Niveau académique </strong> <span style="font-weight:bold; color: {couleur_barre};">{'IDENTIFIABLE' if diplomes else 'NON IDENTIFIABLE ❌'}</span></p>
            <p><strong>Niveau de Langue</strong> <span style="font-weight:bold; color: {couleur_langue};">{langue_message}</span></p>
        </div>
        """

        # Affichage du contenu HTML dans Streamlit
        st.markdown(html_content, unsafe_allow_html=True)
        ####### stack

        taches = parsed_data.get("tache", [])
        hard_skills = parsed_data.get("hard_skills", [])
        job_type = parsed_data.get("job_type", "votre secteur")

        # Calcul du score d'identifiabilité
        score_taches = (
            min(len(taches) / 8, 1) * 6
        )  # 6 points possibles, besoin de 8 tâches pour le max
        score_hard_skills = (
            min(len(hard_skills) / 4, 1) * 6
        )  # 6 points possibles, besoin de 4 compétences pour le max
        score_total = score_taches + score_hard_skills
        score_totale = score_total / 12 * 100
        # Détermination des couleurs et des messages basés sur l'identifiabilité
        couleur_taches = (
            "#006400" if len(taches) >= 8 else "#8B0000"
        )  # Vert foncé si >= 8 tâches, rouge foncé sinon
        couleur_hard_skills = (
            "#006400" if len(hard_skills) >= 4 else "#8B0000"
        )  # Vert foncé si >= 4 compétences, rouge foncé sinon

        commentaire_hard_skills = (
            ""
            if len(hard_skills) >= 4
            else "<p style='color: red; font-size: 14px;text-align: center;'>Spécifiez au moins 4 outil techniques.</p>"
        )
        commentaire_taches = (
            ""
            if len(taches) >= 8
            else f"<p style='color: red; font-size: 14px;font-weight:bold;text-align: center;'>Notez le maximum de taches réalisées lors de vos experiences en relation avec le secteur: {job_type}.</p>"
        )

        # Construction du contenu HTML pour les sections Tâches et Hard Skills avec barre de progression
        html_content = f"""
        <div style="text-align: center; margin-top: 20px;color:#1B4F72;">
            <h2 style="color: rgb(113, 224, 203);">COMPETENCES</h2>
            <div style="background-color: #eee; border-radius: 10px; width: 80%; height: 24px; margin: 10px auto; position: relative;">
                <div style="background-color: rgb(113, 224, 203); width: {score_total / 12 * 100}%; height: 100%; border-radius: 10px; position: absolute;">
                    <span style="color: #fff; font-weight: bold; position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">{int(score_total / 12 * 100)}%</span>
                </div>
            </div>
            <p><strong>Tâches réalisées </strong> <span style="font-weight:bold; color: {couleur_taches};">{'IDENTIFIABLE' if len(taches) >= 8 else 'NON IDENTIFIABLE ❌'}</span></p>
            <p><strong>Outils techniques </strong> <span style="font-weight:bold; color: {couleur_hard_skills};">{'IDENTIFIABLE' if len(hard_skills) >= 4 else 'NON IDENTIFIABLE ❌'}</span></p>
        </div>
        """

        # Affichage du contenu HTML dans Streamlit
        st.markdown(html_content, unsafe_allow_html=True)

        st.markdown(commentaire_hard_skills, unsafe_allow_html=True)
        st.markdown(commentaire_taches, unsafe_allow_html=True)
    st.markdown("#")
    title_html = """
            <div style=" padding: 10px; border-radius: 10px;">
                <h3 style="color:  #34495E; text-align: center; font-weight: bold;">L'opportunitée</h4>
            </div>
            """

    st.markdown(title_html, unsafe_allow_html=True)
    afficher_grade_cv(calculer_note_globale_job(), 1)
    with st.expander("**Details sur le detail de l'offre d'emploi**"):
        parsed_data = st.session_state["job_offer_data"]
        # Extraction des informations pertinentes
        numero = ", ".join(parsed_data.get("numero", []))
        mail = ", ".join(parsed_data.get("mail", []))
        nom_prenom = ", ".join(parsed_data.get("nom_prenom", []))
        adresse = ", ".join(parsed_data.get("location", []))

        # Calculer le pourcentage d'identification
        elements_presents = sum([bool(adresse)])
        pourcentage_identification = int((elements_presents / 1) * 100)

        # Définition des couleurs
        couleur_identifiable = "#006400"  # Vert foncé
        couleur_non_identifiable = "#8B0000"  # Rouge foncé

        html_content = f"""
        <div style="text-align: center;color:#1B4F72;">
            <h2 style="color: rgb(113, 224, 203);">IDENTITÉ</h2>
            <div style="background-color: #eee; border-radius: 10px; width: 80%; height: 24px; margin: 10px auto; position: relative;">
                <div style="background-color: rgb(113, 224, 203); width: {pourcentage_identification}%; height: 100%; border-radius: 10px; position: absolute;">
                    <span style="color: #fff; font-weight: bold; position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">{pourcentage_identification}%</span>
                </div>
            </div> 
            <p><strong>La region de l'opportunitée</strong> <span style="font-weight:bold; color: {couleur_identifiable if adresse else couleur_non_identifiable};">{'IDENTIFIABLE ' if adresse else 'NON IDENTIFIABLE ❌'}</span></p>
            
        </div>
        """
        st.markdown("#")

        st.markdown(html_content, unsafe_allow_html=True)
        ##### SOFT SKILLS
        soft_skills = parsed_data.get("soft_skills", [])

        # Calcul de la note basée sur le nombre de soft skills
        nombre_soft_skills = len(soft_skills)
        note_soft_skills = min(nombre_soft_skills / 4, 1) * 100
        couleur_score = (
            "#ff0000" if note_soft_skills < 50 else "#fff"
        )  # Rouge pour les scores < 50%, blanc sinon
        commentaire = (
            "<p style='color: #006400; font-size: 14px;font-weight:bolder;'>Suffisaments d'élements pour comprendre les attentes de l'équipe </p>"
            if nombre_soft_skills >= 4
            else "<p style='color: red; font-size: 14px;font-weight:bolder;'> Manque d'élement afin d'etre proche des attentes de l'équipe</p>"
        )

        # Construction du contenu HTML pour la section Qui vous êtes
        html_content = f"""
        <div style="text-align: center; margin-top: 20px;color:#1B4F72;">
            <h2 style="color: rgb(113, 224, 203);">SOFT SKILLS</h2>
            <div style="background-color: #eee; border-radius: 10px; width: 80%; height: 24px; margin: 10px auto; position: relative;">
                <div style="background-color: rgb(113, 224, 203); width: {note_soft_skills}%; height: 100%; border-radius: 10px; position: absolute;">
                    <span style="color: {couleur_score}; font-weight: bold; position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">{int(note_soft_skills)}%</span>
                </div>
            </div>
            {commentaire}
        </div>
        """

        # Affichage du contenu HTML dans Streamlit
        st.markdown(html_content, unsafe_allow_html=True)
        ######## ACADEMIQUES
        diplomes = parsed_data.get("diplome", [])
        langues = parsed_data.get("langue", [])

        # Calculer le score d'identifiabilité (50% pour chaque section présente)
        score_diplome = (50 if diplomes else 0) + (50 if langues else 0)
        couleur_barre = (
            "#006400" if score_diplome > 50 else "#8B0000"
        )  # Vert foncé si score > 50, rouge foncé sinon

        langue_message = (
            "IDENTIFIABLE"
            if langues
            else "Aucune langue indiquée dans l'offre.  L'Anglais est une langue habituelement recherchée."
        )
        couleur_langue = (
            "#006400" if langues else "#8B0000"
        )  # Vert foncé si présente, rouge foncé sinon

        # Construction du contenu HTML pour la section Académique et Langues avec barre de progression
        html_content = f"""
        <div style="text-align: center; margin-top: 20px;color:#1B4F72;">
            <h2 style="color: rgb(113, 224, 203);">ACADEMIE</h2>
            <div style="background-color: #eee; border-radius: 10px; width: 80%; height: 24px; margin: 10px auto; position: relative;">
                <div style="background-color: rgb(113, 224, 203); width: {score_diplome}%; height: 100%; border-radius: 10px; position: absolute;">
                    <span style="color: #fff; font-weight: bold; position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">{score_diplome}%</span>
                </div>
            </div>
            <p><strong>Niveau académique </strong> <span style="font-weight:bold; color: {couleur_barre};">{'IDENTIFIABLE' if diplomes else 'NON IDENTIFIABLE ❌'}</span></p>
            <p><strong>Niveau de Langue</strong> <span style="font-weight:bold; color: {couleur_langue};">{langue_message}</span></p>
        </div>
        """

        # Affichage du contenu HTML dans Streamlit
        st.markdown(html_content, unsafe_allow_html=True)
        ####### stack

        taches = parsed_data.get("tache", [])
        hard_skills = parsed_data.get("hard_skills", [])
        job_type = parsed_data.get("job_type", "votre secteur")

        # Calcul du score d'identifiabilité
        score_taches = (
            min(len(taches) / 8, 1) * 6
        )  # 6 points possibles, besoin de 8 tâches pour le max
        score_hard_skills = (
            min(len(hard_skills) / 4, 1) * 6
        )  # 6 points possibles, besoin de 4 compétences pour le max
        score_total = score_taches + score_hard_skills
        score_totale = score_total / 12 * 100
        # Détermination des couleurs et des messages basés sur l'identifiabilité
        couleur_taches = (
            "#006400" if len(taches) >= 8 else "#8B0000"
        )  # Vert foncé si >= 8 tâches, rouge foncé sinon
        couleur_hard_skills = (
            "#006400" if len(hard_skills) >= 4 else "#8B0000"
        )  # Vert foncé si >= 4 compétences, rouge foncé sinon

        commentaire_hard_skills = (
            ""
            if len(hard_skills) >= 4
            else "<p style='color: red; font-size: 14px;text-align: center;'></p>"
        )
        commentaire_taches = (
            ""
            if len(taches) >= 8
            else f"<p style='color: red; font-size: 14px;font-weight:bold;text-align: center;'> l'offre manque un peu d'élement</p>"
        )

        # Construction du contenu HTML pour les sections Tâches et Hard Skills avec barre de progression
        html_content = f"""
        <div style="text-align: center; margin-top: 20px;color:#1B4F72;">
            <h2 style="color: rgb(113, 224, 203);">COMPETENCES</h2>
            <div style="background-color: #eee; border-radius: 10px; width: 80%; height: 24px; margin: 10px auto; position: relative;">
                <div style="background-color: rgb(113, 224, 203); width: {score_total / 12 * 100}%; height: 100%; border-radius: 10px; position: absolute;">
                    <span style="color: #fff; font-weight: bold; position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">{int(score_total / 12 * 100)}%</span>
                </div>
            </div>
            <p><strong>Tâches réalisées </strong> <span style="font-weight:bold; color: {couleur_taches};">{'IDENTIFIABLE' if len(taches) >= 8 else 'NON IDENTIFIABLE ❌'}</span></p>
            <p><strong>Outils techniques </strong> <span style="font-weight:bold; color: {couleur_hard_skills};">{'IDENTIFIABLE' if len(hard_skills) >= 4 else 'NON IDENTIFIABLE ❌'}</span></p>
        </div>
        """

        # Affichage du contenu HTML dans Streamlit
        st.markdown(html_content, unsafe_allow_html=True)

        st.markdown(commentaire_hard_skills, unsafe_allow_html=True)
        st.markdown(commentaire_taches, unsafe_allow_html=True)


def afficher_notation_etoiles(note_sskill):
    if note_sskill < 0:
        html_notation = "<span>❓</span>"
    else:
        etoiles_entieres = note_sskill // 20
        demi_etoile = (note_sskill % 20) // 10
        etoiles_noires = 5 - etoiles_entieres - demi_etoile

        html_notation = (
            "<span style='color: gold;'>"
            + "⭐" * int(etoiles_entieres)
            + "</span>"
            + ("<span style='color: gold;'>⭐½</span>" if demi_etoile else "")
            + "<span style='color: black;'>"
            + "⚫" * int(etoiles_noires)
            + "</span>"
        )

    return html_notation


def afficher_score_card(
    match_score,
    note_sskill,
    note_stack,
    note_tache,
    note_langue,
    note_diplome,
    note_exp,
):
    # Convert negative score to 0 for display

    # Round the match score to two decimal places
    match_score = round(match_score, 0)

    # HTML and CSS
    html_content = f"""
    <style>

        
        .score-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            width: 300px;
            height: 300px;
            border-radius: 50%;
            background: #6200EE;
            color: white;
            margin: auto;
            padding: 20px;
            position: relative;
        }}

        .score-container::after {{
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            width: 0;
            height: 0;
            font-size:3vw;
            border-left: 20px solid transparent;
            border-right: 20px solid transparent;
            border-top: 20px solid #6200EE;
            transform: translateX(-50%);
        }}

        .score {{
            font-size: 2.5vw;
            font-weight: bold;
            line-height: 1em;
        }}

        .score-label {{
            font-size: 1.5vw;
            margin-bottom: 5px;
            text-align:center;
            font-weight: bold;
            text-align:center;
        }}
        
        .summary {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            padding: 15px;
            margin: 2em auto 0 auto;
            width: 90%;
            max-width: 400px;
            
            
        }}
         

        .summary-item {{
            border-bottom: 1px solid #eee;
            padding: 10px 0;
            display: flex;
            font-weight: bold;
            justify-content: space-between;
            font-size: 0.9em;
        }}

        .summary-item2 {{
            border-bottom: 1px solid #eee;
            padding: 10px 0;
            color:rgb(113,224,203);
            display: flex;
            font-weight: bold;
            background: white;
            justify-content: space-between;
            font-size: 0.9em;
        }}

        .summary-item:last-child {{
            border-bottom: none;
        }}
    </style>

    <div class="score-container">
        <div class="score">{int(match_score)}%</div>
        <div class="score-label"> de chance de passer le test IA</div>
    </div>
    
    <div class="summary">
        <div class="summary-item">
            <span>Personnalité </span>
            <span>{afficher_notation_etoiles(note_sskill)}</span>
        </div>
        <div class="summary-item2">
            <span>Outils </span>
            <span>{afficher_notation_etoiles(note_stack)}</span>
        </div>
        <div class="summary-item">
            <span>Taches </span>
            <span>{afficher_notation_etoiles(note_tache)}</span>
        </div>
        <div class="summary-item2">
            <span>Niveau de langue :</span>
            <span>{afficher_notation_etoiles(note_langue)}</span>
        </div>
        <div class="summary-item">
            <span>Formation </span>
            <span>{afficher_notation_etoiles(note_diplome)}</span>
        </div>
        <div class="summary-item2">
            <span>Experience </span>
            <span>{afficher_notation_etoiles(note_exp)}</span>
        </div>
    </div>
    """

    st.markdown(html_content, unsafe_allow_html=True)


def afficher_score_card(nom_metier, description_metier):
    # HTML and CSS
    html_content = f"""
    <style>
        .score-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            flex-direction: column;
            width: 300px;
            height: 300px;
            border-radius: 50%;
            background: #6200EE;
            color: white;
            margin: auto;
            padding: 20px;
            position: relative;
            font-family: 'Arial', sans-serif;
        }}

        .score-container::after {{
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            width: 0;
            height: 0;
            border-left: 20px solid transparent;
            border-right: 20px solid transparent;
            border-top: 20px solid #6200EE;
            transform: translateX(-50%);
        }}

        .job-title {{
            font-size: 1.5vw;
            font-weight: bold;
            text-align: center;
        }}

        .job-description {{
            font-size: 0.8vw;
            text-align: center;
            margin-top: 10px;
        }}
    </style>

    <div class="score-container">
        <div class="job-title">{nom_metier}</div>
        <div class="job-description">{description_metier}</div>
    </div>
    """

    # Utiliser cette ligne pour afficher le HTML dans un environnement de développement web ou un notebook
    st.markdown(html_content, unsafe_allow_html=True)


def afficher_conseil_HS(stacks, note_stack):
    # Afficher un message général basé sur la note de stack
    if note_stack < 0:
        st.info("Pas d'outils requis.")
        return  # Sortir de la fonction car aucune autre vérification n'est nécessaire
    elif note_stack < 30:
        st.error("**Votre maîtrise des outils est plus qu'insuffisante.**")
    elif note_stack < 60:
        st.warning(
            "Votre maîtrise des outils est moyenne, des améliorations sont possibles."
        )
    else:
        st.success("**Votre maîtrise des outils est excellente.**")

    # Extraire les compétences recherchées et leur correspondance RGmatch
    skills_recherche = stacks["skillrecherche"]
    skills_dispo = stacks["skilldispo"]
    rgmatch = stacks["RGmatch"]

    # Identifier les compétences manquantes basées sur un RGmatch de 0
    missing_skills = [
        skill for skill, match in zip(skills_recherche, rgmatch) if match == 0
    ]

    # Limiter à 5 compétences manquantes
    missing_skills_limited = missing_skills[:5]
    missing_skills_str = " -- ".join(missing_skills_limited)

    if missing_skills_limited:
        additional_info = "..." if len(missing_skills) > 5 else ""
        st.warning(
            f"Maitrisez vous ces outils? : **{missing_skills_str}**{additional_info}"
        )
    else:
        st.success("Tous les outils nécessaires sont présents dans votre stack.")


def evaluate_job_fit(talent_type, job_type):
    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

    # In the first column, display the title "Job Category"
    with col1:
        title_html = """
            <div style="background-color: #302838; padding: 10px; border-radius: 10px;">
                <h2 style="color: white; text-align: center; font-weight: bold;">DOMAINE</h2>
            </div>
            """

        st.markdown(title_html, unsafe_allow_html=True)
    with col2:
        if talent_type == job_type:
            # The talent type matches the job type
            st.success(
                f"Votre profil correspond à celui recherché à savoir **{clarifier_nom(job_type)}**."
            )
        else:
            # The talent type does not match the job type
            st.error(
                f"L'offre d'emploi decris un specialiste : **{clarifier_nom(job_type)}**, Mais votre est profil: **{clarifier_nom(talent_type)}**.",
            )


######


def evaluate_stack_fit(match_result):
    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

    # In the first column, display the title "Job Category"
    with col1:
        title_html = """
            <div style="background-color: #302838; padding: 10px; border-radius: 10px;">
                <h2 style="color: white; text-align: center; font-weight: bold;">STACK</h2>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)

    # In the second column, display the graph
    with col2:
        if match_result["NOTE_STACK"] < 0:
            st.info("**Aucune mention d'outils specifique **")

        else:
            st.plotly_chart(
                affichage_scatter(match_result["STACKS"]),
                use_container_width=True,
            )

    # In the third column, display the comments or advice
    with col3:
        title_html = """
            <div style=" padding: 10px; border-radius: 10px;">
                <h4 style="color:  #34495E; text-align: center; font-weight: bold;">Commentaire</h4>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)

        st.markdown("#")
        st.markdown("#")
        afficher_conseil_HS(match_result["STACKS"], match_result["NOTE_STACK"])


def niveau_diplome(diplome_job, diplome_talent, note):
    # Create three columns with specified width ratios
    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

    # Display the title "Language Level" in the first column
    with col1:
        title_html = """
            <div style="background-color: #302838; padding: 10px; border-radius: 10px;">
                <h2 style="color: white; text-align: center; font-weight: bold;">FORMATION</h2>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)

    # Display a simple success graph in the second column
    with col2:

        if note < 0:
            st.info("**Aucune specification sur le niveau de formation**")
        elif note > 70:
            st.success("**Vous etes à niveau**")
        else:
            st.error("**Vous ne semblez pas suffisament qualifié.e** ")

    # Display evaluation comments in the third column
    with col3:
        title_html = """
            <div style=" padding: 10px; border-radius: 10px;">
                <h4 style="color:  #34495E; text-align: center; font-weight: bold;">Commentaire</h4>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)

        st.markdown("#")

        if note == 100:
            st.success(
                f"Niveau requis: **{diplome_job}**, Votre niveau: **{diplome_talent}**."
            )

        elif note < 0:
            pass

        else:
            st.error(
                f"Niveau scolaire requis:**{diplome_job}**, Votre niveau: **{diplome_talent}**. Considérer l'importance du diplome sur ce poste."
            )


def evaluate_language_detail(detail_langue, note):
    echelle_CECR_inverse = {
        6: "niveau C2",
        5: "niveau C1",
        4: "niveau B2",
        3: "niveau B1",
        2: "niveau A2",
        1: "niveau A1",
    }

    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

    with col1:
        title_html = """
            <div style="background-color: #302838; padding: 10px; border-radius: 10px;">
                <h2 style="color: white; text-align: center; font-weight: bold;">LANGUE</h2>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)
    with col2:
        if note < 0:
            st.info("**Aucune specification sur le niveau de langue attendu**")
        else:
            # Placeholder for graphical representation
            afficher_barre_progression_custom(note)

    with col3:
        title_html = """
            <div style=" padding: 10px; border-radius: 10px;">
                <h4 style="color:  #34495E; text-align: center; font-weight: bold;font-size:1.4vw;">Avis Nest</h4>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)

        st.markdown("#")
        for idx in detail_langue["langue"]:
            langue = detail_langue["langue"][idx]
            desired_level = detail_langue["note"][idx]
            candidate_level = detail_langue["note_prospect"][idx]
            notation = detail_langue["notation"][idx]

            niveau_cecr_desired = echelle_CECR_inverse.get(
                desired_level, "Non mentioné"
            )
            niveau_cecr_candidate = echelle_CECR_inverse.get(
                candidate_level, "Non mentioné"
            )

            if notation == 1:
                st.success(
                    f"**{langue.capitalize()}**: Votre niveau pour la langue ({niveau_cecr_candidate}) match la necessité pour ce poste ({niveau_cecr_desired})."
                )
            else:
                st.warning(
                    f"**{langue.capitalize()}**: Votre niveau pour la langue({niveau_cecr_candidate}) est en dessous du niveau requis :({niveau_cecr_desired})."
                )


def evaluate_taches(taches, note_globale):
    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

    with col1:
        title_html = """
            <div style="background-color: #302838; padding: 10px; border-radius: 10px;">
                <h2 style="color: white; text-align: center; font-weight: bold;">Taches</h2>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)
    # Barre de remplissage pour la note globale
    with col2:
        if note_globale < 0:
            st.markdown(
                "Aucune compétence comportementale n'est demandée pour ce poste."
            )
        else:
            st.markdown("### ")
            afficher_barre_progression_custom(int(note_globale))

    with col3:
        title_html = """
            <div style=" padding: 10px; border-radius: 10px;">
                <h4 style="color:  #34495E; text-align: center; font-weight: bold;font-size:1.4vw;">Avis Nest</h4>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)

        for i, skill in enumerate(taches["skillrecherche"][:5]):
            match_ajuste = taches["match_ajuste"][i]
            if match_ajuste > 0.6:
                st.markdown(
                    f":bleu[Besoin: **{skill}:**] ➔| Vous: :green[**{taches['skilldispo'][i]}**]"
                )
            elif match_ajuste > 0.3:
                st.markdown(
                    f":bleu[Besoin: **{skill}:**] ➔| Vous: :orange[**{taches['skilldispo'][i]}**]"
                )
            else:
                st.markdown(f":red[- **{skill.capitalize()}:** Manquant]")
        if len(taches["skillrecherche"]) > 5:
            st.markdown("...")


def afficher_grade_cv(score_global, index):
    # Déterminer le grade et la couleur basée sur le score global
    if score_global >= 75:
        grade, couleur = "A", "#006400"
        conseil = (
            "La comprehension de votre profil est plus que satisfaisant"
            if index == 0
            else "Votre offre est bien structurée pour être I.A compatible"
        )
    elif score_global >= 50:
        grade, couleur = "B", "#FFD700"
        conseil = (
            "Des informations importantes peuvent manquer et donc vous faire rater le passage I.A"
            if index == 0
            else "Votre offre peut manquer d'informations importantes pour un filtrage I.A optimal"
        )
    else:
        grade, couleur = "C", "#8B0000"
        conseil = (
            "Vous allez forcément rater des offres car le format choisi ne permet pas une lisibilité optimale"
            if index == 0
            else "Le format de l'offre risque de faire rater des candidatures potentielles à cause d'une lisibilité non optimale"
        )

    # Construction du contenu HTML pour afficher le grade et le conseil adapté
    html_content = f"""
    <div style="text-align: center; margin-top: 20px;">
        <div style="border: 5px solid {couleur}; display: inline-block; padding: 20px; border-radius: 10px; background-color: #fff;">
            <h1 style="color: {couleur}; font-size: 3vw;">{grade}</h1>
        </div>
        <p style="font-size: 1.0vw;font-weight:bolder; margin-top: 2px;text-align:center;color:{couleur};">{conseil}</p>
    </div>
    """

    # Affichage du contenu HTML dans Streamlit
    st.markdown(html_content, unsafe_allow_html=True)


def evaluate_soft_skills(detail_note_sskill, overall_note):

    col1, col2, col3 = st.columns([0.2, 0.5, 0.3])

    with col1:
        title_html = """
            <div style="background-color: #302838; padding: 10px; border-radius: 10px;">
                <h2 style="color: white; text-align: center; font-weight: bold;">SOFT SKILLS</h2>
            </div>
            """
        st.markdown(title_html, unsafe_allow_html=True)

    with col2:
        if overall_note < 0:
            st.markdown(
                "Aucune compétence comportementale n'est demandée pour ce poste."
            )
        else:
            # Affichage d'une barre de progression représentant la note globale des soft skills
            # La note est supposée être sur 100, ajustez si nécessaire
            afficher_barre_progression_custom(overall_note)

    with col3:
        title_html = """
            <div style=" padding: 10px; border-radius: 10px;">
                <h4 style="color:  #34495E; text-align: center; font-weight: bold;font-size:1.4vw;">Avis Nest</h4>
            </div>
            """

        st.markdown(title_html, unsafe_allow_html=True)
        if overall_note < 0:
            st.markdown(
                "Aucune compétence comportementale n'est demandée pour ce poste."
            )

        else:
            cpt = 0
            for categorie, indice, match in zip(
                detail_note_sskill["categorie"],
                detail_note_sskill["Indices"],
                detail_note_sskill["match"],
            ):

                if match == 1:
                    st.markdown(
                        f":green[- **{categorie}** : {indice if indice else 'Pas de détail spécifique.'}]"
                    )
                    cpt = cpt + 1
                else:
                    st.markdown(f":red[- **{categorie}** : Nécessaire pour ce poste]")
                    cpt = cpt + 1
                if cpt == 4:
                    break


def afficher_barre_progression_custom(note):
    # Définir la couleur du texte en fonction de la note
    couleur_texte = "#fff" if note >= 50 else "#ff0000"  # Blanc si >= 50%, rouge sinon

    # HTML pour la barre de progression
    html_content = f"""
    <div style="background-color: #eee; border-radius: 10px; width: 80%; height: 24px; margin: 10px auto; position: relative;">
        <div style="background-color: rgb(113, 224, 203); width: {note}%; height: 100%; border-radius: 10px; position: absolute;">
            <span style="color: {couleur_texte}; font-weight: bold; position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center;">{note}%</span>
        </div>
    </div>
    """

    # Utiliser st.markdown pour afficher la barre de progression personnalisée
    st.markdown(html_content, unsafe_allow_html=True)


def test_affichage():
    match_result = st.session_state["matching_stat"]
    st.markdown("#")
    st.markdown("#")
    ###### AFFICHAGE DU SCORE GLOBAL
    afficher_score_card(
        match_result["MATCH_SCORE"],
        match_result["NOTE_SSKILL"],
        match_result["NOTE_STACK"],
        match_result["NOTE_TACHE"],
        match_result["NOTE LANGUE"],
        match_result["NOTE_DIPLOME"],
        match_result["NOTE_EXP"],
    )
    st.markdown("#")
    st.markdown("#")
    evaluate_job_fit(match_result["TALENT_TYPE"], match_result["JOB_TYPE"])
    st.markdown("#")
    st.markdown("#")
    niveau_diplome(
        diploma(int(match_result["MAX_DIPLOMA_CANDIDAT"])),
        diploma(int(match_result["REQUIERED_DIPLOMA"])),
        match_result["NOTE_DIPLOME"],
    )
    st.markdown("#")
    st.markdown("#")

    evaluate_soft_skills(
        match_result["DETAIL_NOTE_SSKILL"], match_result["NOTE_SSKILL"]
    )

    st.markdown("#")
    st.markdown("#")
    evaluate_language_detail(match_result["DETAIL_LANGUE"], match_result["NOTE LANGUE"])

    st.markdown("#")
    st.markdown("#")

    evaluate_stack_fit(match_result)

    st.markdown("#")
    st.markdown("#")
    evaluate_taches(match_result["TACHE"], match_result["NOTE_TACHE"])
    # st.write(st.session_state)


def visionnage_test():

    qualité_cv, rest = st.columns([3, 7], gap="large")
    with qualité_cv:
        afficher_qualites_cv()
    with rest:
        test_affichage()
