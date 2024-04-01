import streamlit as st
import pandas as pd
from PIL import Image
import hashlib
import io
from annotated_text import annotated_text
import streamlit as st
import pandas as pd
import time
import json
from pdfminer.high_level import extract_text
import re
import streamlit as st
import pandas as pd
from PIL import Image
import hashlib
import io
from itertools import count
import time

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


from lst_fction import (
    parsing_joboffer,
    matching_offer_with_candidat,
    diploma,
    mise_en_forme_diplome,
    match_visualisation,
    encoder_et_obtenir_equivalences_dedupliquees,
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


@st.cache_data(ttl=3600, show_spinner=False)
def cache_matching(result_cv: dict, result_job_offer: dict):
    return matching_offer_with_candidat(result_cv, result_job_offer)


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

        st.write(result_matching)
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
        col1, col2, col3 = st.columns([0.15, 0.7, 0.15])
        with col2:
            st.markdown("#")
            st.markdown("#")
            matching_process_html_with_intro = """
            <style>
                .matching-container {
                    display: flex;
                    justify-content: space-around;
                    flex-wrap: wrap;
                    gap: 10px; /* Espacement entre les éléments */
                    text-align:center;
                }
                details {
                    flex-basis: 30%; /* Ajustez selon l'espace désiré autour des éléments */
                    border: 1px solid #DAE8FC;
                    border-radius: 10px;
                    padding: 10px;
                    background-color: #DAE8FC;
                    cursor: pointer;
                }
                summary {
                    font-weight: bold;
                    font-size: 1.0vw;
                    color: #34495E; /* Couleur du texte pour les titres */
                }
                p {
                    font-size: 0.9vw;
                    color: #000000; /* Texte noir pour une meilleure lisibilité */
                }
            
                .intro-text {
                    margin-bottom: 30px; /* Espacement avant la section des étapes */
                    color: #34495E; /* Couleur du texte pour l'introduction */
                    font-size: 2.9vw;
                    text-align: center;
                }
            </style>

            <div class="intro-text">
                <p style="font-size: 1.3vw; color: #34495E;">
                    Dans un monde où 80% des processus de recrutement sont désormais assistés par l'intelligence artificielle et des outils numériques de filtrage, comprendre et maîtriser ces technologies devient essentiel.<br>
                    Découvrez comment maximiser vos chances dans ce nouvel écosystème de recrutement :
                </p>
            </div>

            <div class="matching-container">
                <details>
                    <summary>1. Évaluation de vos compétences</summary>
                    <p><br>Votre CV est analysé de manière approfondie pour prendre connaissance de vos capacités.<br>  Prise de connaissance de vos skills et de vos differentes éxperiences et dommaine d'expertise.</p>
                </details>
                <details>
                    <summary>2. Sélectionnez vos opportunités </summary>
                    <p><br>Choisissez les offres parmis plus de 100 000 offres  qui vous intéressent en utilisant les filtres suivant vos besoins, type de contrat ou contraintes de mobilité et d'expérience.</p>
                </details>
                <details>
                    <summary>3. Exploration des compatibilités</summary>
                    <p><br>Nest trouve les points forts de votre candidature et les différentes opportunitées qui vous correspondent le mieux.</p>
                </details>
            </div>
            """

            # Afficher le contenu sur la page Streamlit
            st.markdown(matching_process_html_with_intro, unsafe_allow_html=True)

        st.markdown("#")
        st.markdown(
            """
        <style>
            .upload-instruction {
                font-size: 1.7vw;
                color: #34495E; /* Adaptez cette valeur à la couleur désirée */
                text-align:center;
                font-weight: bolder;
            }
        </style>

        <div class="upload-instruction">
            Commençons par prendre connaissance de votre profil<br>  
        </div>
        """,
            unsafe_allow_html=True,
        )
        col1, col2, col3 = st.columns([0.2, 0.6, 0.2])
        with col2:
            uploaded_file = st.file_uploader(" ", type="pdf")
        col1, col2, col3 = st.columns([0.35, 0.3, 0.35], gap="large")
        with col2:
            st.markdown("#")
            st.markdown("#")

            bouton_validation = st.button("Lancez la recherche")
            if uploaded_file is not None and bouton_validation is not None:
                st.empty()
                # Stockez temporairement le fichier pour éviter la perte lors du rechargement
                st.session_state["temp_uploaded_file"] = uploaded_file
                st.session_state["fichier_telecharger"] = 1
                # Changement de l'état pour passer au parsing
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
                    background-color: #E8F0FE; /* Un bleu clair pour un aspect frais et accueillant */
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
                    background-color: #1B4F72; /* Bleu foncé pour le bouton */
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
            background-color: #E0E0E0; /* Fond gris clair */
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
            <div class="candidat-info">{' - '.join([x.capitalize() for x in data['langue']])}</div>
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
                        "/Users/tuyindig/Documents/Nest_app/nest-app/cities.json",
                    )

                    offres_emploi = ajouter_type_contrat(offres_emploi)
                    offres_emploi = ajouter_niveau_experience(offres_emploi)

                    st.session_state["offres_emploi"] == offres_emploi
                else:
                    offres_emploi = st.session_state["offres_emploi"]
                    offres_emploi = ajouter_region_a_offres_depuis_fichier(
                        offres_emploi,
                        "/Users/tuyindig/Documents/Nest_app/nest-app/cities.json",
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
    st.write(st.session_state["result_matching"])


def job_offer_parser(selected):
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
