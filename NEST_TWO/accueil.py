import streamlit as st
from PIL import Image
from config import settings
from streamlit.components.v1 import html


def set_menu_option_integer(i: int):
    st.session_state["menu_option_integer"] = i


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
        st.markdown(
            """
            <style>
                .musala-container {
                    display: flex;
                    flex-wrap: wrap;
                    text-align: center;
                    justify-content: space-around;
                    color: #34495E; /* Gris Anthracite pour le texte */
                }
                .musala-column {
                    width: 30%;
                    margin-bottom: 20px;
                }
                h2, p {
                    margin: 0 10px;
                    color: #34495E; /* Gris Anthracite */
                    font-size: 16px; /* Adaptez cette taille au besoin */
                }
                h2 {
                    color: #1B4F72; /* Bleu Pétrole */
                    font-size: 20px; /* Adaptez cette taille au besoin */
                    margin-bottom: 10px;
                }
                .musala-header {
                    text-align: center;
                    color: #1B4F72; /* Bleu Pétrole */
                    font-size: 24px; /* Adaptez cette taille au besoin */
                    margin-bottom: 20px;
                }
            </style>
            <div class="musala-header">
                <h1>MUSALA : Votre Portail d'Emploi IA</h1>
                <p>Découvrez les meilleures opportunités adaptées à votre profil avec notre IA d'analyse de CV.</p>
            </div>
            <div class="musala-container">
                <div class="musala-column">
                    <h2>Développement & Ingénierie</h2>
                    <p>Développement de logiciels</p>
                    <p>Ingénierie des données</p>
                    <p>Développement web</p>
                    <p>OpsDev (Opérations de Développement)</p>
                </div>
                <div class="musala-column">
                    <h2>Infrastructure & Cloud</h2>
                    <p>Ingénierie réseau</p>
                    <p>Ingénierie système</p>
                    <p>Développement mobile</p>
                    <p>Cloud computing</p>
                </div>
                <div class="musala-column">
                    <h2>Sécurité & Gestion</h2>
                    <p>Sécurité informatique</p>
                    <p>Chef de projet</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#")
        st.markdown("#")

        _, col2, col3 = st.columns([0.2, 0.4, 0.4])

        with col2:
            st.button(
                "JE CHARGE MES TALENTS",
                on_click=set_menu_option_integer,
                args=(1,),
                help="Mettez en place votre Talent Nest ",
                key="button",
            )
        with col3:
            st.button(
                "JE CHERCHE MON CANDIDAT PARFAIT",
                on_click=set_menu_option_integer,
                args=(3,),
                help="Tirez profit de vos profils",
            )

    with colonne2:
        image = Image.open("nest_logo.png")
        st.image(image)
        st.markdown(
            """
            <h4 style='text-align: center;font-weight: bolder;font-size: 2vw;font-weight: 600;color:rgba(56,181,165,255);'> <bold> Hire with Ease </bold>  </h4>

            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Talent Nest", layout="wide", page_icon="nest_logo-transformed.png"
    )

    accueil()
