import streamlit as st

st.title("Trabajo Práctico Final 2025")
st.header("Modelo de Predicción de Supervivencia en el Titanic")

with st.sidebar:
    opcion = st.radio(
        "Modelos",
        [
            "Árbol de decisión",
            "Método de ensamble",
            "Red neuronal artificial",
            "Otro modelo"
        ]
    )
