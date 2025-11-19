import streamlit as st
import modelo_rna as rna

st.title("Trabajo Práctico Final 2025")
st.header("Modelo de Predicción de Supervivencia en el Titanic")

with st.sidebar:
    opcion=st.radio(
        "Modelos",
        [
            "Árbol de decisión",
            "Método de ensamble",
            "Red neuronal artificial",
            "Otro modelo"
        ]
    )

if opcion=="Árbol de decisión":
    pass # Reemplazar esta línea con la llamada a la predicción de este modelo

if opcion=="Método de ensamble":
    pass # Reemplazar esta línea con la llamada a la predicción de este modelo

if opcion=="Red neuronal artificial":
    rna.prediccion()

if opcion=="Otro modelo":
    pass # Reemplazar esta línea con la llamada a la predicción de este modelo
