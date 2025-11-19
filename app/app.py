import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

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

if opcion=="Red neuronal artificial":
    model=load_model("../red_neuronal_artificial/rna_titanic.h5")
    scaler=joblib.load("../red_neuronal_artificial/rna_scaler.pk1")
    
    st.write("Complete los datos del pasajero:")

    nombre=st.text_input("Nombre del pasajero")
    pclass=st.selectbox("Clase del pasajero", [1,2,3])
    sex=st.radio("Sexo",["Masculino","Femenino"])
    age=st.number_input("Edad",min_value=0.0,max_value=100.0,step=1.0)
    sibsp=st.number_input("Hermanos/esposos a bordo",min_value=0,max_value=10,step=1)
    parch=st.number_input("Padres/hijos a bordo",min_value=0,max_value=10,step=1)
    fare=st.number_input("Monto pagado",min_value=0.0,max_value=1000.0,step=1.0)
    sex_value=1 if sex=="Masculino" else 0

    familysize=sibsp+parch
    if familysize==1:
        isalone=0
    else:
        isalone=1

    if st.button("Sobrevive?"):
        features = np.array([[pclass, sex_value, age, sibsp, parch, fare, familysize, isalone]], dtype="float32")
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0][0]

        st.write("### Resultado:")

        if prediction >= 0.5:
            st.success(f"Probabilidad de supervivencia: {prediction:.2f} → **Sobrevive**")
        else:
            st.error(f"Probabilidad de supervivencia: {prediction:.2f} → **No sobrevive**")