import streamlit as st
import numpy as np
import joblib

@st.cache_resource
def cargar_modelo():
    modelo = joblib.load("../metodo_de_ensamble/modelo_gradient_boosting.joblib")
    return modelo

def prediccion():
    modelo = cargar_modelo()

    st.write("Complete los datos del pasajero:")

    pclass = st.selectbox("Clase del pasajero", [1, 2, 3])
    sex = st.radio("Sexo", ["Masculino", "Femenino"])
    age = st.number_input("Edad", min_value=0.0, max_value=100.0, step=1.0)
    sibsp = st.number_input("Hermanos/esposos a bordo", min_value=0, max_value=10, step=1)
    parch = st.number_input("Padres/hijos a bordo", min_value=0, max_value=10, step=1)
    fare = st.number_input("Monto pagado", min_value=0.0, max_value=1000.0, step=1.0)

    sex_value = 1 if sex == "Masculino" else 0
    familysize = sibsp + parch
    isalone = 1 if familysize == 1 else 0

    if st.button("Sobrevive?"):
        features = np.array([[pclass, sex_value, age, sibsp, parch, fare, familysize, isalone]])

        pred = modelo.predict_proba(features)[0][1]

        st.write("### Resultado:")

        if pred >= 0.5:
            st.success(f"Probabilidad de supervivencia: {pred:.2f} → **Sobrevive**")
        else:
            st.error(f"Probabilidad de supervivencia: {pred:.2f} → **No sobrevive**")
