import streamlit as st
import numpy as np
import joblib

@st.cache_resource
def cargar_modelo():
    model = joblib.load("../arbol_de_decision/arbol_decision_titanic.pkl")
    return model

def prediccion():
    model = cargar_modelo()

    st.write("Complete los datos del pasajero:")

    nombre = st.text_input("Nombre del pasajero")
    pclass = st.selectbox("Clase del pasajero", [1, 2, 3])
    sex = st.radio("Sexo", ["Masculino", "Femenino"])
    age = st.number_input("Edad", min_value=0.0, max_value=100.0, step=1.0)
    sibsp = st.number_input("Hermanos/esposos a bordo", min_value=0, max_value=10, step=1)
    parch = st.number_input("Padres/hijos a bordo", min_value=0, max_value=10, step=1)
    fare = st.number_input("Monto pagado", min_value=0.0, max_value=1000.0, step=1.0)

    # Convertir sexo a valor numerico (1 para masculino, 0 para femenino)
    sex_value = 1 if sex == "Masculino" else 0

    # Calcular FamilySize e IsAlone
    familysize = sibsp + parch + 1
    isalone = 1 if familysize == 1 else 0

    if st.button("¿Sobrevive?"):
        # Crear array con las features en el orden correcto:
        # ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone']
        features = np.array([[pclass, sex_value, age, sibsp, parch, fare, familysize, isalone]])

        # Hacer prediccion
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]

        st.write("### Resultado:")

        if prediction == 1:
            st.success(f"**{nombre if nombre else 'El pasajero'}** tiene una probabilidad de supervivencia de **{prediction_proba[1]:.2%}** → **SOBREVIVE**")
        else:
            st.error(f"**{nombre if nombre else 'El pasajero'}** tiene una probabilidad de supervivencia de **{prediction_proba[1]:.2%}** → **NO SOBREVIVE**")

        # Mostrar detalles adicionales
        with st.expander("Ver detalles de la predicción"):
            st.write(f"**Probabilidad de no sobrevivir:** {prediction_proba[0]:.2%}")
            st.write(f"**Probabilidad de sobrevivir:** {prediction_proba[1]:.2%}")
            st.write(f"**Características ingresadas:**")
            st.write(f"- Clase: {pclass}")
            st.write(f"- Sexo: {sex}")
            st.write(f"- Edad: {age} años")
            st.write(f"- Hermanos/esposos a bordo: {sibsp}")
            st.write(f"- Padres/hijos a bordo: {parch}")
            st.write(f"- Tarifa pagada: ${fare:.2f}")
            st.write(f"- Tamaño de familia: {familysize}")
            st.write(f"- Viaja solo: {'Sí' if isalone == 1 else 'No'}")
