import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Cache para que no recargue el modelo cada vez
@st.cache_resource
def cargar_modelo():

    # Ruta absoluta 
    base_dir = os.path.dirname(os.path.abspath(__file__))
    modelo_path = os.path.join(base_dir, "..", "otro_modelo", "model_reglogistica.pkl")

    modelo = joblib.load(modelo_path)
    return modelo


def obtener_coeficientes(model):
    feature_names = model.named_steps["preproc"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    intercept = model.named_steps["clf"].intercept_[0]
    return feature_names, coefs, intercept

def graficar_contribuciones(model, datos_usuario):
    feature_names, coefs, intercept = obtener_coeficientes(model)

    # Transforma los datos igual que en train (escalado + one-hot)
    datos_transformados = model.named_steps["preproc"].transform(datos_usuario)

    # Valores escalados
    valores = datos_transformados[0]

    # contribución = valor escalado * coef
    contribuciones = valores * coefs

    df = pd.DataFrame({
        "feature": feature_names,
        "value_original": datos_usuario.iloc[0].values,
        "value_escalado": valores,
        "coef": coefs,
        "contrib": contribuciones
    }).sort_values("contrib", ascending=True)

    # Gráfico
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(df["feature"], df["contrib"], color=np.where(df["contrib"]>0, "green", "red"))
    ax.set_xlabel("Contribución al log-odds (β * valor escalado)")
    ax.set_title("Impacto de cada variable en la predicción")
    st.pyplot(fig)

    st.write("### Detalle de contribuciones")
    st.dataframe(df)


def prediccion():
    model = cargar_modelo()

    st.write("Complete los datos del pasajero para hacer la predicción:")

    # inputs
    pclass = st.selectbox("Clase del pasajero (Pclass)", [1, 2, 3])
    sex = st.radio("Sexo", ["Masculino", "Femenino"])
    age = st.number_input("Edad", min_value=0, max_value=100, step=1)
    sibsp = st.number_input("Hermanos/esposos a bordo (SibSp)", min_value=0, max_value=10, step=1)
    parch = st.number_input("Padres/hijos a bordo (Parch)", min_value=0, max_value=10, step=1)
    fare = st.number_input("Monto pagado (Fare)", min_value=0.0, max_value=1000.0, step=1.0)
    embarked = st.selectbox("Puerto de embarque (Embarked)", ["Q", "S"])

    # Selector de threshold
    threshold_option = st.selectbox(
        "Seleccione el threshold de decisión",
        options=[0.50, 0.40],
        format_func=lambda x: f"{x:.2f}"
    )

    # Transformar sexo a 0/1
    sex_value = 1 if sex == "Masculino" else 0

    # Codificación one-hot para Embarked
    embarked_q = 1 if embarked == "Q" else 0
    embarked_s = 1 if embarked == "S" else 0

    # DataFrame con mismas columnas del entrenamiento
    datos = pd.DataFrame([{
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Pclass": pclass,
        "Sex": sex_value,
        "Embarked_Q": embarked_q,
        "Embarked_S": embarked_s
    }])
    
    # boton de predicción
    if st.button("Predecir"):

        prob = model.predict_proba(datos)[0][1]

        pred = 1 if prob >= threshold_option else 0

        st.write("### Resultado:")
        st.write(f"Probabilidad estimada: **{prob:.3f}**")
        st.write(f"Threshold usado: **{threshold_option:.2f}**")

        if pred == 1:
            st.success("Probabilidad de supervivencia: **Sobrevive**.")
        else:
            st.error("Probabilidad de supervivencia: **NO sobrevive**.")

        st.write("## Explicación del modelo (impacto de las variables)")
        graficar_contribuciones(model, datos)
