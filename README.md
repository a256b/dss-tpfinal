# Trabajo Práctico Final - Sistemas de Soporte para la Toma de Decisiones

## Predicción de supervivencia del Titanic con múltiples modelos

Este proyecto implementa un sistema de predicción de supervivencia del Titanic usando cuatro modelos integrados en una aplicación web desarrollada con Streamlit.

Los modelos son:
- Árbol de decisión
- Gradient boosting
- Red neuronal artificial
- Regresión logística

## Dataset

Para comenzar se debe tener una copia de los datasets 'test.csv' y 'train.csv' dentro de la carpeta 'data'.

Se encuentran en: https://www.kaggle.com/competitions/titanic/data

## Dependencias

Instalar con: `pip install -r requierements.txt`

## Análisis Exploratorio de los Datos y Preprocesamiento

Usando **Jupyter Notebook** ejecutar el archivo 'analisis_y_preprocesamiento\preprocesamiento_datos.ipynb' para generar gráficos sobre los datos y preprocesar ambos datasets.

Dentro de la carpeta 'data' se guardarán los dos datasets preprocesados con nombres: 'dataset_train.csv' y 'dataset_test.csv' que se usan para cada modelo.

## Generar modelos

Luego del preprocesamiento, se deben generar los modelos usando **Jupyter Notebook**, cada uno está detallado en su propio .ipynb en las carpetas 'arbol_de_decision', 'metodo_de_ensamble', 'red_neuronal_artificial' y 'otro_modelo'.

Cada implementación crea un archivo .h5 que contiene el modelo y un .pkl en caso de necesitar su scaler.
Estos archivos son importantes para luego realizar las predicciones con Streamlit.

## Aplicación web

Para probar las predicciones con la interfaz, se debe ejecutar 'app\app.py' con: `streamlit run app.py`