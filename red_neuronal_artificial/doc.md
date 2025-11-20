Para usar la predicción con la red neuronal:

1. Ejecutar rna.ipynb usando Jupyter Notebook para entrenar la red neuronal y generar los archivos necesarios para la predicción.

- rna_titanic.h5: modelo entrenado
- rna_scaler.pkl: escalador usado

2. Ejecutar app.py usando Streamlit con:
    streamlit run app.py

3. Esto abre la interfaz web donde se pueden cargar los datos de un pasajero para comprobar el modelo.