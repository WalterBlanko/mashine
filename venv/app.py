from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Cargar el modelo entrenado
clf = joblib.load('modelo_entrenado2.pkl')

@app.route("/")
def home():
    return 'Funciona?'

@app.route("/predecir", methods=["POST"])
def predecir():
    json = request.get_json(force=True)
    medidas = json['Medidas']

    # Crear un DataFrame con las medidas y el mismo orden de características
    datos = pd.DataFrame({
        'MatchKills': [medidas[0]],
        'MatchAssists': [medidas[1]],
        'MatchHeadshots': [medidas[2]],
        'Map': [medidas[3]],
        'InternalTeamId': [medidas[4]]
    })

    # Realizar la predicción utilizando el modelo cargado
    prediccion = clf.predict(datos)[0]  # Obtener el valor de la predicción
    porcentaje = round(prediccion * 100)  # Redondear el porcentaje

    return 'Los datos que proporcionaste corresponden a un {}% de probabilidades de ganar.'.format(porcentaje)

if __name__ == '__main__':
    app.run()