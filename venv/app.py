from flask import Flask, render_template, request
from flask_cors import CORS
import joblib
import pandas as pd
import requests

app = Flask(__name__)
CORS(app)

# Cargar el modelo entrenado
clf = joblib.load('modelo_entrenado2.pkl')

def make_post_request(url, data):
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()  # Verificar si hay errores en la respuesta HTTP

        # Devolver la respuesta JSON
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Error en la solicitud POST:", e)

# url = 'https://fastfoodapi.herokuapp.com/predict'
url = 'http://localhost:8100/predict'

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predecir", methods=["POST"])
def predecir():
    # Obtener los datos ingresados por el usuario
    match_kills = int(request.form['MatchKills'])
    match_assists = int(request.form['MatchAssists'])
    match_headshots = int(request.form['MatchHeadshots'])
    map_selection = int(request.form['Map'])
    internal_team_id = int(request.form['InternalTeamId'])
    email = request.form['Email']

    # Crear un DataFrame con las medidas en el mismo orden de características
    datos = pd.DataFrame({
        'MatchKills': [match_kills],
        'MatchAssists': [match_assists],
        'MatchHeadshots': [match_headshots],
        'Map': [map_selection],
        'InternalTeamId': [internal_team_id]
    })

    # Realizar la predicción utilizando el modelo cargado
    prediccion = clf.predict(datos)[0]  # Obtener el valor de la predicción
    porcentaje = round(prediccion * 100)  # Redondear el porcentaje

    # Crear el objeto de datos a enviar en el cuerpo del POST
    data = {
        'email': email,
        'MatchKills': match_kills,
        'MatchAssists': match_assists,
        'MatchHeadshots': match_headshots,
        'Map': map_selection,
        'InternalTeamId': internal_team_id,
        'prediction': porcentaje
    }

    make_post_request(url, data)

    return render_template('result.html', result=porcentaje)

@app.route("/result")
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run()