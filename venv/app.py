from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import sklearn

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return 'Funciona?'

#@app.route("/predecir", methods=["POST"])
#def predecir():
#    json=request.get_json(force=True)
#    medidas=json['Medidas']
#    clf=joblib.load('modelo_entrenado2.pkl')
#    prediccion=clf.predict(medidas)
#    return 'Los datos que proporcionaste corresponde al porcentaje de {0}'.format(prediccion)

if __name__ == '__main__':
    app.run()

#hola 