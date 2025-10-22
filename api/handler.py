import pandas as pd
from flask import Flask, request, Response
from rossman.Rossman import Rossman
import os



app = Flask(__name__)

pipeline = Rossman()

@app.route('/rossman/predict', methods=['POST'])
def rossman_predict():
    test_json = request.get_json()

    if test_json:

        if isinstance(test_json, dict):
            df = pd.DataFrame(test_json, index=[0]) 
        else:
            df = pd.DataFrame(test_json, columns=test_json[0].keys())


        df_predict = pipeline.predict(df)

        return Response(df_predict.to_json(orient='records'), status=200, mimetype='application/json')
        
    else: 
        return Response('{}', status=200, mimetype='application/json')


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run('0.0.0.0', port=port)
