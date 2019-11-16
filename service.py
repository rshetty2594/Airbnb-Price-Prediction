from flask import Flask
import json
from flask import request
#from flask_cors import CORS
from price_predictor import predict_price
from airbnb_location_recommender import get_top_airbnb_location_recommendations
from competitor_recommender import get_top_competitor_location_recommendations
from OpenSSL import SSL
context = SSL.Context(SSL.PROTOCOL_TLS)
context.use_privatekey_file('server.key')
context.use_certificate_file('server.crt')



app = Flask(__name__)
#CORS(app)

@app.route("/price_predictor", methods = ['GET', 'POST'])
def price_predictor():
    response = predict_price(request.args)
    return json.dumps(response)
     
@app.route("/airbnb_location_recommender", methods = ['GET', 'POST'])
def airbnb_location_recommender():
    location_id = int(request.args.get('location_id',4085439))
    response = get_top_airbnb_location_recommendations(location_id)
    return json.dumps(response)


@app.route("/competitor_recommend_location", methods = ['GET', 'POST'])
def competitor_recommendation():
    location_id = int(request.args.get('location_id',4085439))
    response = get_top_competitor_location_recommendations(location_id)
    return json.dumps(response)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, ssl_context=context)
