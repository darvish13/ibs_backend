from flask import Flask, request
from skyline import get_skyline
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# if app.config["DEBUG"]:
#     @app.after_request
#     def after_request(response):
#         response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
#         response.headers["Expires"] = 0
#         response.headers["Pragma"] = "no-cache"
#         return response


@app.get('/')
def index():
    return 'Listening ...', 200


@app.post('/get-skyline')
def getSkyline():

    # Destructure image, mask, alpha, beta from request data
    image, mask, alpha, beta = request.json['image'], request.json['mask'], request.json['alpha'], request.json['beta']

    skyline = get_skyline(image, mask, alpha, beta)
    return skyline, 200


if __name__ == '__main__':
    app.run(debug=True)

# pip install pyopenssl
# app.run(ssl_context=('cert.pem', 'key.pem'))
# flask run --host=0.0.0.0 --port=5000 --cert=adhoc || --cert=cert.pem --key=key.pem
