from flask import Flask, jsonify, request, Response

flask_app = Flask(__name__)

@flask_app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        pass

print("Successfully started Flask helper")
flask_app.run(host="0.0.0.0", port=9001)
