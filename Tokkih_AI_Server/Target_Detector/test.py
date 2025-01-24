from flask import Flask, jsonify
import threading
import requests
import time

app = Flask(__name__)
MAIN_SERVER_URL = "http://127.0.0.1:8000/api/receive"

class TargetDetector:
    def __init__(self):
        self.target_class = "human"
        self.dist = 10

result = TargetDetector()

@app.route("/classification", methods=["GET"])
def send_data():
    """
    if target detected ==> send data to main server
    """
    with app.app_context():
        while True:
            data = {"class" : result.target_class, "distance" : result.dist}

            response = requests.post(MAIN_SERVER_URL, json=data)

            print(f"{result.target_class} detected, distance : {result.dist}", flush=True)
            # print(f"{response.status_code} - {response.text}")

            time.sleep(2)
            result.dist += 1

            #return jsonify(data), 200
        

if __name__ == "__main__":
    thread = threading.Thread(target=send_data, daemon=True)
    thread.start()

    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)