import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

GEMINI_API_KEY = "AIzaSyC1opi9zoj3kkB-lhzoO69DX8vidzXt_Mw"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"

app = Flask(__name__)
CORS(app)

@app.route('/cure', methods=['POST'])
def get_cure():
    data = request.get_json()
    disease = data.get('disease', '')
    if not disease:
        return jsonify({'error': 'No disease provided'}), 400
    prompt = f"Suggest detailed treatment recommendations for {disease} in plants. Include organic and chemical options, prevention, and recovery timeline."
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    try:
        response = requests.post(GEMINI_API_URL, json=payload)
        if response.status_code == 200:
            gemini_data = response.json()
            text = gemini_data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            return jsonify({"treatment": text})
        else:
            return jsonify({"error": "Gemini API error", "details": response.text}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)