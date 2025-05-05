import base64
import requests
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if 'image_url' in data:
        image = download_image(data['image_url'])
    elif 'image_base64' in data:
        image = decode_base64_image(data['image_base64'])
    else:
        return jsonify({'error': 'Nenhuma imagem fornecida (image_url ou image_base64)'}), 400

    if image is None:
        return jsonify({'error': 'Não foi possível carregar a imagem'}), 400

    image = np.array(image)

    try:
        result = predict_image(image)
        return jsonify(result)
    except Exception as e:
        print(f"Erro na predição: {e}")
        return jsonify({'error': str(e)}), 500

def download_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        print(f"Erro ao baixar a imagem: {e}")
        return None

def decode_base64_image(b64_string):
    try:
        image_data = base64.b64decode(b64_string)
        img = Image.open(BytesIO(image_data))
        return img
    except Exception as e:
        print(f"Erro ao decodificar imagem base64: {e}")
        return None

def predict_image(image):
    if image is None:
        raise ValueError("Imagem inválida")

    height, width = image.shape[:2]
    if width / height != 3 / 4:
        print("Imagem não está no formato 3:4. Continuando mesmo assim...")

    return {
        "label": "real",
        "score": 0.95,
        "prediction_time": 0.5,
        "bbox": [0, 0, width, height]
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
