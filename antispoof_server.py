import base64
import requests
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import numpy as np
from flask_cors import CORS
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import cv2
import os
import time

model_test = AntiSpoofPredict(0)
image_cropper = CropImage()
model_dir = "./resources/anti_spoof_models"


app = Flask(__name__)
CORS(app) 

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
        raise ValueError("Imagem não fornecida ou inválida.")

    # Converte para formato esperado (BGR)
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    height, width, _ = image.shape
    if width / height != 3 / 4:
        print("Imagem não está no formato 3:4. Continuando mesmo assim...")

    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))
    test_speed = 0

    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        start = time.time()
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    label = np.argmax(prediction)
    score = prediction[0][label] / 2

    return {
        "label": "real" if label == 1 else "fake",
        "score": float(score),
        "prediction_time": test_speed,
        "bbox": image_bbox,
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
