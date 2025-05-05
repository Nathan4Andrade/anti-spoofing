import requests
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image_url' not in request.json:
        return jsonify({'error': 'image_url não fornecido'}), 400

    image_url = request.json['image_url']

    # Tenta baixar e processar a imagem da URL
    image = download_image(image_url)
    if image is None:
        return jsonify({'error': 'Não foi possível baixar a imagem'}), 400

    # Converte a imagem para um array NumPy (OpenCV usa esse formato)
    image = np.array(image)

    # Lógica de predição
    try:
        result = predict_image(image)
        return jsonify(result)
    except Exception as e:
        print(f"Erro na predição: {e}")
        return jsonify({'error': str(e)}), 500

def download_image(image_url):
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Levanta um erro se a resposta for 4xx ou 5xx
        img = Image.open(BytesIO(response.content))
        return img
    except requests.exceptions.RequestException as e:
        print(f"Erro ao baixar a imagem: {e}")
        return None

def predict_image(image):
    """Função para fazer a predição da imagem."""
    if image is None:
        raise ValueError(f"Imagem não válida fornecida!")

    height, width, _ = image.shape
    if width / height != 3 / 4:
        print("Imagem não está no formato 3:4. Continuando mesmo assim...")

    # Chame o modelo para fazer a predição (você já tem a lógica aqui)
    # Substitua a linha abaixo pelo seu código de predição real
    result = {"label": "real", "score": 0.95, "prediction_time": 0.5, "bbox": [0, 0, width, height]}

    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
