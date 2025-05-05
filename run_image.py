import os
import time
import cv2
import numpy as np
import base64
import requests
from io import BytesIO
from PIL import Image
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

# Carrega apenas uma vez (importante para performance no servidor)
model_test = AntiSpoofPredict(0)
image_cropper = CropImage()
model_dir = "./resources/anti_spoof_models"

def decode_base64_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(BytesIO(img_data))
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def load_image_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Erro ao baixar imagem da URL: {url}")
    img = Image.open(BytesIO(response.content))
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def predict_image(image):
    if image is None:
        raise ValueError("Imagem não fornecida ou inválida.")

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
