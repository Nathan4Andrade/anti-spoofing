# run_image.py

import os
import time
import cv2
import numpy as np
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import argparse

# Carrega apenas uma vez (importante para performance no servidor)
model_test = AntiSpoofPredict(0)
image_cropper = CropImage()
# Default, mas você pode mudar depois
model_dir = "./resources/anti_spoof_models"


def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Imagem não encontrada: {image_path}")

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
        prediction += model_test.predict(img,
                                         os.path.join(model_dir, model_name))
        test_speed += time.time() - start

    label = np.argmax(prediction)
    score = prediction[0][label] / 2

    result = {
        "label": "real" if label == 1 else "fake",
        "score": float(score),
        "prediction_time": test_speed,
        "bbox": image_bbox,
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Silent-Face-Anti-Spoofing on a single image.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Caminho da imagem para testar")
    parser.add_argument("--model_dir", type=str,
                        default="./resources/anti_spoof_models", help="Pasta dos modelos treinados")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID da GPU (use 0 ou -1 para CPU)")

    args = parser.parse_args()

    # Atualiza se vier argumento
    model_dir = args.model_dir

    print("🔎 Analisando imagem...")

    result = predict_image(args.image_path)

    print("\n🎯 Resultado:")
    print(f"Label: {result['label']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Tempo de predição: {result['prediction_time']:.4f} segundos")
    print(f"BBox detectado: {result['bbox']}")
