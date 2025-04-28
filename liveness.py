import os
import time
import cv2
import numpy as np
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

def predict_liveness(image_path, model_dir, device_id=0):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Imagem não encontrada: {image_path}")

    # Verifica proporção 3:4 (opcional, depende se você quer mesmo restringir)
    height, width, _ = image.shape
    if width/height != 3/4:
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

    result = {
        "label": "real" if label == 1 else "fake",
        "score": float(score),
        "prediction_time": test_speed,
        "bbox": image_bbox,
    }

    return result
