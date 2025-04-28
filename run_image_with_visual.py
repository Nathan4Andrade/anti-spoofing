import os
import time
import cv2
import numpy as np
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import argparse

def predict_liveness(image_path, model_dir, device_id=0):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

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

    return result, image

def display_result(image, result):
    label = result["label"]
    score = result["score"]
    bbox = result["bbox"]
    
    color = (0, 0, 255) if label == "fake" else (255, 0, 0)
    text = f"{label.capitalize()} - {score:.2f}"

    # Desenha a BBox
    cv2.rectangle(image, 
                  (bbox[0], bbox[1]), 
                  (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                  color, 2)
    
    # Coloca o texto
    cv2.putText(image, text, 
                (bbox[0], bbox[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.9, color, 2)
    
    # Exibe a imagem
    cv2.imshow("Resultado", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Silent-Face-Anti-Spoofing on a single image and display result.")
    parser.add_argument("--image_path", type=str, required=True, help="Caminho da imagem para testar")
    parser.add_argument("--model_dir", type=str, default="./resources/anti_spoof_models", help="Pasta dos modelos treinados")
    parser.add_argument("--device_id", type=int, default=0, help="ID da GPU (use 0 ou -1 para CPU)")

    args = parser.parse_args()

    print("🔎 Analisando imagem...")

    result, image = predict_liveness(args.image_path, args.model_dir, args.device_id)

    print("\n🎯 Resultado:")
    print(f"Label: {result['label']}")
    print(f"Score: {result['score']:.4f}")
    print(f"Tempo de predição: {result['prediction_time']:.4f} segundos")
    print(f"BBox detectado: {result['bbox']}")

    # Exibe a imagem com a BBox e o resultado
    display_result(image, result)
