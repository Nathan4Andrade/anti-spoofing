from flask import Flask, request, jsonify
from run_image import predict_image

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    image_path = 'temp_image.jpg'
    file.save(image_path)

    try:
        result = predict_image(image_path)
        return jsonify(result)
    except Exception as e:
        print(f"Erro na predição: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
