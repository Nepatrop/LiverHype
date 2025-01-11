from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import pydicom
from unet_model import unet_model
from io import BytesIO
import base64

app = Flask(__name__)

model = unet_model()
model.load_weights('unet_r.h5')

def adjust_gamma(image, gamma=1.0):
    return np.power(image, gamma)

def adjust_contrast_brightness(image, alpha=2.3, beta=0.1):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    adjusted = adjusted.astype('float32') / 255.0
    adjusted = np.clip(adjusted + beta, 0, 1)
    return adjusted

def preprocess_image(image):
    try:
        if isinstance(image, bytes):
            try:
                dicom = pydicom.dcmread(BytesIO(image))
                img = dicom.pixel_array
            except:
                img = np.frombuffer(image, dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise ValueError("Не удалось прочитать изображение")
        else:
            try:
                dicom = pydicom.dcmread(BytesIO(image))
                img = dicom.pixel_array
            except:
                raise ValueError("Файл не является DICOM-изображением")

        # Нормализация и предобработка
        if isinstance(image, bytes):
            # DICOM
            img = img.astype('float32')
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
            img = adjust_gamma(img, gamma=0.8)
            img_uint8 = (img * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            img = clahe.apply(img_uint8).astype('float32') / 255.0
        else:
            img = img.astype('float32') / 255.0
            img = adjust_gamma(img, gamma=0.8)
            img_uint8 = (img * 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            img = clahe.apply(img_uint8).astype('float32') / 255.0

        img = np.rot90(img)
        img = np.flipud(img)

        if img.shape != (512, 512):
            img = cv2.resize(img, (512, 512))

        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=0)

        return img

    except Exception as e:
        print(f"Ошибка при предобработке изображения: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

def validate_image(file):
    """Валидация загруженного файла"""
    allowed_extensions = {'dcm', 'png', 'jpg', 'jpeg'}
    filename = file.filename.lower()
    return '.' in filename and filename.rsplit('.', 1)[1] in allowed_extensions

@app.route('/process', methods=['POST'])
def process_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if not validate_image(file):
            return jsonify({'error': 'Invalid file format'}), 400

        image = file.read()
        processed_image = preprocess_image(image)

        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400

        try:
            prediction = model.predict(processed_image)
        except Exception as e:
            print(f"Model prediction error: {str(e)}")
            return jsonify({'error': 'Model prediction failed'}), 500

        prediction = np.squeeze(prediction)

        prediction = cv2.GaussianBlur(prediction, (5, 5), 0)
        binary_mask = (prediction > 0.5).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            binary_mask = np.zeros_like(binary_mask)
            cv2.drawContours(binary_mask, [largest_contour], -1, 1, -1)
        
        binary_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (3, 3), 0)
        binary_mask = (binary_mask > 0.5).astype(np.uint8)

        original_image = np.squeeze(processed_image)
        original_image = (original_image * 255).astype(np.uint8)

        _, buffer_original = cv2.imencode('.png', original_image)
        original_base64 = base64.b64encode(buffer_original).decode('utf-8')

        _, buffer_mask = cv2.imencode('.png', (binary_mask * 255).astype(np.uint8))
        mask_base64 = base64.b64encode(buffer_mask).decode('utf-8')

        return jsonify({
            'original_url': f'data:image/png;base64,{original_base64}',
            'mask_url': f'data:image/png;base64,{mask_base64}'
        })

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)