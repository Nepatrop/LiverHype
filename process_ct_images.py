import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.saving import register_keras_serializable
import SimpleITK as sitk

@register_keras_serializable()
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

@register_keras_serializable()
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

model = load_model('unet_r.keras', custom_objects={'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss})

# Функция для предобработки изображения
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=(0, -1))
    return image

# Функция для предобработки DICOM изображений
def load_dicom_image(image_path):
    dicom_image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(dicom_image)

    print(f"Форма изображения: {image_array.shape}")
    if image_array.size > 0:
        print(f"Содержимое изображения: {image_array[0]}")

    if image_array is None or image_array.size == 0:
        print("Ошибка: изображение не загружено или пустое.")
        return None

    if len(image_array.shape) != 3 or image_array.shape[0] != 1:
        print("Ошибка: изображение имеет некорректную форму.")
        return None

    image_array = np.clip(image_array, -200, 300)
    image_array = (image_array + 200) / 500

    image_resized = cv2.resize(image_array[0], (256, 256))
    image_resized = image_resized.astype('float32') 
    image_resized = np.expand_dims(image_resized, axis=-1)
    image_resized = np.expand_dims(image_resized, axis=0)

    return image_resized

# Функция для постобработки и визуализации
def visualize_result(original_image_path, mask):
    original_image = cv2.imread(original_image_path)
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))

    color_mask = np.zeros_like(original_image)
    color_mask[:, :, 1] = mask_resized * 255

    result_image = cv2.addWeighted(original_image, 1, color_mask, 0.5, 0)

    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def process_directory(directory_path):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                preprocessed_image = preprocess_image(image_path)

                prediction = model.predict(preprocessed_image)

                visualize_result(image_path, prediction[0, :, :, 0])


def process_dicom_directory(directory_path):
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.dcm'):
                image_path = os.path.join(root, file)
                preprocessed_image = load_dicom_image(image_path)
                if preprocessed_image is not None:

                    prediction = model.predict(preprocessed_image)

                    visualize_result(image_path, prediction[0, :, :, 0])

image_path = 'Anon_Liver/Abdomen - 3928/ART_2/IM-0013-0001.dcm'
preprocessed_image = load_dicom_image(image_path)

if preprocessed_image is not None:
    prediction = model.predict(preprocessed_image)
    plt.imshow(prediction[0, :, :, 0], cmap='gray')
    plt.title('Prediction')
    plt.show()
else:
    print("Ошибка при загрузке изображения.")

image_path = 'Processed_Images/IM-0013-0096.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (256, 256))
image = image.astype('float32') / 255.0
image = np.expand_dims(image, axis=-1)
image = np.expand_dims(image, axis=0)

print(f"Форма изображения для предсказания: {image.shape}")

prediction = model.predict(image)

print(f"Форма предсказания: {prediction.shape}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image[0, :, :, 0], cmap='gray')
plt.title('Input Image')
plt.subplot(1, 2, 2)
plt.imshow(prediction[0, :, :, 0], cmap='gray')
plt.title('Prediction')
plt.colorbar()
plt.show()

process_directory('Anon_Liver')

process_dicom_directory('Anon_Liver')