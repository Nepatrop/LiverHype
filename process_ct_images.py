import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from unet_model import unet_model

def preprocess_image(image_path):
    """Предобработка изображения"""
    try:
        # Загрузка DICOM файла
        if image_path.endswith('.dcm'):
            dicom = pydicom.dcmread(image_path)
            img = dicom.pixel_array
        else:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
        if img is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Изменение размера
        img = cv2.resize(img, (512, 512))
        
        # Нормализация
        img = img.astype('float32')
        img = img / np.max(img)  # Нормализуем по максимальному значению для DICOM
        
        # Добавление размерностей для batch и channels (channels_first формат)
        img = np.expand_dims(img, axis=0)  # Добавляем batch dimension
        img = np.expand_dims(img, axis=1)  # Добавляем channel dimension в начало
        
        return img
        
    except Exception as e:
        print(f"Ошибка при предобработке {image_path}: {str(e)}")
        return None

def visualize_result(original_image, prediction_mask):
    """Визуализация результатов сегментации"""
    # Создаем фигуру с тремя subplot'ами
    plt.figure(figsize=(15, 5))
    
    # Оригинальное изображение
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    
    # Маска сегментации
    plt.subplot(132)
    plt.title('Segmentation Mask')
    plt.imshow(prediction_mask, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    
    # Наложение маски на оригинальное изображение
    plt.subplot(133)
    plt.title('Overlay')
    
    # Создаем RGB версию оригинального изображения
    overlay = cv2.cvtColor(original_image.astype(np.float32), cv2.COLOR_GRAY2RGB)
    
    # Создаем маску в цветовой схеме jet
    colored_mask = plt.cm.jet(prediction_mask)[:, :, :3]  # Берем только RGB каналы
    
    # Накладываем маску с прозрачностью
    alpha = 0.4
    overlay = overlay * (1 - alpha) + colored_mask * alpha
    
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_image(image_path, model):
    """Обработка одного изображения"""
    try:
        # Загрузка и предобработка оригинального изображения
        if image_path.endswith('.dcm'):
            dicom = pydicom.dcmread(image_path)
            original_image = dicom.pixel_array
        else:
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
        if original_image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
        # Сохраняем оригинальный размер
        original_size = original_image.shape
        
        # Изменяем размер для сети
        original_resized = cv2.resize(original_image, (512, 512))
        original_resized = original_resized.astype('float32') / np.max(original_resized)
        
        # Предобработка для предсказания
        image_for_prediction = preprocess_image(image_path)
        
        print(f"Input shape: {image_for_prediction.shape}")
        
        # Получение предсказания
        prediction = model.predict(image_for_prediction, verbose=0)
        
        # Извлекаем маску из предсказания (формат channels_first)
        prediction_mask = prediction[0, 0]  # Берем первый батч и первый канал
        
        # Визуализация
        visualize_result(original_resized, prediction_mask)
        
        return True
        
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {str(e)}")
        return False

def main():
    print("Создание и компиляция модели...")
    model = unet_model()
    
    print("Загрузка весов модели...")
    try:
        model.load_weights('unet_r.h5', by_name=True, skip_mismatch=True)
        print("Веса успешно загружены\n")
    except Exception as e:
        print(f"Ошибка при загрузке весов: {str(e)}")
        return
    
    print("Тестирование на DICOM изображении...")
    test_image = "Anon_Liver/Abdomen - 3928/ART_2/IM-0013-0096.dcm"
    #test_image = "Processed_Images/1.png"
    process_image(test_image, model)

if __name__ == "__main__":
    main()