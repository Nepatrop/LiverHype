import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from unet_model import unet_model

def preprocess_image(image_path):
    try:
        if image_path.endswith('.dcm'):
            dicom = pydicom.dcmread(image_path)
            img = dicom.pixel_array
            
            min_hu = -100
            max_hu = 200
            img = np.clip(img, min_hu, max_hu)
            
            # Нормализуем в диапазон [0, 1]
            img = (img - min_hu) / (max_hu - min_hu)
            img = np.rot90(img, k=1)
            
        else:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            img = img / 255.0
            img = np.rot90(img, k=1)
            
        img = cv2.resize(img, (512, 512))
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=1)
        
        return img
        
    except Exception as e:
        print(f"Ошибка при предобработке {image_path}: {str(e)}")
        return None

def visualize_result(original_image, prediction_mask):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(132)
    plt.title('Segmentation Mask')
    plt.imshow(prediction_mask, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(133)
    plt.title('Overlay')
    
    overlay = cv2.cvtColor(original_image.astype(np.float32), cv2.COLOR_GRAY2RGB)
    
    colored_mask = plt.cm.jet(prediction_mask)[:, :, :3]
    
    alpha = 0.4
    overlay = overlay * (1 - alpha) + colored_mask * alpha
    
    plt.imshow(overlay)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def adjust_contrast_brightness(image, alpha=1.3, beta=0.1):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    adjusted = adjusted.astype('float32') / 255.0
    adjusted = np.clip(adjusted + beta, 0, 1) 
    return adjusted

def adjust_gamma(image, gamma=1.0):
    return np.power(image, gamma)

def process_image(image_path, model, threshold=0.3):  
    if image_path.lower().endswith('.dcm'):
        dcm = pydicom.dcmread(image_path)
        image = dcm.pixel_array
        
        print("\nИнформация о DICOM изображении:")
        print(f"Размер: {image.shape}")
        print(f"Тип данных: {image.dtype}")
        print(f"Минимальное значение пикселей: {np.min(image)}")
        print(f"Максимальное значение пикселей: {np.max(image)}")
        print(f"Среднее значение пикселей: {np.mean(image)}")
        
        image = image.astype('float32')
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        image = adjust_gamma(image, gamma=0.8)
        
        image_uint8 = (image * 255).astype(np.uint8)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        image = clahe.apply(image_uint8).astype('float32') / 255.0
        
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype('float32') / 255.0
        
        image = adjust_gamma(image, gamma=0.8)
        
        image_uint8 = (image * 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))
        image = clahe.apply(image_uint8).astype('float32') / 255.0

    image = np.rot90(image)
    image = np.flipud(image)
    
    if image.shape != (512, 512):
        image = cv2.resize(image, (512, 512))
    
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    
    print(f"\nInput shape: {image.shape}")
    
    prediction = model.predict({'input_layer': image}, verbose=0)
    prediction = np.squeeze(prediction)

    binary_mask = (prediction > threshold).astype(np.uint8)
    
    num_labels, labels = cv2.connectedComponents(binary_mask)
    
    if num_labels > 1:
        sizes = [np.sum(labels == i) for i in range(1, num_labels)]
        largest_label = 1 + np.argmax(sizes)
        
        max_size = max(sizes)
        min_size = max_size * 0.05
        
        binary_mask = np.zeros_like(labels, dtype=np.float32)
        for i in range(1, num_labels):
            if sizes[i-1] >= min_size:
                binary_mask += (labels == i).astype(np.float32)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    binary_mask = cv2.morphologyEx(binary_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close)
    
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_open)
    
    binary_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (5, 5), 0)
    binary_mask = (binary_mask > 0.5).astype(np.float32)
    
    print("\nРезультаты предсказания:")
    print(f"Минимальное значение: {np.min(prediction):.4f}")
    print(f"Максимальное значение: {np.max(prediction):.4f}")
    print(f"Среднее значение: {np.mean(prediction):.4f}")
    print(f"Порог: {threshold}")
    print(f"Процент пикселей выше порога: {(prediction > threshold).mean()*100:.2f}%")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.title('Исходное изображение')
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.axis('off')
    
    plt.subplot(142)
    plt.title('Вероятности')
    plt.imshow(prediction, cmap='jet')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(143)
    plt.title('Бинарная маска')
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(144)
    plt.title('Наложение')
    plt.imshow(np.squeeze(image), cmap='gray')
    plt.imshow(binary_mask, cmap='jet', alpha=0.5)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    model = unet_model()
    
    print("Загрузка весов модели...")
    try:
        model.load_weights('unet_r.h5')
        
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                print(f"Слой {layer.name}:")
                for w in weights:
                    print(f"  Форма: {w.shape}")
                    
        print("Веса успешно загружены\n")
    except Exception as e:
        print(f"Ошибка при загрузке весов: {str(e)}")
        return
    
    test_image = "Anon_Liver/Abdomen - 3928/ART_2/IM-0013-0096.dcm"
    #test_image = "Processed_Images/1.png"
    process_image(test_image, model)

if __name__ == "__main__":
    main()