import os
import cv2
import SimpleITK as sitk
import numpy as np


def load_dicom_image(image_path):
    dicom_image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(dicom_image)

    image_array = np.clip(image_array, -200, 300)
    image_array = (image_array + 200) / 500

    image_resized = cv2.resize(image_array[0], (256, 256))
    image_resized = image_resized.astype('float32')
    image_resized = np.expand_dims(image_resized, axis=-1)
    image_resized = np.expand_dims(image_resized, axis=0)

    return image_resized

output_dir = 'Processed_Images'
os.makedirs(output_dir, exist_ok=True)

art2_dir = 'Anon_Liver/Abdomen - 3928/ART_2'
for file_name in os.listdir(art2_dir):
    if file_name.endswith('.dcm'):
        image_path = os.path.join(art2_dir, file_name)
        preprocessed_image = load_dicom_image(image_path)
        if preprocessed_image is not None:
            image_to_save = preprocessed_image[0, :, :, 0] * 255
            output_path = os.path.join(output_dir, file_name.replace('.dcm', '.png'))
            cv2.imwrite(output_path, image_to_save)
