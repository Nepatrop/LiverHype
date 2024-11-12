import os
import numpy as np
import pandas as pd
from PIL import Image
from medpy.io import load
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import backend as K
from tqdm import tqdm
from joblib import Parallel, delayed
import logging


logging.basicConfig(level=logging.INFO)

def load_data(data_dir):
    filenames = []

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".dcm"):
                filenames.append(os.path.join(root, file))

    logging.info(f"Found {len(filenames)} DICOM files.")
    return sorted(filenames)

def buffer_img(filename, folder):
    try:
        img, header = load(filename)
        # Преобразуем изображение в 8-битный формат
        img = (img - img.min()) / (img.max() - img.min()) * 255
        img = img.astype(np.uint8)
        pil = Image.fromarray(img.squeeze())

        save_dir = os.path.join(folder, os.path.dirname(filename.replace('\\', '-')))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fname = os.path.join(save_dir, os.path.basename(filename).replace('.dcm', '.tiff'))
        pil.save(fname, 'TIFF', compression='none')
        return fname

    except Exception as e:
        logging.error(f"Error processing {filename}: {e}")
        return None

def buffer_images(filenames):
    folder = 'buffer'
    if not os.path.exists(folder):
        os.makedirs(folder)

    files = Parallel(n_jobs=-1)(delayed(buffer_img)(filename, folder) for filename in tqdm(filenames, position=0))
    files = [f for f in files if f is not None]  # Удаляем файлы, которые не были обработаны
    return pd.DataFrame(files)

# Загрузка датасета
data_dir = 'Anon_Liver'
filenames = load_data(data_dir)

# Буферизация изображений
X = buffer_images(filenames)
logging.info(f"Number of X samples: {len(X)}")

# Разделение данных на тренировочные и тестовые
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Определение метрик и функции потерь
def dice_coef(y_true, y_pred):
    smooth = 1e-7
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# Загрузка предобученной модели
model = load_model('unet_r.h5')
model.compile(optimizer=Adam(lr=1e-4), loss=dice_coef_loss, metrics=[dice_coef])

'''
gen_train_params = {
    'rotation_range': 10,
    'fill_mode': 'reflect',
}

idg_train_data = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    **gen_train_params,
)

train_gen_params = {
    'x_col': 0,
    'target_size': (512, 512),
    'color_mode': 'grayscale',
    'batch_size': 4,
    'class_mode': None,
    'shuffle': True,
    'seed': 42,
}
val_gen_params = train_gen_params.copy()
val_gen_params['shuffle'] = False
val_gen_params['batch_size'] = 1

data_train_generator = idg_train_data.flow_from_dataframe(X_train, **train_gen_params)
data_test_generator = idg_train_data.flow_from_dataframe(X_test, **val_gen_params)

history = model.fit(
    data_train_generator,
    validation_data=data_test_generator,
    steps_per_epoch=X_train.shape[0] // train_gen_params['batch_size'],
    validation_steps=X_test.shape[0] // val_gen_params['batch_size'],
    verbose=1,
    epochs=150,
    callbacks=[early_stop, reduce_lr]
)

model.save('path_to_save_finetuned_model.h5')

def evaluate_model(model, X_test, y_test):
    # Ваш код для оценки модели
    pass

evaluate_model(model, X_test, y_test)
'''