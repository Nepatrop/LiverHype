import numpy as np
import h5py
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout, BatchNormalization
import tensorflow.keras.backend as K
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

@register_keras_serializable()
def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def create_unet_model(input_shape=(256, 256, 1), neurons=16):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(neurons*1, (3, 3), activation='relu', padding='same', name='conv2d_1')(inputs)
    conv1 = Conv2D(neurons*1, (3, 3), activation='relu', padding='same', name='conv2d_2')(conv1)
    conv1 = BatchNormalization(name='batch_normalization_1')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(neurons*2, (3, 3), activation='relu', padding='same', name='conv2d_3')(pool1)
    conv2 = Conv2D(neurons*2, (3, 3), activation='relu', padding='same', name='conv2d_4')(conv2)
    conv2 = BatchNormalization(name='batch_normalization_2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(neurons*4, (3, 3), activation='relu', padding='same', name='conv2d_5')(pool2)
    conv3 = Conv2D(neurons*4, (3, 3), activation='relu', padding='same', name='conv2d_6')(conv3)
    conv3 = BatchNormalization(name='batch_normalization_3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(neurons*8, (3, 3), activation='relu', padding='same', name='conv2d_7')(pool3)
    conv4 = Conv2D(neurons*8, (3, 3), activation='relu', padding='same', name='conv2d_8')(conv4)
    conv4 = BatchNormalization(name='batch_normalization_4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    # Middle
    conv5 = Conv2D(neurons*16, (3, 3), activation='relu', padding='same', name='conv2d_9')(pool4)
    conv5 = Conv2D(neurons*16, (3, 3), activation='relu', padding='same', name='conv2d_10')(conv5)

    # Decoder
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
    up6 = Dropout(0.5)(up6)
    conv6 = Conv2D(neurons*8, (3, 3), activation='relu', padding='same', name='conv2d_11')(up6)
    conv6 = Conv2D(neurons*8, (3, 3), activation='relu', padding='same', name='conv2d_12')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
    up7 = Dropout(0.5)(up7)
    conv7 = Conv2D(neurons*4, (3, 3), activation='relu', padding='same', name='conv2d_13')(up7)
    conv7 = Conv2D(neurons*4, (3, 3), activation='relu', padding='same', name='conv2d_14')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
    up8 = Dropout(0.5)(up8)
    conv8 = Conv2D(neurons*2, (3, 3), activation='relu', padding='same', name='conv2d_15')(up8)
    conv8 = Conv2D(neurons*2, (3, 3), activation='relu', padding='same', name='conv2d_16')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
    up9 = Dropout(0.5)(up9)
    conv9 = Conv2D(neurons*1, (3, 3), activation='relu', padding='same', name='conv2d_17')(up9)
    conv9 = Conv2D(neurons*1, (3, 3), activation='relu', padding='same', name='conv2d_18')(conv9)

    conv10 = Dropout(0.5)(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='conv2d_19')(conv10)

    model = Model(inputs=[inputs], outputs=[conv10])
    model.compile(optimizer="adam", loss=dice_coef_loss, metrics=[dice_coef])
    return model

def load_weights_from_h5(model, h5_path):
    with h5py.File(h5_path, 'r') as f:
        layer_names = list(f.keys())
        for layer in model.layers:
            if layer.name in layer_names:
                g = f[layer.name]
                weights = [np.array(g[weight_name]) for weight_name in g.keys()]
                if 'conv2d' in layer.name:
                    weights[0] = np.transpose(weights[0], (2, 3, 0, 1))
                    if len(weights) > 1:
                        weights[1] = np.array(weights[1])
                elif 'batch_normalization' in layer.name:
                    weights = [np.array(w) for w in weights]
                
                try:
                    print(f"Слой {layer.name} - Ожидаемая форма: {layer.get_weights()[0].shape}, Полученная форма: {weights[0].shape}")
                    layer.set_weights(weights)
                    print(f"Веса успешно загружены для слоя: {layer.name}")
                except Exception as e:
                    print(f"Не удалось загрузить веса для слоя {layer.name}: {str(e)}")
    
    return model

if __name__ == '__main__':
    model = create_unet_model(input_shape=(256, 256, 1), neurons=8)

    model.summary()

    model = load_weights_from_h5(model, 'unet_r.h5')
    
    model.save('unet_r.keras', save_format='keras')
    print("Модель успешно сохранена в файл 'unet_r.keras'")
