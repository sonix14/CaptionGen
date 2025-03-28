import tensorflow as tf
from tqdm import tqdm
import numpy as np
V3_DIMMS = (299, 299)
def load_preformatted_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, V3_DIMMS)
    # Нормализация значений пикселей в интервале (-1; 1)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, img_path

def get_features_model():
    v3_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet')
    v3_input = v3_model.input
    v3_output = v3_model.layers[-1].output
    return tf.keras.Model(v3_input, v3_output)

V3_BATCH_SIZE = 16

def fetch_image_features(img_paths, features_model,
                         batch=V3_BATCH_SIZE):
    img_dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    # Ассоциируем функцию загрузки с датасетом
    img_dataset = img_dataset.map(load_preformatted_image,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Данные будут группироваться в пакеты для оптимизации вычислений
    img_dataset = img_dataset.batch(batch)

    for img, path in tqdm(img_dataset):
        # Применяем модель для получения весов
        batch_features = features_model(img)
        batch_features = tf.reshape(batch_features,
            (batch_features.shape[0], -1, batch_features.shape[3]))

        # Кэшируем данные на диск в файлы формата .npy
        # Имя файла такое же, как и у изображения
        for bf, p in zip(batch_features, path):
            feature_path = p.numpy().decode('utf-8')
            np.save(feature_path, bf.numpy())
