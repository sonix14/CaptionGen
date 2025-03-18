from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

def load_model():
    """Загрузка предобученной модели InceptionV3."""
    model = InceptionV3(weights='imagenet')
    return model

def transform_image(image_path: str):
    """Преобразование изображения для InceptionV3."""
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def generate_caption(image_path: str) -> str:
    """
    Генерирует текстовое описание для изображения с использованием InceptionV3.
    :param image_path: Путь к изображению
    :return: Сгенерированное описание
    """
    model = load_model()
    img_array = transform_image(image_path)
    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=1)[0][0]
    return f"Изображение содержит: {decoded_preds[1]} ({decoded_preds[2] * 100:.2f}% уверенность)"

# Пример использования
if name == "__main__":
    image_path = "example.jpg"  # Замените на путь к вашему изображению
    caption = generate_caption(image_path)
    print("Описание изображения:", caption)
