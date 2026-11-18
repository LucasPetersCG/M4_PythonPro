# PASSO 1: a função que usa o modelo treinado no Teachable Machine.
# get_class(model_path, labels_path, image_path) recebe:
#   model_path  -> caminho do modelo (keras_model.h5)
#   labels_path -> caminho dos rótulos (labels.txt)
#   image_path  -> caminho da imagem que queremos classificar
# e devolve (nome da classe, confiança). A confiança vai de 0 a 1.

from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps  # Instale pillow em vez de PIL
import numpy as np


# O Teachable Machine salva o modelo com uma opção ("groups") que esta versão do Keras
# não conhece. Esta classe aceita essa opção e simplesmente a ignora.
class CompatibleDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, groups=1, **kwargs):
        super().__init__(*args, **kwargs)


def get_class(model_path, labels_path, image_path):
    np.set_printoptions(suppress=True)

    # Carrega o modelo treinado
    model = load_model(
        model_path,
        compile=False,
        custom_objects={"DepthwiseConv2D": CompatibleDepthwiseConv2D},
    )
    # Lê os nomes das classes. Cada linha é algo como "0 Sparrow"
    class_names = open(labels_path, "r", encoding="utf-8").readlines()

    # O modelo espera receber 1 imagem de 224x224 pixels com 3 cores (RGB)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")

    # Redimensiona a imagem para 224x224 sem deformar (corta as sobras)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    # Troca os valores dos pixels de 0..255 para -1..1, como o modelo aprendeu no treino
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data[0] = normalized_image_array

    # Inferência: o modelo dá uma nota para cada classe
    prediction = model.predict(data)
    # Pegamos a classe com a maior nota
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # class_name[2:] tira o número do começo ("1 Pigeon" vira "Pigeon")
    return (class_name[2:], confidence_score)
