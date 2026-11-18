# PASSO 2: o detector de objetos (YOLO)
#
# O nosso classificador (keras_model.h5) só conhece duas respostas: Pardal ou Pombo.
# Se alguém mandar a foto de um gato, ele vai responder "Pardal" ou "Pombo" mesmo assim!
# Para evitar isso, primeiro usamos o YOLO para descobrir O QUE aparece na imagem.
# Só se o YOLO encontrar um pássaro é que chamamos o classificador.
#
# O arquivo do modelo (yolov3.pt) precisa estar na pasta do bot. Baixe em:
# https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/yolov3.pt

from imageai.Detection import ObjectDetection
from PIL import Image

# Carregamos o YOLO uma única vez, quando o bot liga.
# O arquivo tem 240 MB: se carregássemos a cada mensagem, o bot ficaria muito lento.
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("./yolov3.pt")
detector.loadModel()

# O YOLO responde em inglês. Este dicionário traduz alguns nomes para o português.
# Os nomes que não estiverem aqui aparecem em inglês mesmo.
TRADUCAO = {
    "bird": "pássaro",
    "person": "pessoa",
    "cat": "gato",
    "dog": "cachorro",
    "car": "carro",
    "bicycle": "bicicleta",
    "tree": "árvore",
    "bench": "banco",
    "cell phone": "celular",
}


def traduzir(nome):
    # .get() devolve a tradução ou, se não existir, o próprio nome em inglês
    return TRADUCAO.get(nome, nome)


def detectar_objetos(caminho_imagem, caminho_saida):
    # O YOLO procura objetos na imagem e salva uma cópia com os objetos marcados em caixas.
    # Cada objeto encontrado é um dicionário com:
    #   "name" -> nome do objeto (ex: "bird")
    #   "percentage_probability" -> certeza de 0 a 100
    #   "box_points" -> caixa do objeto: [x1, y1, x2, y2]
    deteccoes = detector.detectObjectsFromImage(
        input_image=caminho_imagem,
        output_image_path=caminho_saida,
        minimum_percentage_probability=30,  # ignora objetos com menos de 30% de certeza
    )
    return deteccoes


def recortar_objeto(caminho_imagem, caixa, caminho_recorte):
    # Recorta só o pedaço da imagem onde está o objeto.
    # Assim o classificador olha para o pássaro, e não para o fundo da foto.
    imagem = Image.open(caminho_imagem).convert("RGB")
    recorte = imagem.crop(caixa)  # caixa = [x1, y1, x2, y2]
    recorte.save(caminho_recorte)
    return caminho_recorte
