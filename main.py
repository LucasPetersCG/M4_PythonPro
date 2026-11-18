# BOT DA STARTUP "AveScan" 🐦
# Ideia da startup: quem mora na cidade manda a foto de um pássaro que apareceu na janela
# e o bot diz se é um PARDAL ou um POMBO, com uma dica do que fazer.
#
# PASSO 1: juntar o bot com o modelo treinado.
# Quais arquivos precisam estar na pasta do bot?
#   keras_model.h5 -> o modelo treinado no Teachable Machine
#   labels.txt     -> os nomes das classes (0 Sparrow, 1 Pigeon)
#   model.py       -> a função get_class(), que usa o modelo
# O que o modelo retorna? Uma dupla (nome da classe, confiança). Ex: ("Pigeon\n", 0.99)
#
# PASSO 2: inferência. O bot não devolve só um número: ele RESPONDE ao usuário,
# avisa quando algo dá errado e usa o YOLO (detector.py) para conferir se há um pássaro na foto.

import discord
from discord.ext import commands
from model import get_class
from detector import detectar_objetos, recortar_objeto, traduzir
from dotenv import load_dotenv
import os, random
import requests

# Lê o token do bot que está guardado no arquivo .env
load_dotenv()

intents = discord.Intents.default()
intents.message_content = True  # permite que o bot leia o texto das mensagens

bot = commands.Bot(command_prefix='$', intents=intents)

# Pasta onde guardamos as imagens que os usuários enviam
PASTA = "imagens"
os.makedirs(PASTA, exist_ok=True)

# Só aceitamos imagens nesses formatos
FORMATOS = (".png", ".jpg", ".jpeg")

# Abaixo de 70% de certeza, o bot prefere dizer que não sabe
CONFIANCA_MINIMA = 0.7

# A resposta que o bot dá para cada classe do labels.txt
RESPOSTAS = {
    "Sparrow": "Isso é um **pardal**! 🐦\nDica: ele come insetos e sementes. Um potinho de água na janela ajuda muito nos dias quentes.",
    "Pigeon": "Isso é um **pombo**! 🕊️\nDica: não dê comida para ele e mantenha a janela limpa. As fezes do pombo podem transmitir doenças.",
}


@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')


@bot.event
async def on_message(message):
    # O bot não responde às próprias mensagens
    if message.author == bot.user:
        return

    # Se a pessoa disser "oi" ou "olá", o bot se apresenta, como um chatbot
    if message.content.lower() in ["oi", "olá", "ola"]:
        await message.channel.send("Oi! Eu sou o AveScan 🐦. Digite `$ajuda` para ver o que eu sei fazer.")

    # IMPORTANTE: sem esta linha, os comandos com $ param de funcionar!
    await bot.process_commands(message)


@bot.command('ajuda')
async def ajuda(ctx):
    await ctx.send(
        "Eu sou o **AveScan** 🐦, o bot que identifica pássaros da cidade!\n"
        "`$check` + foto: digo se é um pardal ou um pombo e dou uma dica\n"
        "`$objetos` + foto: mostro tudo o que eu enxergo na imagem\n"
        "`$duck`: mando a foto de um pato aleatório"
    )


def get_duck_image_url():
    url = 'https://random-d.uk/api/random'
    res = requests.get(url)
    data = res.json()
    return data['url']


@bot.command('duck')
async def duck(ctx):
    print('hello')
    image_url = get_duck_image_url()
    await ctx.send(image_url)


async def baixar_imagem(ctx):
    # Confere se o usuário mandou uma imagem válida e salva na pasta "imagens".
    # Devolve o caminho do arquivo, ou None se algo estiver errado.
    if not ctx.message.attachments:
        await ctx.send("Você esqueceu de enviar uma imagem! Mande a foto junto com o comando.")
        return None

    anexo = ctx.message.attachments[0]  # usamos só a primeira imagem

    if not anexo.filename.lower().endswith(FORMATOS):
        await ctx.send("Esse formato eu não entendo. 😕 Envie uma imagem .png, .jpg ou .jpeg.")
        return None

    caminho = f"./{PASTA}/{anexo.filename}"
    await anexo.save(caminho)
    return caminho


def listar_objetos(deteccoes):
    # Transforma a lista do YOLO em um texto. Ex: "gato (98%), pessoa (75%)"
    nomes = []
    for objeto in deteccoes:
        nomes.append(f"{traduzir(objeto['name'])} ({objeto['percentage_probability']:.0f}%)")
    return ", ".join(nomes)


@bot.command('check')
async def check(ctx):
    caminho = await baixar_imagem(ctx)
    if caminho is None:
        return

    await ctx.send("Analisando a imagem... 🔍")

    # try/except: se qualquer coisa der errado na inferência, o bot avisa em vez de travar
    try:
        # 1) O YOLO procura objetos na imagem
        deteccoes = detectar_objetos(caminho, f"./{PASTA}/resultado.jpg")

        # 2) Guardamos só os pássaros
        passaros = []
        for objeto in deteccoes:
            if objeto["name"] == "bird":
                passaros.append(objeto)

        # 3) Nenhum pássaro? Então não faz sentido perguntar ao classificador
        if len(passaros) == 0:
            resposta = "Desculpe, não tenho certeza do que está sendo mostrado na imagem. 🤔 Não encontrei nenhum pássaro."
            if len(deteccoes) > 0:
                resposta += f"\nO que eu enxerguei: {listar_objetos(deteccoes)}"
            await ctx.send(resposta)
            return

        # 4) Recortamos o primeiro pássaro encontrado e mandamos só ele para o classificador
        recorte = recortar_objeto(caminho, passaros[0]["box_points"], f"./{PASTA}/recorte.jpg")
        classe, confianca = get_class(model_path="./keras_model.h5", labels_path="./labels.txt", image_path=recorte)
        classe = classe.strip()  # tira o "\n" que vem do labels.txt

    except Exception as erro:
        print("Erro na inferência:", erro)
        await ctx.send("Ops! Não consegui analisar essa imagem. 😢 Ela pode estar corrompida. Tente outra foto.")
        return

    # 5) Pouca certeza? É melhor admitir do que chutar
    if confianca < CONFIANCA_MINIMA:
        await ctx.send("Achei um pássaro, mas não tenho certeza se é pardal ou pombo. Tente uma foto mais de perto!")
        return

    await ctx.send(f"{RESPOSTAS[classe]}\n(certeza: {confianca:.0%})")


@bot.command('objetos')
async def objetos(ctx):
    caminho = await baixar_imagem(ctx)
    if caminho is None:
        return

    try:
        deteccoes = detectar_objetos(caminho, f"./{PASTA}/resultado.jpg")
    except Exception as erro:
        print("Erro na detecção:", erro)
        await ctx.send("Ops! Não consegui analisar essa imagem. 😢 Tente outra foto.")
        return

    if len(deteccoes) == 0:
        await ctx.send("Não encontrei nenhum objeto que eu conheço nessa imagem.")
        return

    # Mandamos a lista e a imagem com as caixas desenhadas pelo YOLO
    await ctx.send(f"Encontrei: {listar_objetos(deteccoes)}", file=discord.File(f"./{PASTA}/resultado.jpg"))


bot.run(os.getenv("TOKEN_DISCORD"))
