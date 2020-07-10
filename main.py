import logging
from io import BytesIO

import deepmux
from aiogram import Bot, Dispatcher, types
from PIL import Image
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher.filters.state import StatesGroup, State
import numpy as np
from time import ctime, time

from aiogram.utils.executor import start_webhook
from aiohttp import ClientSession

from utils import saveimg
from consts import STYLE_NAMES, TECHNOLOGIES, API_TOKEN, DEEPMUX_TOKEN, WEBHOOK_PATH, WEBAPP_HOST, WEBAPP_PORT
from image_transform import transform


logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
session = ClientSession()
dp.middleware.setup(LoggingMiddleware())


class StyleForm(StatesGroup):
    content = State()
    style = State()


class StartForm(StatesGroup):
    started = State()


async def on_startup(dispatcher):
    logging.warning('Starting on ' + str(ctime(time())))


async def on_shutdown(dispatcher):
    logging.warning('Shutdown on ' + str(ctime(time())))


@dp.message_handler(state='*', commands='cancel')
async def cancel_handler(message, state):
    await message.reply("До свидания! Введи /start, чтобы снова начать!",
                        reply_markup=types.ReplyKeyboardRemove())
    await state.finish()


@dp.message_handler(commands='start')
async def bot_start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(*TECHNOLOGIES)
    await message.reply("Привет! Умею украшать твои фотографии! Что бы ты хотел сделать? Введи /help, чтобы показать"
                        " подсказку!",
                        reply_markup=markup)
    await StartForm.started.set()


@dp.message_handler(state=StartForm.started, commands='help')
async def bot_help(message):
    await message.reply("NST (Neural Style Transfer) - технология из 2015 года, позволяющая быстро переносить "
                        "обученный стиль рисования на фотографию.\nCycleGAN - технология из 2017 года, позволяющая"
                        " делать необычные переносы стиля, начиная от переноса стиля художника на фотографию и заканчив"
                        "ая превращением зебр в лошадей или летних фотографий в зимние.")


@dp.message_handler(state=StartForm.started, commands='contacts')
async def about_us(message):
    await message.reply('Наши контакты:\n\tVK: vk.com/megetler\n\tInstagram: instagram.com/monsu.egetler')


@dp.message_handler(state=StartForm.started, content_types=['text'])
async def choice_of_tech(message, state):
    async with state.proxy() as data:
        if message.text.lower() in STYLE_NAMES.keys():
            data['tech'] = message.text.lower()
            await state.finish()
            await StyleForm.content.set()
            await message.reply('Пришли исходную фотографию!', reply_markup=types.ReplyKeyboardRemove())
        else:
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
            markup.add(*TECHNOLOGIES)
            await message.reply('Я не знаю таких технологий. Выбери из тех, что снизу.', reply_markup=markup)


@dp.message_handler(state=StyleForm.content, content_types=['photo'])
async def bot_content(message, state):
    async with state.proxy() as data:
        picture = BytesIO()
        await message.photo[-1].download(picture)
        picture.seek(0)
        await StyleForm.style.set()
        data['content'] = Image.open(picture)
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(*STYLE_NAMES[data['tech']].keys())
        await message.reply('Теперь выбери стиль!', reply_markup=markup)


@dp.message_handler(state=StyleForm.style, content_types=['text'])
async def bot_style(message, state):
    async with state.proxy() as data:
        style = STYLE_NAMES[data['tech']].get(message.text, None)
        if not style:
            markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
            await markup.add(*STYLE_NAMES[data['tech']].keys())
            await message.reply(f'Стиля "{message.text}" у меня нету. Выбери стиль.', reply_markup=markup)
            return
        if data['tech'] == 'nst':
            pic = nst_process(data['content'], style)
        elif data['tech'] == 'cyclegan':
            pic = cyclegan_process(data['content'], style)
    await message.reply_photo(pic, caption='Готово! Твой стиль: ' + message.text +
                                           ".\nВведи /start, чтобы попробовать снова.",
                              reply_markup=types.ReplyKeyboardRemove())
    await state.finish()


def cyclegan_process(content, style):
    content = transform(content)
    model = deepmux.get_model(model_name=f'{style}_pretrained_pepilica_dls',
                              token=DEEPMUX_TOKEN)
    container = BytesIO()
    output = model.run(content.unsqueeze(0).numpy()).squeeze(0).transpose(1, 2, 0)
    image_numpy = (output + 1) / 2.0 * 255.0
    # image_numpy = output.transpose(1, 2, 0) * 255
    image_pil = Image.fromarray(image_numpy.astype(np.uint8))
    image_pil.save(container, format='PNG')
    return container.getvalue()


def nst_process(content, style):
    content = transform(content)
    model = deepmux.get_model(model_name=style + '_pepilica_nst_project',
                              token=DEEPMUX_TOKEN)
    output = model.run(content.unsqueeze(0).numpy()).squeeze(0).transpose(1, 2, 0)
    state, img = saveimg(output, container=True)
    container = BytesIO(img)
    return container.getvalue()


if __name__ == '__main__':
    start_webhook(dispatcher=dp,
                  webhook_path=WEBHOOK_PATH,
                  on_startup=on_startup,
                  on_shutdown=on_shutdown,
                  skip_updates=False,
                  host=WEBAPP_HOST,
                  port=WEBAPP_PORT)
