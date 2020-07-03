import logging
from io import BytesIO
import deepmux
from aiogram import Bot, Dispatcher, executor, types
from PIL import Image
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import StatesGroup, State
from utils import saveimg
from random import randint
from consts import NST_STYLE_NAMES, DEEPMUX_TOKEN, API_TOKEN
from image_transform import transform


logging.basicConfig(level=logging.INFO)

NUM_STYLES = 4
bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)


class Form(StatesGroup):
    content = State()
    style = State()


@dp.message_handler(commands='start')
async def bot_start(message):
    await Form.content.set()

    await message.reply("Привет! Могу переносить стиль с одной картинки на другую. Пришли исходную фотографию!",
                        reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler(state=Form.content, content_types=['photo'])
async def bot_content(message, state):
    async with state.proxy() as data:
        picture = BytesIO()
        await message.photo[-1].download(picture)
        picture.seek(0)
        await Form.style.set()
        data['content'] = Image.open(picture)
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
        markup.add(*NST_STYLE_NAMES.keys())
        await message.reply('Теперь выбери стиль!', reply_markup=markup)


@dp.message_handler(lambda message: message.text in NST_STYLE_NAMES.keys(), state=Form.style)
async def bot_style(message, state):
    async with state.proxy() as data:
        style = NST_STYLE_NAMES.get(message.text)
        pic, num_style = process_image(data['content'], style)
    await message.reply_photo(pic, caption='Готово! Ваш стиль: ' + message.text, reply_markup=types.ReplyKeyboardRemove())
    await state.finish()


@dp.message_handler(lambda message: message.text not in NST_STYLE_NAMES.keys(), state=Form.style)
async def no_style_found(message, state):
    style = message.text
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, selective=True)
    markup.add(*NST_STYLE_NAMES.keys())
    await message.reply(f'Стиля "{style}" у меня нету. Выбери стиль.', reply_markup=markup)


def process_image(content, style):
    num_style = randint(0, 3)
    content = transform(content)
    model = deepmux.get_model(model_name=style+'_pepilica_nst_project',
                              token=DEEPMUX_TOKEN)
    output = model.run(content.unsqueeze(0).numpy()).squeeze(0).transpose(1, 2, 0)
    state, img = saveimg(output, container=True)
    container = BytesIO(img)
    return container.getvalue(), num_style


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
