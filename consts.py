from os import environ
from urllib.parse import urljoin


API_TOKEN = environ['API_TOKEN']
DEEPMUX_TOKEN = environ['DEEPMUX_TOKEN']
WEBHOOK_HOST = environ['WEBHOOK_HOST_ADDR']
WEBHOOK_PATH = '/webhook/' + WEBHOOK_HOST
WEBHOOK_URL = urljoin(WEBHOOK_HOST, WEBHOOK_PATH)
WEBAPP_HOST = '0.0.0.0'
WEBAPP_PORT = environ['PORT']
NST_STYLES = ['mosaic', 'van_gogh', 'popova', 'kandinsky']
CYCLEGAN_STYLES = ['winter2summer_yosemite', 'summer2winter_yosemite']
STYLE_NAMES = {
    'nst': {
        'Мозайка': 'mosaic',
        'Ван Гог': 'van_gogh',
        'Попова': 'popova',
        'Кандинский': 'kandinsky'},
    'cyclegan': {
        'Из зимы в лето': 'winter2summer_yosemite',
        'Из лета в зиму': 'summer2winter_yosemite'}
}
TECHNOLOGIES = ['CycleGAN', 'NST']
