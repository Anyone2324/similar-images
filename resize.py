import os

from PIL import Image

path = "data_test/"
dirs = os.listdir(path)


def resize():
    for item in dirs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            f, e = os.path.splitext(path + item)
            imResize = im.resize((64, 64), Image.ANTIALIAS)
            print(item)
            imResize.save(f + '.jpg', 'JPEG', quality=90)


resize()
