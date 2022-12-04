import os
from keras.preprocessing.image import save_img
from keras.preprocessing.image import img_to_array
import PIL.Image as Image
from keras.preprocessing.image import load_img
import shutil
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image
from PIL import ImageColor
import argparse
import os
from torch import IMG_EXTENSTIONS


def check_img(name):

    i = 0

    while(i < len(name)):
        if(name[i] == '.'):
            break
        i += 1

    if(name[i:].find(IMG_EXTENSTIONS)):
        return 1
    else:
        return 0


def convert_to_grayscale(name):
    gray_image = cv2.cvtColor(name, cv2.COLOR_BGR2GRAY)

    return gray_image


def flip_image(name, flag):

    img = name

    if(flag == "vertical"):
        img = name.transpose(method=Image.FLIP_TOP_BOTTOM)
    elif(flag == "horizontal"):
        img = name.transpose(method=Image.FLIP_LEFT_RIGHT)

    return img


def size_image(name):
    width, height = name.size

    left = 10
    top = height / 7
    right = 13
    bottom = 1 * height / 7

    im1 = name.crop((left, top, right, bottom))
    bottom += width
    newsize = (400, 400)
    im1 = im1.resize(newsize)

    return im1


def colour(img, colour):
    img = Image.open(img)
    img = img.convert("RGB")

    temp = img.getdata()

    newimg = []

    img2 = ImageColor.getrgb(colour)

    img.putdata(img2)

    img.save("coloured.jpg")
