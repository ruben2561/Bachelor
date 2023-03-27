from PIL import Image, ImageDraw, ImageFont
import random
import string
import os
import glob
import functions
import Generate_Text_images

#remove previous generated files in folder
files = glob.glob('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/text_images/*')
for f in files:
    os.remove(f)

Generate_Text_images.generate_image(functions.generate_word())

#values = range(6)
#for i in values:
    #functions.JPEG_compression("text_images/" + text + ".jpg", i*20)
    #functions.pixelate("text_images/" + text + ".jpg", i+1)
files = glob.glob('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/text_images/*')
for f in files:
    for i in range(6):
        functions.JPEG_compression(f, i*20)
        functions.pixelate(f, i+1)


def process(line):
    print("1. " + line)

#with open('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/email_dataset_small.txt') as f:
#    for line in f:
#        generate_image(line)