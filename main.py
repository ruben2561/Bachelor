from PIL import Image, ImageDraw, ImageFont
import random
import string
import os
import glob
import functions
import generateTextImages
import time

def generate_random(amount):
    for i in range(amount):
        jpegPercentage = random.choice([0,10,20,30,40,50,60,70,80,90,100])
        pixelPercentage = random.choice([1,2,3,4,5,6,7,8,9])
        text = functions.generate_word()
        generateTextImages.generate_image_with_jpeg_pixelation(jpegPercentage,pixelPercentage, text)

def generate_notgood():
    #quantity
    quantity = 100
    paths = []

    for i in range(quantity):
        generateTextImages.generate_image(functions.generate_word())

    #collect all generated images in a list
    files = glob.glob('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/text_images/*')
    for f in files:
        paths.append(f)

    for path in paths:
        jpegPercentage = random.choice([0,10,20,30,40,50,60,70,80,90,100])
        pixelPercentage = random.choice([1,2,3,4,5,6,7,8,9])
        generateTextImages.JPEG_compression(path, jpegPercentage)
        generateTextImages.pixelate(path.split(".jpg")[0] + "__jpeg" + str(jpegPercentage) + ".jpg", pixelPercentage)
        os.remove(path)
        os.remove(path.split(".jpg")[0] + "__jpeg" + str(jpegPercentage) + ".jpg")


#####
#####
##### execute when running
#####
#####

#remove previous generated files in folder
files = glob.glob('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/text_images/*')
for f in files:
    os.remove(f)

start = time.time()

generate_random(2500)

end = time.time()
print(str(end - start) + " seconds")

#####
#####
#####
#####
#####






def process(line):
    print("1. " + line)

#read all lines in file and execute process
def read_in_file():
    with open('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/email_dataset_small.txt') as f:
        for line in f:
            process(line)

