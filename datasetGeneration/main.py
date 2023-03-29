from PIL import Image, ImageDraw, ImageFont
import random
import string
import os
import glob
#import datasetGeneration.functions as functions
#import datasetGeneration.generateTextImages as generateTextImages
import generateTextImages
import functions
import time

def remove_previous():
    #remove previous generated files in folder
    files = glob.glob('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/text_images/*')
    for f in files:
        os.remove(f)

def generate_random(amount, save_path):

    for i in range(amount):
        jpegPercentage = random.choice([0,10,20,30,40,50,60,70,80,90,100])
        pixelPercentage = random.choice([1,2,3,4,5,6,7,8,9])
        text = functions.generate_word()
        generateTextImages.generate_image_with_jpeg_pixelation(jpegPercentage, pixelPercentage, text, save_path)

def generate_specific(text, save_path):
    jpegPercentage = random.choice([0,10,20,30,40,50,60,70,80,90,100])
    pixelPercentage = random.choice([1,2,3,4,5,6,7,8,9])
    generateTextImages.generate_image_with_jpeg_pixelation(jpegPercentage, pixelPercentage, text, save_path)



def process(line, save_path):
    text = line.strip()
    generate_specific(text, save_path)

#read all lines in file and execute process
def read_in_file(path, save_path):
    with open(path) as f:
        for line in f:
            process(line, save_path)

#####
#####
##### execute when running
#####
#####



start = time.time()

#generate_random(10000, "images_emails_feature_extraction/")
read_in_file('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/email_dataset_small.txt', 'images_email/')
read_in_file('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/email_dataset.txt', 'images_emails_feature_extraction/')

end = time.time()
print(str(end - start) + " seconds")

#####
#####
#####
#####
#####





