from PIL import Image, ImageDraw, ImageFont
import random
import string
import os
import glob
import functions


#remove previous generated files in folder
files = glob.glob('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/text_images/*')
for f in files:
    os.remove(f)

# Define the font and font size
font = ImageFont.truetype("arial.ttf", 32)

def generate_image(line):
    # Define the text to be rendered
    #text = functions.generate_email()
    text = line.strip('\n')

    # Determine the size of the text
    text_width, text_height = font.getsize(text)

    # Define the size of the image to be created based on the text size
    image_size = (text_width + 20, text_height + 20)

    # Create a new image with a white background
    image = Image.new("RGB", image_size, (255, 255, 255))

    # Create a drawing object and render the text in the specified font
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, font=font, fill=(0, 0, 0))

    # Save the image as a JPEG file
    image.save("text_images/" + text + ".jpg", "JPEG")

    #values = range(6)
    #for i in values:
        #functions.JPEG_compression("text_images/" + text + ".jpg", i*20)
    #    functions.pixelate("text_images/" + text + ".jpg", i+1)    

def process(line):
    print("1. " + line)

with open('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/email_dataset.txt') as f:
    for line in f:
        generate_image(line)

