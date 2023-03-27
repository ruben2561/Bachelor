from PIL import Image, ImageDraw, ImageFont
import random
import string
import os
import glob
import functions


# Define the font and font size
font = ImageFont.truetype("arial.ttf", 50)

def generate_image(text):
    #text = line.strip('\n')

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


