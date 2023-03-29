from PIL import Image, ImageDraw, ImageFont
import random
import string
import os
import glob
import functions


# Define the font and font size
font = ImageFont.truetype("arial.ttf", 50)

#alter image with jpeg resolution
def JPEG_compression(path, percentage):
    # Open the image
    image = Image.open(path)
    substring = path.split(".jpg")[0]  # Select the second substring (index 1)
    new_name = substring + "__jpeg" + str(percentage) + ".jpg"
    # Compress and save the image as JPEG with quality=90
    image.save(new_name, format='JPEG', quality=percentage)


#alter image with pixelation function
def pixelate(path, pixelation_amount):
    # Open the image
    image = Image.open(path)

    # Get the aspect ratio of the image
    aspect_ratio = float(image.width) / float(image.height)

    # Calculate the size of the pixelated image
    pixelated_width = int(image.width / pixelation_amount)
    pixelated_height = int(pixelated_width / aspect_ratio)

    # Resize the image to the pixelated size
    image_small = image.resize((pixelated_width, pixelated_height), resample=Image.BOX)

    # Scale the pixelated image back up to the original size
    pixelated = image_small.resize((image.width, image.height), resample=Image.NEAREST)

    # Save the pixelated image
    substring = path.split(".jpg")[0]  
    new_name = substring + "__pix" + str(pixelation_amount) + ".jpg"
    pixelated.save(new_name, format='JPEG', quality=100)


#generate a image based on input text
def generate_image(text, save_path):

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
    image.save(save_path + text + ".jpg", "JPEG")    


#generate a image with jpeg, pixelation and input text
def generate_image_with_jpeg_pixelation(jpeg, pixelation, text, save_path):
    
    # Determine the size of the text
    text_width, text_height = font.getsize(text)

    # Define the size of the image to be created based on the text size
    image_size = (text_width + 20, text_height + 20)

    # Create a new image with a white background
    image = Image.new("RGB", image_size, (255, 255, 255))

    # Create a drawing object and render the text in the specified font
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, font=font, fill=(0, 0, 0))

    # Get the aspect ratio of the image
    aspect_ratio = float(image.width) / float(image.height)

    # Calculate the size of the pixelated image
    pixelated_width = int(image.width / pixelation)
    pixelated_height = int(pixelated_width / aspect_ratio)

    # Resize the image to the pixelated size
    image_small = image.resize((pixelated_width, pixelated_height), resample=Image.BOX)

    # Scale the pixelated image back up to the original size
    pixelated = image_small.resize((image.width, image.height), resample=Image.NEAREST)

    # Save the pixelated image
    pixelated.save(save_path + text + "__jpeg" + str(jpeg) + "__pix" + str(pixelation) + ".jpg", format='JPEG', quality=jpeg)
