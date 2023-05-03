from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import string
import os
import glob
import functions
import textwrap


# Define the font and font size
font = ImageFont.truetype("arial.ttf", 36)

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

def gaussian_filter(path, kernel):
    # Open the image
    image = Image.open(path)
    substring = path.split(".jpg")[0]  # Select the second substring (index 1)
    new_name = substring + "__gaus" + str(kernel) + ".jpg"
    image = image.filter(ImageFilter.GaussianBlur(radius=kernel))
    # Compress and save the image as JPEG with quality=100
    image.save(new_name, format='JPEG', quality=100)

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
def generate_image_with_jpeg_pixelation(jpeg, pixelation, text, save_path, save_path_correct):
    
    #set font and font size
    font = ImageFont.truetype("arial.ttf", size=24)

    # Determine the size of the text
    text_width, text_height = font.getsize(text)

    # Define the size of the image to be created based on the text size
    image_size = (text_width + 20, text_height + 20)

    # Create a new image with a white background
    image = Image.new("RGB", image_size, (255, 255, 255))

    # Create a drawing object and render the text in the specified font
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, font=font, fill=(0, 0, 0))

    path2 = save_path_correct
    image.save(path2 + text + "__correct.jpg", format='JPEG', quality=jpeg)

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

#generate a image with jpeg, pixelation and input text
def generate_image_with_jpeg_gaussian(jpeg, gaussian, text, save_path):
    
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

    image = image.filter(ImageFilter.GaussianBlur(radius=gaussian))

    # Save the pixelated image
    image.save(save_path + text + "__jpeg" + str(jpeg) + "__gais" + str(gaussian) + ".jpg", format='JPEG', quality=jpeg)


def generate_lookup2(font, font_size, pixelation):
    with open('sequence2.txt', 'r') as file:
        text = file.read()

    font = ImageFont.truetype(font, size=font_size)

    image_size = font.getmask(text).size

    lst = list(image_size)
    lst[1] = lst[1] + 10
    lst[0] = lst[0] + 11
    image_size = tuple(lst)

    print(image_size)

    img = Image.new("RGB", image_size, (255,255,255))

    draw = ImageDraw.Draw(img)
    draw_point = (5, 2)

    draw.multiline_text(draw_point, text, font=font, fill=(0,0,0))

    text_window = img.getbbox()
    img = img.crop(text_window)

    # Save the pixelated image
    img.save("sequence_" + "arial" + "_" + str(font_size) + ".jpg", format='JPEG', quality=100)

#generate a image with jpeg, pixelation and input text
def generate_image_with_jpeg_pixelation_random_crops(jpeg, pixelation, text, label, save_path):
    
    #set font and font size
    font = ImageFont.truetype("arial.ttf", size=56)

    # Determine the size of the text
    text_width, text_height = font.getsize(text)

    # Define the size of the image to be created based on the text size
    #image_size = (text_width + 20, text_height + 20)
    image_size = (244, 244)

    # Create a new image with a white background
    image = Image.new("RGB", image_size, (255, 255, 255))

    # Create a drawing object and render the text in the specified font
    draw = ImageDraw.Draw(image)
    x0 = int((100-text_width)/2)
    y0 = int(((100-text_height)/2))
    draw.text((x0,y0), text, font=font, fill=(0, 0, 0), align='center')

    # Get the aspect ratio of the image
    aspect_ratio = float(image.width) / float(image.height)
    # Calculate the size of the pixelated image
    pixelated_width = int(image.width / pixelation)
    pixelated_height = int(pixelated_width / aspect_ratio)
    # Resize the image to the pixelated size
    image_small = image.resize((pixelated_width, pixelated_height), resample=Image.BOX)

    # Scale the pixelated image back up to the original size
    pixelated = image_small.resize((image.width, image.height), resample=Image.NEAREST)

    #box = (40, 40, 40+25, 40+25)
    #cropped_img = pixelated.crop(box)
    # Scale the pixelated image back up to the original size
    #final_img = cropped_img.resize((244, 244), resample=Image.NEAREST)

    pixelated.save(save_path + label + "__" + text + "__jpeg" + str(jpeg) + "__pix" + str(pixelation) + ".jpg", format='JPEG', quality=jpeg)


    # Save the pixelated image
    """rang = int((pixelated.width-12)/2)
    for i in range(rang):
        if i > 26:
            if i % 5 == 0 and i < 40:  
                distance = ((pixelated_width-i)-(i))
                crop_distance = (pixelated_height-distance)/2

                box = (0+(i), 0+crop_distance, pixelated.width-(i), pixelated.height-crop_distance)
                cropped_img = pixelated.crop(box)
            
                cropped_img.save(save_path + label + "__" + text + "__jpeg" + str(jpeg) + "__pix" + str(pixelation) + "__crop" + str(i) + ".jpg", format='JPEG', quality=jpeg)
                """
    
#generate a image with jpeg, pixelation and input text
def generate_image_with_jpeg_pixelation_random_crops2(jpeg, pixelation, text, label, save_path):
    
    pixelation = pixelation  * 8.5

    #set font and font size
    font = ImageFont.truetype("arial.ttf", size=240)

    # Determine the size of the text
    text_width, text_height = font.getsize(text)

    # Define the size of the image to be created based on the text size
    #image_size = (text_width + 20, text_height + 20)
    image_size = (976, 244)

    # Create a new image with a white background
    image = Image.new("RGB", image_size, (255, 255, 255))

    # Create a drawing object and render the text in the specified font
    draw = ImageDraw.Draw(image)
    x0 = 0 #int((732-text_width)/2)
    y0 = 0 # int(((244-text_height)/2))
    draw.text((x0,y0), text, font=font, fill=(0, 0, 0), align='center')

    # Get the aspect ratio of the image
    aspect_ratio = float(image.width) / float(image.height)
    # Calculate the size of the pixelated image
    pixelated_width = int(image.width / pixelation)
    pixelated_height = int(pixelated_width / aspect_ratio)
    # Resize the image to the pixelated size
    image_small = image.resize((pixelated_width, pixelated_height), resample=Image.BOX)

    # Scale the pixelated image back up to the original size
    pixelated = image_small.resize((image.width, image.height), resample=Image.NEAREST)

    #box = (40, 40, 40+25, 40+25)
    #cropped_img = pixelated.crop(box)
    # Scale the pixelated image back up to the original size
    #final_img = cropped_img.resize((244, 244), resample=Image.NEAREST)

    pixelated.save(save_path + label + "__" + text + "__jpeg" + str(jpeg) + "__pix" + str(pixelation) + ".jpg", format='JPEG', quality=jpeg)
