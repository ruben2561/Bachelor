import random
import string
from PIL import Image
from random_word import RandomWords
import numpy as np

def generate_email():
    letters = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.ascii_lowercase + string.ascii_lowercase + string.ascii_lowercase + string.ascii_lowercase + string.ascii_lowercase
    only_letters = string.ascii_lowercase + string.ascii_uppercase + string.ascii_lowercase + string.ascii_lowercase + string.ascii_lowercase
    username = ''.join(random.choice(letters) for i in range(random.randint(4, 10)))
    username2 = ''.join(random.choice(letters) for i in range(random.randint(4, 10)))
    domain_name = ''.join(random.choice(letters) for i in range(random.randint(5, 10)))
    domain_extension = ''.join(random.choice(only_letters) for i in range(random.randint(2, 3)))
    domain = domain_name + '.' + domain_extension
    email = username + random.choice(['.', '_', '']) + username + '@' + domain
    return email

def generate_word():
    r = RandomWords()
    return r.get_random_word()

def JPEG_compression(path, percentage):
    # Open the image
    image = Image.open(path)
    substring = path.split(".jpg")[0]  # Select the second substring (index 1)
    new_name = substring + "__jpeg" + str(percentage) + ".jpg"
    # Compress and save the image as JPEG with quality=90
    image.save(new_name, format='JPEG', quality=percentage)

#define our sick pixelation function
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

