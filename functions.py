import random
import string
from PIL import Image
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

def JPEG_compression(path, percentage):
    # Open the image
    image = Image.open(path)
    substring = path.split(".jpg")[0]  # Select the second substring (index 1)
    new_name = substring + "__jpeg" + str(percentage) + ".jpg"
    # Compress and save the image as JPEG with quality=90
    image.save(new_name, format='JPEG', quality=percentage)

#define our sick pixelation function
def pixelate(path, pixelation_amount):
    image = Image.open(path)
    width, height = image.size
    pixel_size = pixelation_amount

    # Calculate the number of pixel blocks horizontally and vertically
    num_horiz_blocks = (width + pixel_size - 1) // pixel_size
    num_vert_blocks = (height + pixel_size - 1) // pixel_size

    # Resize the image to the pixelated size
    image_tiny = image.resize((num_horiz_blocks, num_vert_blocks), Image.NEAREST)

    # Resize the pixelated image back to the original size
    pixelated = image_tiny.resize((width, height), Image.NEAREST)

    # Save the pixelated image
    substring = path.split(".jpg")[0]  
    new_name = substring + "__pix" + str(pixelation_amount) + ".jpg"
    pixelated.save(new_name, format='JPEG', quality=100)

