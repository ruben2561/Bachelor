import random
import string
from PIL import Image
from random_word import RandomWords
import numpy as np
import cv2
import statistics

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

#crop mosaiced image
def crop_mosaic_image_simple(save_path):
    im = Image.open(save_path)
    grayscale_image = im.convert("L")
    box = (10, 10, im.width-10, im.height-10)
    cropped_img = im.crop(box)
    
    substring = save_path.split(".jpg")[0]  # Select the second substring (index 1)
    cropped_img.save(substring + "__cropped.jpg")


#crop mosaiced image
def crop_mosaic_image(save_path):
    #open image
    im = Image.open(save_path)
    grayscale_image = im.convert("L")

    pix_val = list(grayscale_image.getdata())
    #pix_val_flat = [x for sets in pix_val for x in sets]
    width, height = im.size
    #print(pix_val)
    print(len(pix_val))
    print(width)
    print(height)
    print("----")

    x_0 = 0
    x_1 = width
    y_0 = 0
    y_1 = height

    for i in range(len(pix_val)):
        if pix_val[i] <= 250:
            if pix_val[i+1] <= 250 and pix_val[i+2] <= 250:
                print("value: " + str(i))
                y_0 = i//width
            break

    for i in range(len(pix_val)):
        if pix_val[len(pix_val) - i -1 ] <= 250:
            if pix_val[len(pix_val) - i-2] <= 250 and pix_val[len(pix_val) - i-3] <= 250:
                print("value: " + str(len(pix_val) - i-1))
                y_1 = (len(pix_val)-i-1)//width
            break

    for x in range(width):
        if x_0 == 0:
            for y in range(height):
                i = x + y*width   # calculate the index of the current pixel
                if pix_val[i] <= 250:
                    if pix_val[i+1] <= 250 and pix_val[i+2] <= 250:
                        x_0 = x
                        print(f"Pixel ({x}, {y}): {pix_val[i]}")
                    break

    for x in reversed(range(width)):
        if x_1 == width:
            for y in reversed(range(height)):
                i = x + y*width   # calculate the index of the current pixel
                i_2 = x + (y-1)*width   # calculate the index of the second pixel
                i_3 = x + (y-2)*width   # calculate the index of the third pixel
                if pix_val[i] <= 250:
                    if pix_val[i_2] <= 250 and pix_val[i_3] <= 250:
                        x_1 = x
                        print(f"Pixel ({x}, {y}): {pix_val[i]}")
                    break

    print('y_0: '+ str(y_0) + '   x_0:' + str(x_0))
    print('y_1: '+ str(y_1) + '   x_1:' + str(x_1))
    
    # Crop the image to the bounding box
    box = (x_0, y_0, x_1 + 1,  y_1 + 1)
    cropped_img = im.crop(box)
    
    substring = save_path.split(".jpg")[0]  # Select the second substring (index 1)
    cropped_img.save(substring + "__cropped.jpg")

def calculate_distance(list1, list2):
    """Calculate the Euclidean distance between two lists of grayscale values."""
    distance = np.sqrt(np.sum((np.array(list1) - np.array(list2))**2))
    return distance

def find_best_match(original_list, collection):
    """Find the list in a collection with the smallest distance to an original list."""
    best_distance = np.inf
    best_match = None
    counter = 0
    for candidate_list in collection:
        distance = calculate_distance(original_list, candidate_list)
        if distance < best_distance:
            best_distance = distance
            best_match = candidate_list
            print(counter)

        counter = counter + 1
    print("best match is: " + counter)
    return best_match

def crop_image_auto(save_path):
    image=Image.open(save_path)
    image.load()

    image_data = np.asarray(image)
    image_data_bw = 255 - image_data.max(axis=2)
    non_empty_columns = np.where(image_data_bw.max(axis=0)>0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1)>0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))

    image_data_new = image_data[cropBox[0]-5:cropBox[1]+6, cropBox[2]-5:cropBox[3]+6 , :]

    new_image = Image.fromarray(image_data_new)
    new_image.save('L_2d_cropped.png')

def display_grayscale_values(collection, width):
    i = 0
    string = ""
    for value in collection:
        if len(str(value)) == 3:
            string += " " + str(value) + " "
        elif len(str(value)) == 2:
            string += "  " + str(value) + " "
        elif len(str(value)) == 1:
            string += "  " + str(value) + "  "
        i = i + 1
        if (i-1)%width == 0:
            string += "\n"

    print(string)







 