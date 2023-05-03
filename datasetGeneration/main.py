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
from string import ascii_lowercase as alc
import numpy as np


def remove_previous(path):
    #remove previous generated files in folder
    files = glob.glob(path)
    for f in files:
        os.remove(f)

def generate_random_pixelated(amount, save_path):
    for i in range(amount):
        jpegPercentage = random.choice([0,10,20,30,40,50,60,70,80,90,100])
        pixelPercentage = random.choice([1,2,3,4,5,6,7,8,9])
        text = functions.generate_word()
        generateTextImages.generate_image_with_jpeg_pixelation(jpegPercentage, pixelPercentage, text, save_path)

def generate_random_blurred(amount, save_path):
    for i in range(amount):
        jpegPercentage = random.choice([0,10,20,30,40,50,60,70,80,90,100])
        gaussianKernal = random.choice([1,2,3,4,5,6,7])
        text = functions.generate_word()
        generateTextImages.generate_image_with_jpeg_gaussian(jpegPercentage, gaussianKernal, text, save_path)

def generate_temp(amount, save_path):
    for i in alc:
        newpath = r'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/' + "test" 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        text = i
        for k in range(3):
            save_pathh = 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/' + "test" + "/"

            jpegPercentage = random.choice([0,10,20,30,40,50,60,70,80,90,100])
            gaussianKernal = random.choice([1,2,3,4,5,6,7,8])
            pixelPercentage = random.choice([1,2,3,4,5,6,7,8,9,10])
            generateTextImages.generate_image_with_jpeg_pixelation(jpegPercentage, pixelPercentage, i, save_pathh)
            generateTextImages.generate_image_with_jpeg_gaussian(jpegPercentage, gaussianKernal, i, save_pathh)

        
def generate_specific_word(text, save_path):
    jpegPercentage = random.choice([0,10,20,30,40,50,60,70,80,90,100])
    pixelPercentage = random.choice([1,2,3,4,5,6,7,8,9])
    generateTextImages.generate_image_with_jpeg_pixelation(jpegPercentage, pixelPercentage, text, save_path)

def generate_specific_word_and_values(text, save_path, jpegPercentage, pixelPercentage):
    jpegPercentage2 = random.choice([0,10,20,30,40,50,60,70,80,90,100])
    #pixelPercentage2 = random.choice([1,2,3,4,5,6,7,8,9])
    generateTextImages.generate_image_with_jpeg_pixelation(jpegPercentage2, pixelPercentage, text, save_path)

def generate_all_numbers(save_path, quantity, pixel):
    for i in range(quantity):
        if i < 10:
            generate_specific_word_and_values("000" + str(i), save_path, 100, pixel)
        elif i < 100:
            generate_specific_word_and_values("00" + str(i), save_path, 100, pixel)
        elif i < 1000:
             generate_specific_word_and_values("0" + str(i), save_path, 100, pixel)
        else:
            generate_specific_word_and_values(str(i), save_path, 100, pixel)

def generate_word_with_font_size_pixelation(fontt, sizee, text):
    font = ImageFont.truetype(fontt, size=sizee)

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
    img.save("testim_arial_24" + ".jpg", format='JPEG', quality=100)

#read all lines in file and execute process
def read_in_file(path, save_path, save_path_correct, pixel):
    with open(path) as f:
        for line in f:
            text = line.strip()
            jpegPercentage = random.choice([70,80,90,100])
            generateTextImages.generate_image_with_jpeg_pixelation(100, pixel, text, save_path, save_path_correct)

def itterate_folder(save_path):
    for filename in os.listdir(save_path):
        f = os.path.join(save_path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            im = Image.open(f)
            if im.height != 47:
                print(filename)
            #functions.crop_mosaic_image(f)
            #if "cropped" not in f:
            #    os.remove(f)
                
def find_match(image_path, folder_path):
    collection = []
    for filename in os.listdir(folder_path):
        f = os.path.join(folder_path, filename)
        # checking if it is a file
        if os.path.isfile(f):
            im = Image.open(f)
            grayscale_image = im.convert("L")
            pix_val = list(grayscale_image.getdata())
            collection.append(pix_val)

    im = Image.open(image_path)
    grayscale_image = im.convert("L")
    number = list(grayscale_image.getdata())

    word = functions.find_best_match(number, collection)
    print(word)

def de_bruijn_2(n):
    # Generate the alphabet of 62 characters
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    k = len(alphabet)
    a = [0] * (k * n)
    sequence = ''

    def db(t, p):
        nonlocal sequence
        if t > n:
            if n % p == 0:
                sequence += ''.join([alphabet[a[j]] for j in range(1, p + 1)])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)

    db(1, 1)
    sequence += sequence[0:n-1]
    return sequence

#not finished
def find_word(image_path, search_path):
    im = Image.open(image_path)
    grayscale_image = im.convert("L")
    pix_val = list(grayscale_image.getdata())
    size = len(pix_val)
    width1 = grayscale_image.width
    height1 = grayscale_image.height

    im2 = Image.open(search_path)
    grayscale_image2 = im2.convert("L")
    pix_val2 = list(grayscale_image2.getdata())
    size2 = len(pix_val2)
    width2 = grayscale_image2.width
    height2 = grayscale_image2.height

    x = 0
    y = 0
    size_width = 19
    size_height = 22

    for i in range(width1-size_width):
        best_distance = np.inf
        best_match = None

        assert x + size_width <= width1 and y + size_height <= height1, "The square region goes beyond the image boundaries"
        square_data = []
        for j in range(y, y+size_height):
            row = []
            for o in range(x, x+size_width):
                pixel_value = grayscale_image.getpixel((o,j))  # get the grayscale value of the pixel
                row.append(pixel_value)
            square_data.append(row)
        
        x2 = 0
        y2 = 3
        for k in range(54263-size_width+1):
            assert x2 + size_width <= width2 and y2 + size_height <= height2, "The square region goes beyond the image boundaries"
            square_data2 = []
            for m in range(y2, y2+size_height):
                row2 = []
                for l in range(x2, x2+size_width):
                    pixel_value2 = grayscale_image2.getpixel((l,m))  # get the grayscale value of the pixel
                    row2.append(pixel_value2)
                square_data2.append(row2)
            x2 += 1

            current_distance = functions.calculate_distance(square_data, square_data2)
            if current_distance < best_distance:
                best_distance = current_distance
                best_match = square_data2
                
        print(i)
        p = Image.fromarray(np.asarray(best_match).astype(np.uint8))
        p.save("results2/result" + str(i) + ".png")
        x +=4

    print(square_data)

def all_letter_combinations(start):
    jpegPercentage = [75, 80, 90, 100]
    pixelPercentage = [5]
    letters = ' ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    nummers = '0123456789'

    for i in range(0, len(nummers)):
        for j in range(0, len(nummers)):
            nummer1 = nummers[i]
            nummer2 = nummers[j]
            newpath = "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/nummers4/"  + nummer1 + nummer2
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            for jpeg in jpegPercentage:
                for pixel in pixelPercentage:
                    #for letter1 in nummers:
                     #   for letter2 in nummers:
                            text = str( nummer1 + nummer2)
                            label = nummer1 + nummer2
                            generateTextImages.generate_image_with_jpeg_pixelation_random_crops(jpeg, pixel, text, label, newpath+"/")
            print(str(i) + ": " + str(int(time.time()-start)))


    



#####
#####
##### execute when running
#####
#####



start = time.time()

#remove_previous('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_text_gaussian/*')
#generate_temp(1, "dataset/images_paper2/")
#generate_temp(1,"gg")
#read_in_file('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/email_dataset_small.txt', 'images_email/')
#read_in_file('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/email_dataset.txt', 'images_emails_feature_extraction/')
#generate_all_numbers("C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_all_numbers/", 10000, 8)
#read_in_file('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/email_dataset.txt', "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_emails_model_training_pix7/", "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_emails_model_training_correct_pix7/")

#functions.crop_mosaic_image("C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/0490__jpeg40__pix8.jpg")
#functions.get_gray_scale_vector("C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/12345__jpeg100__pix8__cropped.jpg", 8)
#generate_specific_word_and_values("0490", "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/", 100, 8)
#find_match('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/0490__jpeg40__pix8.jpg', 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_all_numbers')
#generateTextImages.generate_lookup2("arial.ttf", 24, 8)
#functions.crop_image_auto("output.png")
#generate_word_with_font_size_pixelation("arial.ttf", 24, "hello world")
#generateTextImages.pixelate("C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/testim_arial_24.jpg", 4)
#find_word("C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/testim_arial_24.jpg", "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/search_images/sequence_arial_24.jpg")
#itterate_folder( "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_emails_model_training_correct/")

#generateTextImages.generate_image_with_jpeg_pixelation_random_crops2(100, 5, "173894", "173894", "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/")
#all_letter_combinations(start)



end = time.time()
print(str(end - start) + " seconds")

#####
#####
#####
#####
#####





