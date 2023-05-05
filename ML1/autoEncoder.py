import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt
import time
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import string
import os
import glob
import textwrap
from random_word import RandomWords
import matplotlib.pyplot as plt
import essential_generators
import re

folder_path = "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor_zelf/"


# Define the font and font size
font = ImageFont.truetype("arial.ttf", 36)

#generate a image with jpeg, pixelation and input text
def generate_image_with_jpeg_pixelation(jpeg, pixelation, text, font_type, font_size, save_path, save_path_correct, save_correct_bool):
    
    #set font and font size
    font = ImageFont.truetype(font_type, size=font_size)

    # Determine the size of the text
    text_width, text_height = font.getsize(text)

    # Define the size of the image to be created based on the text size
    image_size = (text_width + 20, text_height + 20)

    # Create a new image with a white background
    image = Image.new("RGB", image_size, (255, 255, 255))

    # Create a drawing object and render the text in the specified font
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), text, font=font, fill=(0, 0, 0))

    if save_correct_bool == True:
        if not os.path.exists(folder_path + save_path_correct):
            os.makedirs(folder_path + save_path_correct)
        path2 = folder_path + save_path_correct
        cleaned_filename = re.sub(r'[<>:"/\\|?*]', '', text)
        cleaned_filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', cleaned_filename)
        cleaned_filename = re.sub(r'\s+', ' ', cleaned_filename) # Replace multiple spaces with a single space
        cleaned_filename = cleaned_filename.strip() # Remove leading/trailing spaces
        image.save(path2 + cleaned_filename + "__correct.jpg", format='JPEG', quality=jpeg)

    # Get the aspect ratio of the image
    aspect_ratio = float(image.width) / float(image.height)

    # Calculate the size of the pixelated image
    pixelated_width = int(image.width / pixelation)
    pixelated_height = int(pixelated_width / aspect_ratio)

    # Resize the image to the pixelated size
    image_small = image.resize((pixelated_width, pixelated_height), resample=Image.BOX)

    # Scale the pixelated image back up to the original size
    pixelated = image_small.resize((image.width, image.height), resample=Image.NEAREST)

    if not os.path.exists(folder_path + save_path):
        os.makedirs(folder_path + save_path)
    # Save the pixelated image
    cleaned_filename = re.sub(r'[<>:"/\\|?*]', '', text)
    cleaned_filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', '', cleaned_filename)
    cleaned_filename = re.sub(r'\s+', ' ', cleaned_filename) # Replace multiple spaces with a single space
    cleaned_filename = cleaned_filename.strip() # Remove leading/trailing spaces
    pixelated.save(folder_path + save_path + cleaned_filename + "__jpeg" + str(jpeg) + "__pix" + str(pixelation) + ".jpg", format='JPEG', quality=jpeg)

def read_in_file(path, save_path, save_path_correct, pixel, save_correct_bool):
    with open(path) as f:
        for line in f:
            text = line.strip()
            #jpegPercentage = random.choice([70,80,90,100])
            generate_image_with_jpeg_pixelation(100, pixel, text, "arial.ttf", 24, save_path, save_path_correct, save_correct_bool)

def generate_dataset():
    pixelation = [5, 6, 7, 8]

    for pix in pixelation:
        save_path = "dataset/images_emails_model_training_pix" + str(pix) + "/"
        save_path_correct = "dataset/images_emails_model_training_correct_pix" + str(pix) + "/"
        if pix == 5:
            save_correct_bool = True
        else:
            save_correct_bool = False
        read_in_file(folder_path + "email_dataset.txt", save_path, save_path_correct, pix, save_correct_bool)
    
    for pix in pixelation:
        save_path = "dataset/images_numbers_model_training_pix" + str(pix) + "/"
        save_path_correct = "dataset/images_numbers_model_training_correct_pix" + str(pix) + "/"
        if pix == 5:
            save_correct_bool = True
        else:
            save_correct_bool = False
        for i in range(25500):
            text = str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + " " + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + " " + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + " " + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9))
            generate_image_with_jpeg_pixelation(100, pix, text, "arial.ttf", 24, save_path, save_path_correct, save_correct_bool)

    


###########################################
###########################################
###########################################

def train_model_convolutional_layers(save_pathName, save_path_correctName, save_path_result, widthName, heightName):
    input_img = keras.Input(shape=(heightName, widthName, 1))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(input_img, decoded) 
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    ####
    collection = []
    labels = []
    save_path = folder_path + save_pathName
    save_path_correct = folder_path + save_path_correctName

    if not os.path.exists(save_path):
            os.makedirs(save_path)
    if not os.path.exists(save_path_correct):
            os.makedirs(save_path_correct)

    for filename in os.listdir(save_path):
            f = os.path.join(save_path, filename)
            # checking if it is a file
            if os.path.isfile(f):
                im = Image.open(f).convert("L")

                # Get current width
                width, height = im.size
                # Calculate the amount of padding needed on each side
                padding = (widthName - width) // 2
                # Create a new white image with the desired size
                new_size = (widthName, heightName)
                new_im = Image.new("L", new_size, color=255)
                # Paste the original image into the center of the new image
                new_im.paste(im, (padding, 0))

                #grayscale_image = im.convert("L")
                pix_val = new_im.getdata()
                
                collection.append(pix_val)
    
    for filename in os.listdir(save_path_correct):
            f = os.path.join(save_path_correct, filename)
            # checking if it is a file
            if os.path.isfile(f):
                im = Image.open(f).convert("L")

                # Get current width
                width, height = im.size
                # Calculate the amount of padding needed on each side
                padding = (widthName - width) // 2
                # Create a new white image with the desired size
                new_size = (widthName, heightName)
                new_im = Image.new("L", new_size, color=255)
                # Paste the original image into the center of the new image
                new_im.paste(im, (padding, 0))

                #grayscale_image = im.convert("L")
                pix_val = new_im.getdata()
                
                labels.append(pix_val)


    # create the x_train and x_test arrays
    x_train = np.asarray(collection[:15000], dtype=np.float32)
    x_test = np.asarray(collection[15000:20000], dtype=np.float32)
    y_train = np.asarray(labels[:15000], dtype=np.float32)
    y_test = np.asarray(labels[15000:20000], dtype=np.float32)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = y_train.astype('float32') / 255.
    y_test = y_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), heightName, widthName, 1))
    x_test = np.reshape(x_test, (len(x_test), heightName, widthName, 1))
    y_train = np.reshape(y_train, (len(y_train), heightName, widthName, 1))
    y_test = np.reshape(y_test, (len(y_test), heightName, widthName, 1))

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    autoencoder.fit(x_train, y_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test, y_test))
    
    # saving whole model
    autoencoder.save(save_path_result)


def use_model_convolutional_layers(autoencoderName, save_pathName, save_path_correctName, resultName, widthName, heightName):
    autoencoder = keras.models.load_model(folder_path + autoencoderName)
    
    email_adresses = []
    collection = []
    labels = []
    save_path = folder_path + save_pathName
    save_path_correct = folder_path + save_path_correctName

    

    files = sorted(os.listdir(save_path))
    files_correct = sorted(os.listdir(save_path_correct))

    last_files = files[-100:]
    last_files2 = files_correct[-100:]

    print(len(last_files))
    print(len(last_files2))

    print("Reading in files..\n")
    for filename in last_files:
            f = os.path.join(save_path, filename)
            # checking if it is a file
            if os.path.isfile(f):
                im = Image.open(f).convert("L")

                email_adres = filename.split("__")[0]
                email_adresses.append(email_adres)

                # Get current width
                width, height = im.size
                # Calculate the amount of padding needed on each side
                padding = (widthName - width) // 2
                # Create a new white image with the desired size
                new_size = (widthName, heightName)
                new_im = Image.new("L", new_size, color=255)
                # Paste the original image into the center of the new image
                new_im.paste(im, (padding, 0))

                #grayscale_image = im.convert("L")
                pix_val = new_im.getdata()
                
                collection.append(pix_val)

    for filename in last_files2:
            f = os.path.join(save_path_correct, filename)
            # checking if it is a file
            if os.path.isfile(f):
                im = Image.open(f).convert("L")

                # Get current width
                width, height = im.size
                # Calculate the amount of padding needed on each side
                padding = (widthName - width) // 2
                # Create a new white image with the desired size
                new_size = (widthName, heightName)
                new_im = Image.new("L", new_size, color=255)
                # Paste the original image into the center of the new image
                new_im.paste(im, (padding, 0))

                #grayscale_image = im.convert("L")
                pix_val = new_im.getdata()
                
                labels.append(pix_val)
    
    # create the x_train and x_test arrays
    x_depix = np.asarray(collection, dtype=np.float32)
    x_depix = x_depix.astype('float32') / 255.
    x_depix = np.reshape(x_depix, (len(x_depix), heightName, widthName, 1))

    x_correct = np.asarray(labels, dtype=np.float32)
    x_correct = x_correct.astype('float32') / 255.
    x_correct = np.reshape(x_correct, (len(x_correct), heightName, widthName, 1))

    print("Depixizing files..\n")
    print(x_depix.size)
    decoded_imgs = autoencoder.predict(x_depix)

    for i in range(100):

        # Display the reconstructed image
        img1 = x_depix[i].reshape((heightName, widthName))
        img1 = Image.fromarray((img1 * 255).astype('uint8'), mode='L')
        #img1.save('image.blurred.png')
        #img1.show()

        img2 = decoded_imgs[i].reshape((heightName, widthName))
        img2 = Image.fromarray((img2 * 255).astype('uint8'), mode='L')
        #img2.save('image.reconstruct.png')
        #img2.show()
 
        img3 = x_correct[i].reshape((heightName, widthName))
        img3 = Image.fromarray((img3 * 255).astype('uint8'), mode='L')
        #img3.save('image_original.png')
        #img3.show()

        def get_concat_v(im1, im2, im3):
            dst = Image.new('L', (im1.width, im1.height + im2.height + im3.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (0, im1.height))
            dst.paste(im3, (0, im1.height + im2.height))
            return dst

        if not os.path.exists(folder_path + resultName):
            os.makedirs(folder_path + resultName)
        get_concat_v(img1, img2, img3).save(folder_path + resultName + str(email_adresses[i]) + ".jpg")

def use_model_convolutional_layers_one(autoencoderName, save_pathName, resultName, widthName, heightName):
    autoencoder = keras.models.load_model(folder_path + autoencoderName)

    f = os.path.join(folder_path, save_pathName)
    # checking if it is a file
    if os.path.isfile(f):
        im = Image.open(f).convert("L")
        email_adres = save_pathName.split("__")[0]

        # Get current width
        width, height = im.size
        # Calculate the amount of padding needed on each side
        padding = (widthName - width) // 2
        # Create a new white image with the desired size
        new_size = (widthName, heightName)
        new_im = Image.new("L", new_size, color=255)
        # Paste the original image into the center of the new image
        new_im.paste(im, (padding, 0))
        #grayscale_image = im.convert("L")
        pix_val = new_im.getdata()          
    
    # create the x_train and x_test arrays
    x_depix = np.asarray(pix_val, dtype=np.float32)
    x_depix = x_depix.astype('float32') / 255.
    x_depix = np.reshape(x_depix, (len(x_depix), heightName, widthName, 1))

    print("Depixizing file..\n")
    print(x_depix.size)
    decoded_imgs = autoencoder.predict(x_depix)

    # Display the reconstructed image
    img1 = x_depix.reshape((heightName, widthName))
    img1 = Image.fromarray((img1 * 255).astype('uint8'), mode='L')

    img2 = decoded_imgs.reshape((heightName, widthName))
    img2 = Image.fromarray((img2 * 255).astype('uint8'), mode='L')

    def get_concat_v(im1, im2):
        dst = Image.new('L', (im1.width, im1.height + im2.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (0, im1.height))
        return dst

    if not os.path.exists(folder_path + resultName):
        os.makedirs(folder_path + resultName)
    get_concat_v(img1, img2).save(folder_path + resultName + str(email_adres) + ".jpg")


#####################
########main#########
#####################
#dont forget to enter folder path at the top
#example fonts: "C:/Windows/Fonts/OCRAEXT.ttf", "arial.ttf"

start = time.time()

"""train_model_convolutional_layers("dataset/images_numbers_model_training_pix7/",
                                  "dataset/images_numbers_model_training_correct_pix7/",
                                  "trained_models/autoencoder_model_convolutional_numbers_pix7.h5",
                                   308,
                                   44)"""

"""use_model_convolutional_layers("trained_models/autoencoder_model_convolutional_numbers_pix7.h5",
                               "dataset/images_numbers_model_training_pix7/",
                               "dataset/images_numbers_model_training_correct_pix7/",
                               "results_convolutional/numbers_pix_7/",
                                308,
                                44)"""



"""print("start dataset generation 6 pix")
try:
    read_in_file('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor_zelf/dataset/email_dataset.txt', "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor_zelf/dataset/images_emails_model_training_pix6/", "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_emails_model_training_correct_pix6/", 6)
except:
    c = 0
end = time.time()
print(str(end - start) + " seconds")"""
count = 0
while count < 25100:
    try:
        start2 = time.time()
        
        def clean_sentence(sentence):
            cleaned_sentence = ''.join(c for c in sentence if c.isalnum() or c.isspace() or c in string.punctuation)
            return cleaned_sentence.strip()

        gen = essential_generators.DocumentGenerator()

        while True:
            sentence = gen.sentence()
            if len(sentence) > 100 or any(ord(c) >= 128 for c in sentence):
                continue
            cleaned_sentence = clean_sentence(sentence)
            if len(cleaned_sentence) > 0:
                break

        print(sentence + "   " + str(len(sentence)))
        generate_image_with_jpeg_pixelation(100, 5, sentence, "arial.ttf", 24, "dataset/images_sentences_model_training_pix5/", "dataset/images_sentences_model_training_correct_pix5/", True)
        end2 = time.time()
        print(str(end2 - start2) + " seconds")
        count += 1
    except:
        pass

train_model_convolutional_layers("dataset/images_sentences_model_training_pix5/",
                                  "dataset/images_sentences_model_training_correct_pix5/",
                                  "trained_models/autoencoder_model_convolutional_sentences_pix5.h5",
                                   1200,
                                   48)


end = time.time()
print(str(end - start) + " seconds")
#####################
#####################