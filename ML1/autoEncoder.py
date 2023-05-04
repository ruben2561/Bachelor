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


folder_path = "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/"


# Define the font and font size
font = ImageFont.truetype("arial.ttf", 36)

#generate a image with jpeg, pixelation and input text
def generate_image_with_jpeg_pixelation(jpeg, pixelation, text, save_path, save_path_correct):
    
    #set font and font size
    font = ImageFont.truetype("C:/Windows/Fonts/OCRAEXT.ttf", size=24)

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

def read_in_file(path, save_path, save_path_correct, pixel):
    with open(path) as f:
        for line in f:
            text = line.strip()
            #jpegPercentage = random.choice([70,80,90,100])
            generate_image_with_jpeg_pixelation(100, pixel, text, save_path, save_path_correct)

def train_model_connected_layers():
    # This is the size of our encoded representations
    encoding_dim = 256 # changed to 2784 was 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # This is our input image
    input_img = keras.Input(shape=(68400,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(68400, activation='sigmoid')(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)

    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)

    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='mse')

    ####
    collection = []
    labels = []
    collection2 = []
    save_path = 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_emails_feature_extraction/'
    save_path2 = 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_email/'
    for filename in os.listdir(save_path):
            f = os.path.join(save_path, filename)
            # checking if it is a file
            if os.path.isfile(f):
                im = Image.open(f).convert("L")
                label = filename.split("__")[0]

                # Get current width
                width, height = im.size
                # Calculate the amount of padding needed on each side
                padding = (900 - width) // 2
                # Create a new white image with the desired size
                new_size = (900, height)
                new_im = Image.new("L", new_size, color=255)
                # Paste the original image into the center of the new image
                new_im.paste(im, (padding, 0))

                #grayscale_image = im.convert("L")
                pix_val = new_im.getdata()
                
                labels.append(label)
                collection.append(pix_val)

    k = 0
    for filename in os.listdir(save_path2):
            if k < 20:
                f = os.path.join(save_path2, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    im = Image.open(f).convert("L")

                    # Get current width
                    width, height = im.size
                    # Calculate the amount of padding needed on each side
                    padding = (900 - width) // 2
                    # Create a new white image with the desired size
                    new_size = (900, height)
                    new_im = Image.new("L", new_size, color=255)
                    # Paste the original image into the center of the new image
                    new_im.paste(im, (padding, 0))

                    #grayscale_image = im.convert("L")
                    pix_val = new_im.getdata()
                    
                    collection2.append(pix_val)
                    k += 1
    

    # create the x_train and x_test arrays
    x_train = np.asarray(collection[:10000], dtype=np.float32)
    x_test = np.asarray(collection[10000:15000], dtype=np.float32)
    x2_test = np.asarray(collection2, dtype=np.float32)


    ####

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x2_test = x2_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    x2_test = x2_test.reshape((len(x2_test), np.prod(x2_test.shape[1:])))
    
    print(x_train.shape)
    print(x_test.shape)
    print(x2_test.shape)

    print("start fitting")
    autoencoder.fit(x_train, x_train,
                    epochs=50, #was 50
                    batch_size=256,
                    shuffle=True,
                    validation_data=(x_test, x_test))

    # saving in json format
    json_model = autoencoder.to_json()
    json_file = open('autoencoder_json.json', 'w')
    json_file.write(json_model)


    # saving whole model
    autoencoder.save('autoencoder_model.h5')


def use_model_connected_layers():
    encoder = keras.models.load_model('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/autoencoder_model.h5')
    #encoded_imgs = encoder.predict(x_test)
    #decoded_imgs = decoder.predict(encoded_imgs)
    # This is the size of our encoded representations
    encoding_dim = 256 # changed to 2784 was 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # This is our input image
    input_img = keras.Input(shape=(68400,))
    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(68400, activation='sigmoid')(encoded)
    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)
    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)
    # This is our encoded (32-dimensional) input
    encoded_input = keras.Input(shape=(encoding_dim,))
    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))
    autoencoder.compile(optimizer='adam', loss='mse')


    collection = []
    save_path = 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_email/'

    k = 0
    for filename in os.listdir(save_path):
            if k < 20:
                f = os.path.join(save_path, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    im = Image.open(f).convert("L")
                    # Get current width
                    width, height = im.size
                    # Calculate the amount of padding needed on each side
                    padding = (900 - width) // 2
                    # Create a new white image with the desired size
                    new_size = (900, height)
                    new_im = Image.new("L", new_size, color=255)
                    # Paste the original image into the center of the new image
                    new_im.paste(im, (padding, 0))
                    #grayscale_image = im.convert("L")
                    pix_val = new_im.getdata()
                    collection.append(pix_val)
                    k += 1
    

    # create the x_train and x_test arrays
    x_test = np.asarray(collection, dtype=np.float32)
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    autoencoder = keras.models.load_model('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/autoencoder_model.h5')
    encoded_imgs = encoder.predict(x_test)
    print(encoded_imgs[0])
    decoded_imgs = decoder.predict(encoded_imgs)
    print("\n\n")
    print(decoded_imgs[0])

    # Use Matplotlib (don't ask)
    import matplotlib.pyplot as plt

    for i in range(1):

        """# Display the reconstructed image
        img = x_test[i].reshape((76, 900))
        img = Image.fromarray((img * 255).astype('uint8'), mode='L')
        img.save('image_original.png')
        img.show()"""

        img = decoded_imgs[i].reshape((76, 900))
        img = Image.fromarray((img * 255).astype('uint8'), mode='L')
        img.save('image.reconstruct.png')
        img.show()

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

start = time.time()

"""train_model_convolutional_layers("dataset/images_numbers_model_training_pix7/",
                                  "dataset/images_numbers_model_training_correct_pix7/",
                                  "autoencoder_model_convolutional_numbers_pix7.h5",
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
    read_in_file('C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/email_dataset.txt', "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_emails_model_training_pix6/", "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_emails_model_training_correct_pix6/", 6)
except:
    c = 0
end = time.time()
print(str(end - start) + " seconds")
start = time.time()
print("end dataset generation pix6" + "\n")"""

"""print("start model generation pix6")
try:
    train_model_convolutional_layers(6)
except:
    c = 0
end = time.time()
print(str(end - start) + " seconds")
start = time.time()
print("end model generation pix6" + "\n")"""

"""print("start model testing pix6")
try:
    use_model_convolutional_layers(6)
except:
    c = 0

end = time.time()
print(str(end - start) + " seconds")
start = time.time()
print("end model testing pix6" + "\n")"""

"""for i in range(25500):
    text = str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + " " + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + " " + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + " " + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9)) + str(random.randint(0,9))
    generate_image_with_jpeg_pixelation(100, 7, text, 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_numbers_model_training_pix7/', 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_numbers_model_training_correct_pix7/')"""

"""print("start words generation pix5")
for i in range(10000):
        if i < 10:
            generate_image_with_jpeg_pixelation(100, 5, "000" + str(i), 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_numbers_model_training_pix5/', 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_numbers_model_training_correct_pix5/')
        elif i < 100:
            generate_image_with_jpeg_pixelation(100, 5, "00" + str(i), 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_numbers_model_training_pix5/', 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_numbers_model_training_correct_pix5/')
        elif i < 1000:
             generate_image_with_jpeg_pixelation(100, 5, "0" + str(i), 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_numbers_model_training_pix5/', 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_numbers_model_training_correct_pix5/')
        else:
            generate_image_with_jpeg_pixelation(100, 5, str(i), 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_numbers_model_training_pix5/', 'C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/images_numbers_model_training_correct_pix5/')
    

end = time.time()
print(str(end - start) + " seconds")
print("end words generation pix5" + "\n")"""



#####################
#####################