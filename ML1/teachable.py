from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# arial.ttf
# 24

def function(image):
        i = 1
    

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("C:/Users/ruben/Downloads/converted_keras3/keras_Model.h5", compile=False)

# Load the labels
class_names = open("C:/Users/ruben/Downloads/converted_keras3/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

guesses = []

im = Image.open("C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/dataset/059684__059684__jpeg100__pix5__crop2.2.jpg")
for i in range((im.width//5)-4):
    x = i * 5
    box = (x, 0, x+(5*5), im.height)
    cropped_img = im.crop(box)

    # Replace this with the path to your image
    image = cropped_img.convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    #image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    #image = image.resize((224,224), resample=Image.NEAREST)
    # turn the image into a numpy array
    image_array = np.asarray(image)
    if len(guesses) < 1: image.show()
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    guesses.append(class_name)

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)

print(guesses)