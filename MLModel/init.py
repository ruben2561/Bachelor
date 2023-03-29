import hiddenMarkovModel
import extractAndClusterFeatures
import loadImages

directory = "C:/Users/ruben/Documents/school/school 2022-2023/bachelor/Bachelor/images_text/"
images, labels = loadImages.load_images(directory)
features = extractAndClusterFeatures.extract_and_return_features_for_trainig_data(images, labels)
hiddenMarkovModel.train_model("train_data.npy")
