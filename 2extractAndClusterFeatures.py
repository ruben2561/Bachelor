import os
import cv2
from skimage.feature import hog, local_binary_pattern
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def extract_features(path):
    # Define the HOG parameters
    orientations = 8
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)

    # Define the LBP parameters
    radius = 3
    n_points = 8 * radius

    # Initialize the feature vectors
    hog_features = []
    sift_features = []
    lbp_features = []

    # Loop over all images in the directory
    for filename in os.listdir(path):
        # Load the character image
        img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)

        # Extract HOG features
        hog_feature = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm='L2-Hys')
        hog_features.append(hog_feature)

        # Extract SIFT features
        sift = cv2.xfeatures2d.SIFT_create()
        kp, sift_feature = sift.detectAndCompute(img, None)
        sift_features.append(sift_feature)

        # Extract LBP features
        lbp_feature = local_binary_pattern(img, n_points, radius, method='uniform')
        lbp_feature = np.histogram(lbp_feature.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))[0]
        lbp_features.append(lbp_feature)

    # Scale the features
    scaler = StandardScaler()
    hog_features = scaler.fit_transform(np.array(hog_features))
    sift_features = scaler.fit_transform(np.array(sift_features).reshape(-1, 128))
    lbp_features = scaler.fit_transform(np.array(lbp_features))

    # Cluster the features using KMeans
    n_clusters = 32
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    hog_clusters = kmeans.fit_predict(hog_features)
    sift_clusters = kmeans.fit_predict(sift_features)
    lbp_clusters = kmeans.fit_predict(lbp_features)

    # Concatenate the feature clusters
    features = np.concatenate([hog_clusters.reshape(-1, 1), sift_clusters.reshape(-1, 1), lbp_clusters.reshape(-1, 1)], axis=1)

    return features


def extract_features(images):
    # Define the HOG parameters
    orientations = 8
    pixels_per_cell = (8, 8)
    cells_per_block = (2, 2)

    # Define the LBP parameters
    radius = 3
    n_points = 8 * radius

    # Initialize the feature vectors
    hog_features = []
    sift_features = []
    lbp_features = []

    # Loop over all images in the directory
    for img in images:

        # Extract HOG features
        hog_feature = hog(img, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm='L2-Hys')
        hog_features.append(hog_feature)

        # Extract SIFT features
        sift = cv2.xfeatures2d.SIFT_create()
        kp, sift_feature = sift.detectAndCompute(img, None)
        sift_features.append(sift_feature)

        # Extract LBP features
        lbp_feature = local_binary_pattern(img, n_points, radius, method='uniform')
        lbp_feature = np.histogram(lbp_feature.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))[0]
        lbp_features.append(lbp_feature)

    # Scale the features
    scaler = StandardScaler()
    hog_features = scaler.fit_transform(np.array(hog_features))
    sift_features = scaler.fit_transform(np.array(sift_features).reshape(-1, 128))
    lbp_features = scaler.fit_transform(np.array(lbp_features))

    # Cluster the features using KMeans
    n_clusters = 32
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    hog_clusters = kmeans.fit_predict(hog_features)
    sift_clusters = kmeans.fit_predict(sift_features)
    lbp_clusters = kmeans.fit_predict(lbp_features)

    # Concatenate the feature clusters
    features = np.concatenate([hog_clusters.reshape(-1, 1), sift_clusters.reshape(-1, 1), lbp_clusters.reshape(-1, 1)], axis=1)

    return features