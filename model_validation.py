"""
    ***ASSIGNED TO: OMAR***
    To be Reviewed: Friday, 5/8/2920


    Have the network go through ONLY the test set, displaying each image
    passed to the network, with its respective prediction. Use this to
    get a visual of the predictions on a dataset.

    TIP: Dont make it display only one image, Display a square of 25 images with there prediction.

    Arguments:
        1. full path to hdf5 file containing tensorflow model.
        2. Path to the test dataset ONLY.
"""
import dispatcher
import random
import matplotlib.image as mpimg # for reading images as numpy arrays
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import sys
import os

def getTrainImagesInOrder(test_dir):
    """
    Get one image from each of 26 types to test out
    """
    image_labels = []
    no_of_images = 200

    x = 0
    image_array = np.zeros((26, 64, 64, 3), dtype=np.float32) # 200x200 RGB images
    for category in dispatcher.CATEGORIES:
        path = os.path.join(test_dir, category)

        images = os.listdir(path)
        # Choose a random index for a random image
        rand_index = random.randint(0, no_of_images)
        rand_image_path = os.path.join(path, images[rand_index])
        # Get letter from image name
        label = rand_image_path[rand_image_path.rfind("\\") + 1]
        # Append image name and data.
        image_array[x] = mpimg.imread(rand_image_path)
        image_labels.append(dispatcher.CATEGORIES.index(label))
        x += 1

    return image_array, np.array(image_labels)

def getTrainImagesOutOfOrder(test_dir):
    """
    Get 26 random images
    """
    image_list = []
    image_labels = []
    image_array = np.zeros((26, 64, 64, 3), dtype=np.float32) # 200x200 RGB images
    no_of_images = 5200
    for category in dispatcher.CATEGORIES:
        path = os.path.join(test_dir, category)

        images = os.listdir(path)
        for img in images:
            image_list.append(os.path.join(path, img))

    for x in range(0, 26):
        rand_index = random.randint(0, no_of_images)
        image_array[x] = mpimg.imread(image_list[rand_index])
        label = image_list[rand_index][image_list[rand_index].rfind("\\") + 1]
        image_labels.append(dispatcher.CATEGORIES.index(label))

    return image_array, np.array(image_labels)

if __name__ == '__main__':
    test_di = sys.argv[1]
    model_dir = sys.argv[2]

    model = keras.models.load_model(model_dir)

    # Get random images and labels
    (test_images, test_labels) = getTrainImagesOutOfOrder(test_dir)
    test_images = test_images / 255.0

    # Get predictions of model
    predictions = model.predict(test_images)

    tally = 0
    plt.figure(figsize=(15,10))
    for i in range(26):
        plt.subplot(5,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        actual = dispatcher.CATEGORIES[test_labels[i]]
        predicted = dispatcher.CATEGORIES[np.argmax(predictions[i])]
        img_label = "Actual: " + actual + " Predicted: " + predicted
        if actual == predicted:
            tally += 1
        plt.xlabel(img_label)
    plt.show()

    print(str(tally) + " out of 26 correct")

