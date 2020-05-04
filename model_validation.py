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

def getRandomTrainImages(test_dir):
    """
    Get 26 random images to test out
    """
    image_labels = []
    no_of_images = 200

    x = 0
    image_array = np.zeros((26, 64, 64, 3), dtype=np.float32) # 200x200 RGB images
    for category in dispatcher.CATEGORIES:
        path = os.path.join(test_dir, category)

        images = os.listdir(path)
        # Choose a random index for a random image
        index = random.randint(0, no_of_images)
        image_name = os.path.join(path, images[index])
        # Get letter from image name
        label = image_name[image_name.rfind("\\") + 1]
        # Append image name and data.
        image_array[x] = mpimg.imread(image_name)
        image_labels.append(dispatcher.CATEGORIES.index(label))
        x += 1

    return image_array, np.array(image_labels)

if __name__ == '__main__':
    test_dir = sys.argv[1]
    # model_dir = sys.argv[2]

    # Get random images and labels
    (test_images, test_labels) = getRandomTrainImages(test_dir)

    test_images = test_images / 255.0

    """
    plt.figure()
    plt.imshow(test_images[0])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    """

    plt.figure(figsize=(10,10))
    for i in range(26):
        plt.subplot(5,6,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel(dispatcher.CATEGORIES[test_labels[i]])
    plt.show()

    
    # Load the model data
    # model = keras.models.load_model(model_dir)





    import matplotlib.pyplot as plt
