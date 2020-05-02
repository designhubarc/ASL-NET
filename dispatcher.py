import numpy as np
import matplotlib.image as mpimg # for reading images as numpy arrays
import os
import random
import sys


CATEGORIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
              'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

"""
    To train network we need to generate a random bucket of images to parse,
    one bucket at a time.
"""

"""
    Post the data base as a zip file to media fire or drive and then
    write a download tool to download and unzip the database into a user desired
    directory.
"""

class Dataset:

    def __init__(self, directory, batch_size):

        self.shape = (64,64,3) # 64 x 64 colored images
        self.classifications = 26

        # Directories for train, test, and validation images
        self.train_dir = os.path.join(directory, "train")
        self.test_dir = os.path.join(directory, "test")
        self.val_dir = os.path.join(directory, "val")

        # Default batch size
        self.batch_size = batch_size
        # total number of batches
        self.train_number_of_batches = 2600 // self.batch_size
        self.val_test_number_of_batches = 200 // self.batch_size

        # current number of batches processed
        self.current_train_batch = 0
        self.current_test_batch = 0
        self.current_val_batch = 0

        # current number of cycles through training dataset
        self.current_epoch = 0
        self.epoch_threshold = 65 # arbitrary value for now

        # Shuffled list of all randomized images for batch
        # These are numpy arrays that are not static
        self.train_image_array = self.get_shuffled_image_array(self.train_dir)
        self.test_image_array = self.get_shuffled_image_array(self.test_dir)
        self.val_image_array = self.get_shuffled_image_array(self.val_dir)


    def get_shuffled_image_array(self, image_dir):
        """
        Takes a directory (train, test, or val) and returns a shuffled numpy
        array containing every file name in that directory
        """
        image_list = []
        for category in CATEGORIES:
            path = os.path.join(image_dir, category)

            images = os.listdir(path)
            for img in images:
                image_list.append(os.path.join(path, img))

        random.shuffle(image_list)

        return np.array(image_list)


    """
        TO-DO:
        Inside asl_net.py, Make a CNN following the tensorflow tutorial

    """

    def generate_train_batch(self):
        """
        Returns two parallel numpy arrays: one for image data and one for
        indexes of CATEGORIES
        """

        batch_images = np.zeros((self.batch_size, self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32) # 200x200 RGB images
        batch_labels = []

        # Batch starts at the last batch's end, to make sure we do not reuse images
        start = self.current_train_batch * self.batch_size
        end = (self.current_train_batch+1) * self.batch_size

        index = 0
        for x in range(start, end):
            batch_images[index] = mpimg.imread(self.train_image_array[x])
            # Get letter from image name
            label = self.train_image_array[x][self.train_image_array[x].rfind("\\") + 1]
            # Append letter index
            batch_labels.append(CATEGORIES.index(label))
            index += 1

        #increment batch
        self.current_train_batch += 1
        # shuffle if needed
        if self.current_train_batch == self.train_number_of_batches:
            np.random.shuffle(self.train_image_array)
            self.current_train_batch = 0 # reset batch number
            self.current_epoch += 1 # new epoch

        return batch_images, np.array(batch_labels)

    def generate_test_batch(self):
        """
        Returns two parallel numpy arrays: one for image data and one for
        indexes of CATEGORIES
        """
        batch_images = np.zeros((self.batch_size, self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32) # 200x200 RGB images
        batch_labels = []

        # Batch starts at the last batch's end, to make sure we do not reuse images
        start = self.current_test_batch * self.batch_size
        end = (self.current_test_batch+1) * self.batch_size

        index = 0
        for x in range(start, end): # make sure we do not reuse images
            batch_images[index] = mpimg.imread(self.test_image_array[x])
            # Get letter from image name
            label = self.test_image_array[x][self.test_image_array[x].rfind("\\") + 1]
            # Append letter index
            batch_labels.append(CATEGORIES.index(label))
            index += 1

        # increment batch
        self.current_test_batch += 1
        # no shuffle
        return batch_images, np.array(batch_labels)

    def generate_val_batch(self):
        """
        Returns two parallel numpy arrays: one for image data and one for
        indexes of CATEGORIES
        """
        # Make np array of 200x200 RGB images, filled with zeros
        batch_images = np.zeros((self.batch_size, self.shape[0], self.shape[1], self.shape[2]), dtype=np.float32) # 200x200 RGB images
        batch_labels = []

        # Batch starts at the last batch's end, to make sure we do not reuse images
        start = self.current_val_batch * self.batch_size
        end = (self.current_val_batch+1) * self.batch_size

        index = 0
        for x in range(start, end): # make sure we do not reuse images
            batch_images[index] = mpimg.imread(self.val_image_array[x])
            # Get letter from image name
            label = self.val_image_array[x][self.val_image_array[x].rfind("\\") + 1]
            # Append letter index
            batch_labels.append(CATEGORIES.index(label))
            index += 1

        #increment batch
        self.current_val_batch += 1
        # shuffle if needed
        if self.current_val_batch == self.val_test_number_of_batches:
            np.random.shuffle(self.val_image_array)
            self.current_val_batch = 0

        return batch_images, np.array(batch_labels)


if __name__ == '__main__':
    # TESTING
    dataset_directory = str(sys.argv[1])
    dataset = Dataset(dataset_directory, 40)
    val_batch = dataset.generate_val_batch()
    test_batch = dataset.generate_test_batch()
    train_batch = dataset.generate_train_batch()
    # Output labels
    print(train_batch[1])
    print(test_batch[1])
    print(val_batch[1])
