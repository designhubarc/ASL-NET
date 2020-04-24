import numpy as np
import matplotlib.image as mpimg # for reading images as numpy arrays
import tensorflow as tf
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
        self.epoch_threshold = 10 # arbitrary value for now

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

        Make 3 generate_batch functions for each type (train, test, val)

        Where is epoch incremented?


    """

    def generate_train_batch(self): 
        """
        Takes train numpy array and returns a numpy array containing image data
        of a full batch. Each batch increments current_batch. When current_batch
        is same as number of batches, a reshuffle is applied for the next epoch.
        """

        batch = np.zeros((self.batch_size, 200, 200, 3), dtype=np.float32) # 200x200 RGB images
        start = self.current_train_batch * self.batch_size
        end = (self.current_train_batch+1) * self.batch_size
        index = 0
        for x in range(start, end): # make sure we do not reuse images
            batch[index] = mpimg.imread(self.train_image_array[x])
            index += 1

        #increment batch
        self.current_train_batch += 1 
        # shuffle if needed
        if self.current_train_batch == self.train_number_of_batches:
            np.random.shuffle(self.train_image_array) 

        return batch

    def generate_test_batch(self): 

        batch = np.zeros((self.batch_size, 200, 200, 3), dtype=np.float32) # 200x200 RGB images
        start = self.current_test_batch * self.batch_size
        end = (self.current_test_batch+1) * self.batch_size
        index = 0
        for x in range(start, end): # make sure we do not reuse images
            batch[index] = mpimg.imread(self.test_image_array[x])
            index += 1

        # increment batch
        self.current_test_batch += 1
        # no shuffle 
        return batch

    def generate_val_batch(self): 
        """
        Todo:
        Recognize when no more batches to make?
        """
        # Make np array of 200x200 RGB images, filled with zeros
        batch = np.zeros((self.batch_size, 200, 200, 3), dtype=np.float32) 
        start = self.current_val_batch * self.batch_size
        end = (self.current_val_batch+1) * self.batch_size
        index = 0
        for x in range(start, end): # make sure we do not reuse images
            batch[index] = mpimg.imread(self.val_image_array[x])
            index += 1
            
        #increment batch
        self.current_val_batch += 1 
        # shuffle if needed
        if self.current_val_batch == self.val_test_number_of_batches:
            np.random.shuffle(self.val_image_array) 
            
        return batch
    

if __name__ == '__main__':
    # TESTING
    dataset_directory = str(sys.argv[1])
    dataset = Dataset(dataset_directory, 40)
    train_batch = dataset.generate_train_batch()
    train_batch = dataset.generate_train_batch()
    test_batch = dataset.generate_test_batch()
    test_batch = dataset.generate_test_batch()
    val_batch = dataset.generate_val_batch()
    val_batch = dataset.generate_val_batch()

    print(test_batch[5])
