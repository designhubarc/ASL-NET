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
        self.current_train_batch = 0 # which batch are we on
        self.current_epoch = 0 # current number of cycles through training dataset

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
            image_list += os.path.join(path, os.listdir(path)) # full file path
 
        random.shuffle(image_list)

        return np.array(image_list)


    """
        TO-DO: 

        Make 3 generate_batch functions for each type (train, test, val)

        Track batch number and shuffle when full dataset has been batched and
        moving onto next epoch


    """
    def generate_train_batch(self): 
        """
        Takes train numpy array and returns a numpy array containing image data
        of a full batch 
        """
        batch = np.zeros((self.batch_size, 200, 200, 3), dtype=np.float32) # 200x200 RGB images
        for x in range(self.current_batch*self.batch_size, (self.current_batch+1)*self.batch_size): # make sure we do not reuse images
            batch[x] = mpimg.imread(specific_image_array[x])
            
        self.current_batch += 1 #increment batch
        # shuffle if needed
        return batch
        
        
    def generate_test_batch(self): 

    def generate_val_batch(self): 


if __name__ == '__main__':
    dataset_directory = str(sys.argv[1])
    dataset = Dataset(dataset_directory, 20)
    batch = dataset.generate_batch(dataset.test_image_array)
    print(batch)
    print(len(dataset.train_image_array))