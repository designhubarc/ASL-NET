import dispatcher, image_resizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm # prettier loops
import numpy as np
import matplotlib.image as mpimg # for reading images as numpy arrays when in test mode
import os # used in test mode


# Running Tensorflow 2.1.0

def BuildNetwork(dataset):
    # Create model
    model = Sequential()
    # Images are 64x64, with 3 channels (RGB)

    # 2D convolutional network with 5x5 filter
    model.add(Conv2D(32, kernel_size=(5, 5),
                        input_shape = (64, 64, 3),
                        activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Conv2D(64, kernel_size=(5, 5),
                        activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

    model.add(Dropout(0.4)) # dropout layer

    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))

    # 26 alphabet letters
    model.add(Dense(dataset.classifications))
    model.add(Activation('softmax'))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def TrainAndValidateNetwork(model, dataset):
    # Train and validate network
    while(dataset.current_epoch < dataset.epoch_threshold): # go for some number of epochs
        # train
        for step in tqdm(range(dataset.train_number_of_batches), desc = "Training Model " + "- Epoch " + str(int(dataset.current_epoch+1))): # for each batch in the epoch
            (train_photos, train_labels) = dataset.generate_train_batch() # get next batch of training images
            train_photos = train_photos / 255.0 # standardize RGB values between 0-1

            stats = model.train_on_batch(train_photos, train_labels)
            #print("Epoch " + str(dataset.current_epoch) + " - Loss & Accuracy: ")
            #print(stats)

        # validate
        num_correct = 0
        total_predictions = 0
        for step in tqdm(range(dataset.val_test_number_of_batches), desc = "Validating Model"): # lets load all in memory then evaluate in test mode for validation
                (val_photos, val_labels) = dataset.generate_val_batch() # get next batch of val images
                val_photos = val_photos / 255.0 # standardize RGB values between 0-1
                predictions = model.predict_on_batch(val_photos)

                # go through and see which predictions are correct
                for i in range(batch_size):
                    if val_labels[i] == np.argmax(predictions[i]):
                        num_correct += 1

                total_predictions += batch_size # we predict batch size at a time

        print("Correct = " + str(num_correct) + ", Total Predictions = " + str(total_predictions) + ", Validation Accuracy = " + str(float(num_correct)/float(total_predictions)))
    return model

# later will be renamed to TestTrainingNetwork, to specify this happens during training
def TestNetwork(model, dataset):
    # Test set
    num_correct = 0
    total_predictions = 0
    for step in tqdm(range(dataset.val_test_number_of_batches), desc = "Testing Model"):
        (test_photos, test_labels) = dataset.generate_test_batch()
        test_photos = test_photos / 255.0 # standardize RGB values
        predictions = model.predict_on_batch(test_photos)

        # go through and see which predictions are correct
        for i in range(batch_size):
            if test_labels[i] == np.argmax(predictions[i]):
                num_correct += 1

        total_predictions += batch_size # we predict batch size at a time

    print("Correct = " + str(num_correct) + ", Total Predictions = " + str(total_predictions) + ", Validation Accuracy = " + str(float(num_correct)/float(total_predictions)))


if __name__ == "__main__":

    # get directory and batch size
    dataset_directory = str(sys.argv[1]) # in train mode, it will be a path to a dir with train/ test/ val/.. in test mode it is a directory of images (images are expected to be set to 64x64 rgb)
    batch_size = int(sys.argv[2]) 
    file_path = str(sys.argv[3]) # complete file path to save the trained model. File path must be to an hdf5 file.ex: C:\Users\Bob\my_model.hdf5

    if (len(sys.argv) == 5 and str(sys.argv[4]) == "test"):
        model = tf.keras.models.load_model(file_path) # load trained model
        image_resizer.Resize(dataset_directory, dataset_directory, 64, 64) # resize images to test them
        images = os.listdir(dataset_directory) # list of all images
        numOfBatches = len(images) / batch_size if len(images) % batch_size == 0 else len(images) / batch_size + 1
        
        # go through each batch
        for j in range(int(numOfBatches)):
            if j == int(numOfBatches)-1:
                batch_images = np.zeros((len(images) - j*batch_size, 64, 64, 3), dtype=np.float32) # 200x200 RGB images
                for i in range(len(images) - j*batch_size):
                    batch_images[i] = mpimg.imread(os.path.join(dataset_directory,images[j*batch_size + i])) # get the images 1 by 1
            else:
                batch_images = np.zeros((batch_size, 64, 64, 3), dtype=np.float32) # 200x200 RGB images
                for i in range(batch_size):
                    batch_images[i] = mpimg.imread(os.path.join(dataset_directory,images[j*batch_size + i])) # get the images 1 by 1
            
            predictions = model.predict_on_batch(batch_images) # get predictions
            
            # print the results
            i = 0
            for p in predictions:
                print('Image Name: ' + images[j*batch_size + i], end=' ') # for now we will just print the predictions
                print('Prediction: ' + chr(65 + np.argmax(p)))
                i += 1
    else:
        dataset = dispatcher.Dataset(dataset_directory, batch_size) # handle on our dataset

        model = BuildNetwork(dataset) # build network
        model = TrainAndValidateNetwork(model, dataset) # training and validation for each epoch
        TestNetwork(model, dataset) # run the network on the test set

        model.save(file_path) # save the model
