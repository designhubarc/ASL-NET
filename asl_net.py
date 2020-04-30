import dispatcher
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import sys


# Running Tensorflow 2.0.0

if __name__ == "__main__":
    dataset_directory = str(sys.argv[1])
    batch_size = int(sys.argv[2])

    # handle on our dataset
    dataset = dispatcher.Dataset(dataset_directory, batch_size) 

    # Get test images and labels
    (test_x, test_y) = dataset.generate_test_batch()
    # Get train images and labels
    (train_x, train_y) = dataset.generate_train_batch()
    # Get val images and labels
    (val_x, val_y) = dataset.generate_val_batch()


    # scale/normalize RGB values for neural network
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    val_x = val_x / 255.0

    # Successfully displays images from numpy array train_x with labels
    """
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_x[i], cmap=plt.cm.binary)
        plt.xlabel(dispatcher.CATEGORIES[train_y[i]])
    plt.show()
    """

    # Create model
    model = Sequential()
    # Images are 200x200, with 3 channels (RGB)

    # 2D convolutional network with 5x5 filter
    model.add(Conv2D(64, (5, 5), input_shape = (200, 200, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(batch_size))

    # 26 alphabet letters
    model.add(Dense(26))
    model.add(Activation('sigmoid'))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer='adam',
                  metrics=['sparse_categorical_accuracy'])
    
    model.fit(train_x, train_y, 
              batch_size=dataset.batch_size, 
              validation_data=(val_x, val_y))
