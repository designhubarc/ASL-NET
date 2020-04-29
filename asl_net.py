import dispatcher
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import sys


# Running Tensorflow 2.0.0

if __name__ == "__main__":
    dataset_directory = str(sys.argv[1])
    batch_size = int(sys.argv[2])

    # handle on our dataset
    dataset = dispatcher.Dataset(dataset_directory, batch_size) 

    # Get test images and labels
    (test_images, test_labels) = dataset.generate_test_batch()
    # Get train images and labels
    (train_images, train_labels) = dataset.generate_train_batch()
    # Get val images and labels
    (val_images, val_labels) = dataset.generate_val_batch()


    # scale for neural network
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    val_images = val_images / 255.0

    # Successfully displays images from numpy array train_images with labels
    """
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(dispatcher.CATEGORIES[train_labels[i]])
    plt.show()
    """

    # Create model
    model = keras.Sequential([
        keras.layers.Conv2D(40, (3, 3), input_shape = (3, 200, 200))
        keras.layers.Flatten(input_shape=(3, 200, 200)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(26) # 26 alphabet letters
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    model.fit(train_images, train_labels, epochs=1)