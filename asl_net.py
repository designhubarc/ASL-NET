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
import fitz # for pdf results


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
        output_pdf_path = os.path.join(os.path.dirname(os.getcwd()),'aslnet_results.pdf') # full path to where we hold the output pdf.. (later on will need names to account for potentially multiple users)
        output_pdf = fitz.open() # pdf handle
        cur_page = output_pdf.newPage() # insert first page
        
        # predefine rectangles for photos per page (15 per page).. later on maybe locations will want to be dynamic but for now we will keep reporting assumed to be 64x64 images to have a compact report
        image_rectangles = [fitz.Rect(67, 52, 131, 116), fitz.Rect(265, 52, 329, 116), fitz.Rect(463, 52.2, 527, 116),
                      fitz.Rect(67, 220, 131, 284), fitz.Rect(265, 220, 329, 284), fitz.Rect(463, 220, 527, 284),
                      fitz.Rect(67, 388, 131, 452), fitz.Rect(265, 388, 329, 452), fitz.Rect(463, 388, 527, 452),
                      fitz.Rect(67, 556, 131, 620), fitz.Rect(265, 556, 329, 620), fitz.Rect(463, 556, 527, 620),
                      fitz.Rect(67, 724, 131, 788), fitz.Rect(265, 724, 329, 788), fitz.Rect(463, 724, 527, 788)]
        
        # descriptions on page will be relative to the image placements (also 15 per page)
        results_rectangles = [fitz.Rect(image_rectangles[0].x0-32, image_rectangles[0].y1+15, image_rectangles[0].x1+32, image_rectangles[0].y1+45),
                              fitz.Rect(image_rectangles[1].x0-32, image_rectangles[1].y1+15, image_rectangles[1].x1+32, image_rectangles[1].y1+45),
                              fitz.Rect(image_rectangles[2].x0-32, image_rectangles[2].y1+15, image_rectangles[2].x1+32, image_rectangles[2].y1+45),
                              fitz.Rect(image_rectangles[3].x0-32, image_rectangles[3].y1+15, image_rectangles[3].x1+32, image_rectangles[3].y1+45),
                              fitz.Rect(image_rectangles[4].x0-32, image_rectangles[4].y1+15, image_rectangles[4].x1+32, image_rectangles[4].y1+45),
                              fitz.Rect(image_rectangles[5].x0-32, image_rectangles[5].y1+15, image_rectangles[5].x1+32, image_rectangles[5].y1+45),
                              fitz.Rect(image_rectangles[6].x0-32, image_rectangles[6].y1+15, image_rectangles[6].x1+32, image_rectangles[6].y1+45),
                              fitz.Rect(image_rectangles[7].x0-32, image_rectangles[7].y1+15, image_rectangles[7].x1+32, image_rectangles[7].y1+45),
                              fitz.Rect(image_rectangles[8].x0-32, image_rectangles[8].y1+15, image_rectangles[8].x1+32, image_rectangles[8].y1+45),
                              fitz.Rect(image_rectangles[9].x0-32, image_rectangles[9].y1+15, image_rectangles[9].x1+32, image_rectangles[9].y1+45),
                              fitz.Rect(image_rectangles[10].x0-32, image_rectangles[10].y1+15, image_rectangles[10].x1+32, image_rectangles[10].y1+45),
                              fitz.Rect(image_rectangles[11].x0-32, image_rectangles[11].y1+15, image_rectangles[11].x1+32, image_rectangles[11].y1+45),
                              fitz.Rect(image_rectangles[12].x0-32, image_rectangles[12].y1+15, image_rectangles[12].x1+32, image_rectangles[12].y1+45),
                              fitz.Rect(image_rectangles[13].x0-32, image_rectangles[13].y1+15, image_rectangles[13].x1+32, image_rectangles[13].y1+45),
                              fitz.Rect(image_rectangles[14].x0-32, image_rectangles[14].y1+15, image_rectangles[14].x1+32, image_rectangles[14].y1+45)]
                              
        num_pics_on_page = 0 # current number of photos on the page
        
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
            for i in range(len(predictions)):
                if num_pics_on_page < len(image_rectangles): # add to current page if there is space
                    cur_page.insertImage(image_rectangles[num_pics_on_page], filename=os.path.join(dataset_directory,images[j*batch_size + i]))
                    cur_page.insertTextbox(results_rectangles[num_pics_on_page], 'Image: ' + images[j*batch_size + i] + '\n' + 'Prediction: ' + chr(65 + np.argmax(predictions[i])), align=1)
                else:
                    cur_page = output_pdf.newPage() # create a new page
                    num_pics_on_page = 0 # fresh page
                    cur_page.insertImage(image_rectangles[num_pics_on_page], filename=os.path.join(dataset_directory,images[j*batch_size + i])) # add to the new page
                    cur_page.insertTextbox(results_rectangles[num_pics_on_page], 'Image: ' + images[j*batch_size + i] + '\n' + 'Prediction: ' + chr(65 + np.argmax(predictions[i])), align=1)
                    
                print('Image Name: ' + images[j*batch_size + i], end=' ') # print statements for quick visual that its working
                print('Prediction: ' + chr(65 + np.argmax(predictions[i])))
                
                num_pics_on_page += 1 # increment images on page
        
        output_pdf.save(output_pdf_path) # save pdf one dir back
        
    else:
        dataset = dispatcher.Dataset(dataset_directory, batch_size) # handle on our dataset

        model = BuildNetwork(dataset) # build network
        model = TrainAndValidateNetwork(model, dataset) # training and validation for each epoch
        TestNetwork(model, dataset) # run the network on the test set

        model.save(file_path) # save the model
