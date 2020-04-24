import dispatcher

if __name__ == "__main__":
    dataset_directory = str(sys.argv[1])
    batch_size = int(sys.argv[2])

    data_set = dispatcher.Dataset(dataset_directory, batch_size) # handle on our dataset

    # build/train/test a 2D CNN
