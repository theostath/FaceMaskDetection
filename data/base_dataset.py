import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical


class DataLoader():
    """
    Load the dataset to DataFrames and manage them.
    Create and use custom Data Loaders for train, validate and test data.
    """
    
    def __init__(self) -> None:
        pass

    
    def load_data(self, opt):
        """
        Load and pre-process our training data.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        # Grab the list of images in our dataset directory, then initialize
        # the list of data (i.e., images) and class images
        print("(Info) loading images...")
        ImagePaths = list(paths.list_images(os.path.join(opt.dataroot,"train")))    # train directory

        # Initialize
        data = []
        labels = []

        # Loop over the image paths
        for ImagePath in ImagePaths:

            # Extract the class label from the filename (with mask / without mask)
            label = ImagePath.split(os.path.sep)[-2]

            # Load the input image, resize it to (224x224) and preprocess it
            image = load_img(ImagePath, target_size = (224, 224))
            image = img_to_array(image)
            image = preprocess_input(image) # preprocess for MobileNetV2 neural network (scale pixel intensities to the range [-1,1])

            # Update the data and labels lists, respectively
            data.append(image)
            labels.append(label)
        
        # Convert the data and labels to NumPy arrays
        data = np.array(data, dtype="float32")
        labels = np.array(labels)

        # Perform one-hot encoding on the labels (two-value arrays where only on value is equal to 1 and the other is 0)
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        labels = to_categorical(labels)

        return data, labels



    def data_generators(self, opt, data, labels):
        """
        Training and validation (and test) data generator used for data augmentation.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        # Partition the data (and their labels) into training and testing splits using 80% of
        # the data for training and the remaining 20% for testing (random state is to ensure reproducibility)
        (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)


        # The choice for horizontal flip is:
        flip = not opt.no_flip

        # Define the train data generator (real-time data augmentation --> to help generalization of the network).
        train_datagen = ImageDataGenerator(rotation_range = 20,
                                           width_shift_range = 0.2,
                                           height_shift_range = 0.2,
                                           shear_range = 0.15,
                                            zoom_range = 0.15,
                                            horizontal_flip = flip,
                                            fill_mode = 'nearest')

        # Give the directory of the images to the data generator to augment the pictures from train directory.
        train_generator = train_datagen.flow(trainX, trainY, batch_size = opt.batch_size)

        return train_generator, trainX, testX, testY