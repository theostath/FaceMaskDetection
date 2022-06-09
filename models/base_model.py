import os
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from sklearn.metrics import classification_report


class BaseModel():
    """This class is an abstract base class for models.
    To create a subclass, you need to implement the function:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
    """

    def __init__(self, opt):
        """
        Initialize the BaseModel class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        """
    
        self.opt = opt
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir (and load for test)


    def setup(self, opt, model):
        """
        Print network if --verbose and load network if we run test.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            model -- the CNN we created from [models] / __init__.py
        """

        if opt.verbose:
            self.print_network(opt, model)

        if not self.isTrain:
            self.load_network(opt)

       
    def load_network(self, opt):
        """
        Load network for testing model.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        if not self.isTrain:   
            if opt.suffix:
                suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
                opt.name = opt.name + suffix
                        
            load_dir = self.save_dir

            load_filename = 'model-%s_%s.h5' % (opt.n_epochs, opt.name)

            load_path = os.path.join(load_dir, load_filename)

            # If we have trained the model once we don't need to train it again. We can simply load the saved model from the previous step.
            model = keras.models.load_model(load_path)

            return model


    def print_network(self, opt, model):
        """
        Print the total number of parameters in the network and network architecture.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            model -- the CNN we created from [models] / __init__.py
        """
        print('---------- Network initialized -------------')

        model.summary()
        
        print('-----------------------------------------------')

        # Save to .txt    
        from contextlib import redirect_stdout

        file_dir = os.path.join(opt.checkpoints_dir, opt.name)
        file_name = 'model_summary.txt'
        file_path = os.path.join(file_dir, file_name)
        with open(file_path, 'w') as f:
            with redirect_stdout(f):
                model.summary()


    def train_network(self, opt, model, train_generator, trainX, testX, testY) :
        """
        Train the network and evaluate it (figure). Also, save txt with training logs.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            model -- the CNN we created from [models] / __init__.py
            train_generator -- Generator for train images (from train directory)
            testX -- test set images from original data
            testY -- test set labels from original data
        """

        print("(Info) training head of network...")

        history = model.fit(train_generator,    # providing batches of mutated image data
                                steps_per_epoch = len(trainX) // opt.batch_size,
                                epochs = opt.n_epochs,
                                validation_data = (testX, testY),
                                validation_steps = len(testX) // opt.batch_size)

        # Evaluate the model. Accuracy and loss figures.
        fig = plt.figure()
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')
        plt.xlabel('Epoch #')
        plt.ylabel('Accuracy/Loss')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.show()
        save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        fig_filename = "accuracy_figure_%s.png" % (opt.name)
        fig_path = os.path.join(save_dir, fig_filename)
        fig.savefig(fig_path)
    
        
        # Save the training logs in a txt file.
        save_filename = 'train_logs_%s_%s.txt' % (opt.n_epochs, opt.name)
        save_path = os.path.join(save_dir, save_filename)

        # Get the values
        losses = history.history['loss']     
        accuracy = history.history['accuracy']        
        val_loss = history.history['val_loss']
        val_accuracy = history.history['val_accuracy']

        message = "\n"
        for epoch in range(opt.n_epochs):
            message+= 'epoch: %d - iters: %d - loss: %.4f - accuracy: %.4f - val_loss: %.4f - val_accuracy: %.4f\n' % ((epoch+1), (len(trainX) // opt.batch_size), losses[epoch], accuracy[epoch], val_loss[epoch], val_accuracy[epoch])

        # Create a logging file to store training losses
        with open(save_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
            log_file.write('%s\n' % message)  # save the message


    def evaluate_network(self, opt, model, testX, testY, labels):
        """
        Evaluate tha trained network on testing set.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            model -- the CNN we created from [models] / __init__.py and trained from train.py
            testX -- test set images from original data
            testY -- test set labels from original data
        """

        # Make predictions on test set
        print("(Info) evaluating network...")
        preds = model.predict(testX, batch_size = opt.batch_size)

        # For each image in the testing set we need to find the index of the
        # label with corresponding largest predicted probability
        preds = np.argmax(preds, axis=1)

        print(classification_report(testY.argmax(axis = 1), preds, target_names = ['with_mask', 'without_mask']))


    def test_network(self, opt, model):
        """
        Test the pretrained network.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
            model -- the CNN we created from [models] / __init__.py and trained from train.py
        """

        # Define rectangle size (where the message mask/without mask is shown).
        rect_size = 4

        cap = cv2.VideoCapture(0) # Read video from first camera or webcam frame by frame.

        # Haar cascade classifier is used for frontal face detection from the video capture of the webcam. (Change this to your own path!)
        haarcascade = cv2.CascadeClassifier('/home/teo/anaconda3/envs/DataScienceProjects/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml')

        while True:
            # Read Video Capture from webcam to frames.
            (rval, frame) = cap.read() # rval is a Boolean value indicating if a frame was read correctly.

            # Flip frames.
            # second argument (flip code): 1 means flipping around y-axis
            frame = cv2.flip(frame,1) 

            # Rectangle size based on the frame size (// is floor division)
            rerect_size = cv2.resize(frame, ((frame.shape[1] // rect_size), (frame.shape[0] // rect_size)))

            # Detect face in the rectangle. Returns 4 values (x, y, w, h).
            faces = haarcascade.detectMultiScale(rerect_size)
    
            for f in faces:
                # Resize (x, y, w, h)
                (x, y, w, h) = [v * rect_size for v in f] 
        
                # Define the place in the frame where the face is.
                face_img = frame[y:y+h, x:x+w]

                # Preprocess the face_image the same way we did during training.
                # Resize the face image
                rerect_sized = cv2.resize(face_img,(224,224))
                # Normalize pixel values in range [0,1]
                normalized = rerect_sized/255.0
                # Reshape the normalized face image to feed it into the pre-trained model.
                reshaped = np.reshape(normalized,(1,224,224,3))
                reshaped = np.vstack([reshaped])
        
                # Pass the face through the model to determine if the face has a mask or not
                (mask, withoutMask) = model.predict(reshaped)[0]

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output frame
                if max(mask, withoutMask) > opt.confidence:    # confidence level of network that detects face mask
                    frame = cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                    frame = cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            cv2.imshow('LIVE', frame) # LIVE = window name
            key = cv2.waitKey(20)
    
            if key == 27: # ESC key
                break

        # Release the video capture
        cap.release()

        # Close all the frames
        cv2.destroyAllWindows()