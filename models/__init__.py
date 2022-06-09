# Imports
from turtle import end_fill
from keras.models import Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Flatten, Dense,Dropout, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def create_model(opt):

    """
    Create a model given the option. Define parameters for the architecture of the CNN model.
     
    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """


    # Load the MobileNetV2 network, ensuring the head FC layer sets are left off
    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))  # pre-trained ImageNet weights
    
    # Construct the (new) head of the model that will be placed on top of the the base model

    headModel = baseModel.output
    
    headModel = AveragePooling2D(pool_size=(7, 7))(headModel)  # filter size (7,7)
    
    headModel = Flatten(name="flatten")(headModel)  # flattens multi-dimensional data into single dimension
    
    headModel = Dense(128, activation="relu")(headModel)  # Fully Connected layer of 128 neurons
    
    headModel = Dropout(0.5)(headModel)  # drop 50% of neurons - make the input values zero
    
    headModel = Dense(2, activation="softmax")(headModel) # Fully Connected layer of 2 neurons. It is the final layer.
    
    # Place the head FC model on top of the base model (this will become the actual model we will train)
    model = Model(inputs = baseModel.input, outputs = headModel)

    # Loop over all layers in the base model and freeze them so they will *not* be updated during the first training process
    # Only the head layer weights will be fine-tuned!
    for layer in baseModel.layers:
        layer.trainable = False

    # Compile our model
    print ("(Info) compiling model...")
    
    # Define optimizer
    optim = Adam(learning_rate=opt.lr, decay=opt.lr/opt.n_epochs, beta_1 = opt.beta1, beta_2 = opt.beta2)
    model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['accuracy'])


    return model
