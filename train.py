# Imports
import os

# Import Classes
from options.train_options import TrainOptions  # to import the parser for the train options
from models import create_model     # to create CNN model
from data.base_dataset import DataLoader    # to define data generators
from models.base_model import BaseModel     # to print network architecture & train model


# Get training options

opt = TrainOptions().parse()

# Load images from --dataroot

images, labels = DataLoader().load_data(opt)

# Build the model 

model = create_model(opt)

# Print network if --verbose

BaseModel(opt).setup(opt, model)

# Image data Generation / Augmentation 

train_generator, trainX, testX, testY = DataLoader().data_generators(opt, images, labels)

# Train the model

BaseModel(opt).train_network(opt, model, train_generator, trainX, testX, testY)

# Save model in .h5 format

save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
save_filename = 'model-%s_%s.h5' % (opt.n_epochs, opt.name)
save_path = os.path.join(save_dir, save_filename)
model.save(save_path) # format .h5 for the saved file.

# Evaluate the network

BaseModel(opt).evaluate_network(opt,model,testX,testY, labels)