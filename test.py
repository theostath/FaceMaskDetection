# Import Classes
from models.base_model import BaseModel     # to load model

from options.test_options import TestOptions  # to import the parser for the train options

# Get testing options

opt = TestOptions().parse()

# Load model

model = BaseModel(opt).load_network(opt)

# Test the model in live feed from webcam

BaseModel(opt).test_network(opt, model)