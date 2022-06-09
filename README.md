# FaceMaskDetection
Real-time face mask detection using the webcam. 

Date: 9 June 2022

## Prerequisites
- Linux or Windows
- Python 3
- Anaconda

### Getting started

- Clone this repo:
```bash
git clone https://github.com/theostath/FaceMaskDetection FaceMask
cd FaceMask
```

- Install the required dependencies.
  For pip users, please type the command `pip install -r requirements.txt`.
  
  For Conda users,  you can create a new Conda environment using `conda env create -f environment.yml`.

### Dataset

Download the dataset from this link:
https://data-flair.training/blogs/download-face-mask-data/

Unzip and put the data in file 'FaceMask/face-mask-dataset' like this:
```bash
└───FaceMask
    └───face-mask-dataset
        ├───test
        │   ├───without_mask
        │   └───with_mask
        └───train
            ├───without_mask
            └───with_mask
```

### Training and Test
Train the model:
```bash
python train.py --name MaskDetect --verbose
```

For more options you can see the parser created in [options]/base_options.py or train_option.py.

Test the model:
```bash
python test.py --name MaskDetect
```

For more options you can see the parser created in [options]/base_options.py or test_option.py.

When you run test.py a window will open that detects whether the persons in the frame wear a mask or not.
