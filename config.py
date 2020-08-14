from imports import *

#path
ROOT_DIR = "Training_Data/"
FNAME = 'binary_classification.csv'
TB_DIR = 'runs/Image_Classification_30_Epochs'
BUCKETNAME = 'healthyvalidinvalid' 

#utils
SEED = 42 #0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#training parameters
VAL_SIZE = 0.2
TEST_SIZE = 0.1

CLASSES = ['invalid','valid']
NUM_CLASSES = len(CLASSES)

BATCH_SIZE = 32
NUM_EPOCHS = 30 

## network paramters for documentation
COMMENT = 'Image classification 30 epochs \r\n Training model - customized network based on MobileNetV2 plus additional layers'
LOSS = 'NLLLoss'
OPTIM = 'Adam'


'''
This file includes configuration and global parameters:

-- ROOT_DIR : data root directory
-- FNAME : csv file name which includes all data labels and paths
-- TB_DIR : tensor board directory to save all outputs

-- SEED : np and pytorch seed

-- VAL_SIZE : validation set ratio
-- TEST_SIZE : test set ratio
-- CLASSES : class name tuple
-- BATCH_SIZE : batch size
-- NUM_EPOCHS : number off epochs

-- COMMENT : an indicative description of the model
-- LOSS : criterion used in the training
-- OPTIM : optimizer used in the training

'''