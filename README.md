# ImageClassification

## Introduction to project
The project was done as internship during the Y-Data program in a Tel-Aviv based startup company, focusing on image classification.
The goal was to have binary classification that meets the target function of maximum recall at 90% precision, while detecting anomalies among unseen images.  
#### Note: 
  The data is unavailable due to confidentiality , therefore the files cannot be run.  
  However, the notebook consists of running flow plus results of smaller dataset.  
  The Resuls appearing below are based on the run on the whole dataset as presented at the end of the project.

#### Explanation of the files: <br />

* Main - jupyter notebook including all the steps of the run.
* binary_classification.csv - list of images, used for uploading the files to the dataloader
* config.py - defining values for different parameters
* eval_model.py - calculating metrics on a given dataset running on the trained model
* imports.py - importing different python libraries
* load_data.py - creating dataloaders
* model.py - creating the model, based on pretrained MobileNetV2 plus additional layers
* predict.py - predicting a class based on the trained model
* train_model.py - training the model
* transformer.py - different transformations to images (resize, flip, crop etc.)
* utilities.py - including the procedure generating the csv file

## Description of the solution
### The binary classification is based on the next steps:
  - CNN comprised of pretrained MobileNetV2 (without its activation function and classifier) embedded with additional convolutional layers to reduce number of dimensions.
The network ends FC layer classifying two labels.
  - Prior to the last layer in the classifier, features at lower dimensions were extracted (50 dimensions) in order to cluster them and plot their embedding after PCA on 2/3 dimensions.
![Model_](https://github.com/OsnatMel/ImageClassification/blob/master/Images4Summary/Model_MobileNetV2_Customized.png)

## Results 
  - The network met the target function – 98% recall with 90% precision, running 30 epochs.
  ![PrecisionRecallCurve_TestSet](https://github.com/OsnatMel/ImageClassification/blob/master/Images4Summary/PrecisionRecallCurve_TestSet.png)
  
  - The features' embedding showed a good separation with a dense and homogeneous positive-labeled cluster while the other negative-labeled data-points lie in more sparsely in the embedding space.

  - After training the model, the test dataset was evaluated and its 50 features were reduced to 2 dimensions using PCA.
  A nice separation between the two classes is shown either in after the PCA or by a GMM method.
  ![PCA_2D_TestSet](https://github.com/OsnatMel/ImageClassification/blob/master/Images4Summary/PCA_2D_TestSet.png)
  ![GMM_TestSet](https://github.com/OsnatMel/ImageClassification/blob/master/Images4Summary/GMM_TestSet.png)
  
  - Agglomertive clustering method was able to capture the two classes. (Not shown in README, but appears in the notebook).

 - The advanced goal was to see how the network deals with zero shot learning, i.e. testing images that it wasn't trained on and seeing where they lie in the embedding space.
- On 2d the separation is visible but not sufficient.  
 ![PCA_2D_TestSet_ZeroShot](https://github.com/OsnatMel/ImageClassification/blob/master/Images4Summary/PCA_2D_TestSet_And_ZeroShot.png)

- on 3d it's shown that the valid (green) images lie in a dense area versus the other different invalid images that lie surronding it.
 ![PCA_3D_TestSet_ZeroShot](https://github.com/OsnatMel/ImageClassification/blob/master/Images4Summary/PCA_3D_TestSet_ZeroShot.gif)
