# ImageClassification

Introduction to project
The project was done as internship during the Y-Data program in a Tel-Aviv based startup company, focusing on image classification.
The goal was to have binary classification that meets the target function of maximum recall at 90% precision, while detecting anomalies among unseen images.

Description of the solution
The binary classification is based on the next steps:
•	CNN comprised of pretrained MobileNetV2 (without its activation function and classifier) embedded with additional convolutional layers to reduce number of dimensions.
The network ends FC layer classifying two labels.
•	Prior to the last layer in the classifier, features at lower dimensions were extracted (50 dimensions) in order to cluster them and plot their embedding after PCA on 2/3 dimensions.

Results 
•	The network met the target function – 98% recall with 90% precision, running 30 epochs.
•	The features' embedding showed a good separation with a dense and homogeneous positive-labeled cluster while the other negative-labeled data-points lie in more sparsely in the embedding space.
Agglomertive clustering method was able to capture the two classes. 
