# ImageClassification

Binary classification of images as followed:

CNN on pretrained network called MobileNetV2 plus additional layers in order to reduce number of dimensions.
The network classifier end with two classes, but features are extracted at lower dimension prior to the classification.

The features are used for clustering and visualized on 2d\3d after PCA.